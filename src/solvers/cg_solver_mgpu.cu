/**
 * @file cg_solver_mgpu.cu
 * @brief Multi-GPU CG solver with MPI+NCCL - Step 1: Skeleton
 *
 * Architecture: 1 MPI rank = 1 GPU
 * This file builds incrementally:
 *   Step 1: MPI+NCCL init/cleanup (current)
 *   Step 2: CUDA kernels
 *   Step 3: Distributed dot product
 *   Step 4: CG loop
 *   Step 5: Optimization
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-06
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "solvers/cg_solver_mgpu.h"
#include "spmv.h"
#include "spmv_csr.h"
#include "io.h"

// Forward declare stencil multi-GPU kernel
__global__ void stencil5_csr_direct_mgpu_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y_local,
    int row_offset,
    int local_rows,
    int grid_size,
    double alpha
);

// Helper to partition CSR matrix rows
static void partition_csr_rows(CSRMatrix* full_csr, int row_offset, int n_local,
                                int** local_row_ptr, int** local_col_idx,
                                double** local_values, int* local_nnz) {
    // Extract row segment [row_offset : row_offset + n_local]
    int start_nnz = full_csr->row_ptr[row_offset];
    int end_nnz = full_csr->row_ptr[row_offset + n_local];
    *local_nnz = end_nnz - start_nnz;

    // Allocate local arrays
    *local_row_ptr = (int*)malloc((n_local + 1) * sizeof(int));
    *local_col_idx = (int*)malloc(*local_nnz * sizeof(int));
    *local_values = (double*)malloc(*local_nnz * sizeof(double));

    // Copy and adjust row_ptr offsets
    for (int i = 0; i <= n_local; i++) {
        (*local_row_ptr)[i] = full_csr->row_ptr[row_offset + i] - start_nnz;
    }

    // Copy col_idx and values
    memcpy(*local_col_idx, &full_csr->col_indices[start_nnz], *local_nnz * sizeof(int));
    memcpy(*local_values, &full_csr->values[start_nnz], *local_nnz * sizeof(double));
}

// NCCL error checking
#define CHECK_NCCL(call) \
{ \
    ncclResult_t result = (call); \
    if (result != ncclSuccess) { \
        fprintf(stderr, "[Rank %d] NCCL error at %s:%d: %s\n", \
                g_mpi_rank, __FILE__, __LINE__, ncclGetErrorString(result)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}

// Global MPI rank for error messages
static int g_mpi_rank = -1;

// ========== CUDA Kernels (Step 2) ==========

/**
 * @brief atomicAdd for double (CC < 6.0 compatibility)
 */
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/**
 * @brief Local dot product kernel (partial sum)
 * Computes local contribution: sum(x[i] * y[i]) for i in [0, n_local)
 */
__global__ void local_dot_kernel(const double* x, const double* y,
                                  double* partial_sum, int n_local) {
    __shared__ double sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Local accumulation
    double sum = 0.0;
    if (i < n_local) {
        sum = x[i] * y[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAddDouble(partial_sum, sdata[0]);
    }
}

/**
 * @brief AXPY: y = a*x + y
 */
__global__ void axpy_kernel(double a, const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

/**
 * @brief AXPBY: y = a*x + b*y
 */
__global__ void axpby_kernel(double a, const double* x, double b, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + b * y[i];
    }
}

/**
 * @brief Vector copy: y = x
 */
__global__ void copy_kernel(const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}

/**
 * @brief Helper: Compute local dot product contribution
 */
static double compute_local_dot(const double* d_x, const double* d_y, int n_local,
                                 double* d_work, cudaStream_t stream) {
    // Zero work buffer
    CUDA_CHECK(cudaMemsetAsync(d_work, 0, sizeof(double), stream));

    // Launch reduction kernel
    int threads = 256;
    int blocks = (n_local + threads - 1) / threads;
    local_dot_kernel<<<blocks, threads, 0, stream>>>(d_x, d_y, d_work, n_local);

    // Copy result to host
    double h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_work, sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return h_result;
}

/**
 * @brief Multi-GPU CG solver with MPI+NCCL
 *
 * Algorithm: Conjugate Gradient with row-band matrix decomposition
 * - Each rank owns a row segment of matrix A
 * - Full vector replication (x, r, p, Ap on each GPU)
 * - NCCL AllReduce for global dot products
 * - No AllGather needed with full replication approach
 *
 * Tested: 1 GPU (local), ready for 2-4 GPUs (VastAI)
 */
int cg_solve_mgpu(SpmvOperator* spmv_op,
                  MatrixData* mat,
                  const double* b,
                  double* x,
                  CGConfigMultiGPU config,
                  CGStatsMultiGPU* stats) {

    // ========== MPI Initialization ==========
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    g_mpi_rank = rank;

    // ========== GPU Assignment ==========
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int local_gpu = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(local_gpu));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, local_gpu));

    if (rank == 0) {
        printf("\n========================================\n");
        printf("Multi-GPU CG Solver (MPI+NCCL)\n");
        printf("========================================\n");
        printf("MPI ranks: %d\n", world_size);
        printf("GPUs per node: %d\n", num_gpus);
        printf("Problem size: %d unknowns\n", mat->rows);
        printf("Max iterations: %d\n", config.max_iters);
        printf("Tolerance: %.1e\n", config.tolerance);
        printf("========================================\n\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("[Rank %d] GPU %d: %s (CC %d.%d)\n",
           rank, local_gpu, prop.name, prop.major, prop.minor);
    MPI_Barrier(MPI_COMM_WORLD);

    // ========== NCCL Initialization (skip if single GPU) ==========
    ncclComm_t nccl_comm = NULL;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (world_size > 1) {
        ncclUniqueId nccl_id;
        if (rank == 0) {
            CHECK_NCCL(ncclGetUniqueId(&nccl_id));
        }
        MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
        CHECK_NCCL(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));

        if (rank == 0 && config.verbose >= 2) {
            printf("NCCL initialized (%d GPUs)\n\n", world_size);
        }
    } else {
        if (rank == 0 && config.verbose >= 2) {
            printf("Single GPU mode (NCCL skipped)\n\n");
        }
    }

    // ========== Domain Decomposition ==========
    int n = mat->rows;
    int base_rows = n / world_size;
    int remainder = n % world_size;
    int n_local = base_rows + (rank < remainder ? 1 : 0);
    int row_offset = rank * base_rows + (rank < remainder ? rank : remainder);

    if (config.verbose >= 2) {
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Rows: [%d:%d) (%d rows)\n",
               rank, row_offset, row_offset + n_local, n_local);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ========== Matrix Partitioning with MPI Scatter ==========
    int *h_local_row_ptr = NULL, *h_local_col_idx = NULL;
    double *h_local_values = NULL;
    int local_nnz = 0;

    // Rank 0: Build CSR and partition for all ranks
    if (rank == 0) {
        if (config.verbose >= 1) {
            printf("\nBuilding and partitioning CSR matrix...\n");
        }
        build_csr_struct(mat);
    }

    // Calculate partition sizes
    int* all_n_local = (int*)malloc(world_size * sizeof(int));
    int* all_row_offsets = (int*)malloc(world_size * sizeof(int));
    int* all_local_nnz = (int*)malloc(world_size * sizeof(int));

    for (int r = 0; r < world_size; r++) {
        all_n_local[r] = base_rows + (r < remainder ? 1 : 0);
        all_row_offsets[r] = r * base_rows + (r < remainder ? r : remainder);
    }

    // Rank 0: Partition and scatter
    if (rank == 0) {
        // Partition for all ranks
        for (int r = 0; r < world_size; r++) {
            int *tmp_row_ptr, *tmp_col_idx;
            double *tmp_values;
            partition_csr_rows(&csr_mat, all_row_offsets[r], all_n_local[r],
                               &tmp_row_ptr, &tmp_col_idx, &tmp_values, &all_local_nnz[r]);

            if (r == 0) {
                // Keep rank 0's partition
                h_local_row_ptr = tmp_row_ptr;
                h_local_col_idx = tmp_col_idx;
                h_local_values = tmp_values;
                local_nnz = all_local_nnz[0];
            } else {
                // Send to other ranks
                MPI_Send(&all_local_nnz[r], 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(tmp_row_ptr, all_n_local[r] + 1, MPI_INT, r, 1, MPI_COMM_WORLD);
                MPI_Send(tmp_col_idx, all_local_nnz[r], MPI_INT, r, 2, MPI_COMM_WORLD);
                MPI_Send(tmp_values, all_local_nnz[r], MPI_DOUBLE, r, 3, MPI_COMM_WORLD);

                free(tmp_row_ptr);
                free(tmp_col_idx);
                free(tmp_values);
            }
        }
    } else {
        // Receive partition
        MPI_Recv(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        h_local_row_ptr = (int*)malloc((n_local + 1) * sizeof(int));
        h_local_col_idx = (int*)malloc(local_nnz * sizeof(int));
        h_local_values = (double*)malloc(local_nnz * sizeof(double));

        MPI_Recv(h_local_row_ptr, n_local + 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_local_col_idx, local_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_local_values, local_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    free(all_n_local);
    free(all_row_offsets);
    free(all_local_nnz);

    if (config.verbose >= 2) {
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] CSR partition: %d rows, %d nnz\n", rank, n_local, local_nnz);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Transfer local CSR to device
    int *d_local_row_ptr, *d_local_col_idx;
    double *d_local_values;

    CUDA_CHECK(cudaMalloc(&d_local_row_ptr, (n_local + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_local_col_idx, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_local_values, local_nnz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_local_row_ptr, h_local_row_ptr,
                          (n_local + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_col_idx, h_local_col_idx,
                          local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_values, h_local_values,
                          local_nnz * sizeof(double), cudaMemcpyHostToDevice));

    // ========== Memory Allocation ==========
    // Full vectors on each GPU (required for stencil neighbors)
    double *d_x, *d_r, *d_p, *d_Ap;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n * sizeof(double)));

    // Initialize to zero (critical for multi-GPU)
    CUDA_CHECK(cudaMemset(d_r, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Ap, 0, n * sizeof(double)));

    // Local output buffers for SpMV (size = n_local)
    double *d_Ap_local;
    CUDA_CHECK(cudaMalloc(&d_Ap_local, n_local * sizeof(double)));

    // Work buffer for dot products
    double* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(double)));

    // RHS vector (full)
    double* d_b;
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize x = x0 (full)
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));

    double mem_mb = (5.0 * n + n_local) * sizeof(double) / 1e6;
    if (config.verbose >= 2) {
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Memory: %.2f MB (%.2f MB full vectors, %.2f MB local)\n",
               rank, mem_mb, (4.0 * n * sizeof(double)) / 1e6,
               n_local * sizeof(double) / 1e6);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ========== CG Algorithm ==========
    if (rank == 0 && config.verbose >= 1) {
        printf("\nStarting CG iterations...\n");
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    int threads = 256;
    int blocks_local = (n_local + threads - 1) / threads;  // For local BLAS1 operations
    int threads_spmv = 256;
    int blocks_spmv = (n_local + threads_spmv - 1) / threads_spmv;

    // Compute initial residual: r = b - A*x
    // SpMV local: Ap_local = A_local * x (using multi-GPU kernel)
    stencil5_csr_direct_mgpu_kernel<<<blocks_spmv, threads_spmv, 0, stream>>>(
        d_local_row_ptr, d_local_col_idx, d_local_values,
        d_x, d_Ap_local,
        row_offset, n_local, mat->grid_size, 1.0
    );
    CUDA_CHECK(cudaGetLastError());

    // AllGather Ap segments (NCCL GPU-direct)
    if (world_size > 1) {
        // Use NCCL Broadcast pattern to implement AllGather
        CHECK_NCCL(ncclGroupStart());

        // Each rank broadcasts its local segment
        for (int r = 0; r < world_size; r++) {
            int r_base_rows = n / world_size;
            int r_remainder = n % world_size;
            int r_n_local = r_base_rows + (r < r_remainder ? 1 : 0);
            int r_offset = r * r_base_rows + (r < r_remainder ? r : r_remainder);

            if (r == rank) {
                // Broadcast my segment from d_Ap_local to d_Ap[r_offset]
                CHECK_NCCL(ncclBroadcast(d_Ap_local, d_Ap + r_offset,
                                         r_n_local, ncclDouble, r, nccl_comm, stream));
            } else {
                // Receive rank r's segment into d_Ap[r_offset]
                CHECK_NCCL(ncclBroadcast(d_Ap + r_offset, d_Ap + r_offset,
                                         r_n_local, ncclDouble, r, nccl_comm, stream));
            }
        }

        CHECK_NCCL(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));  // Wait for AllGather to complete
    } else {
        // Single GPU: just copy local to full
        CUDA_CHECK(cudaMemcpy(d_Ap, d_Ap_local, n_local * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // r_local = b_local - Ap_local (compute only local segment)
    // Step 1: r = Ap
    copy_kernel<<<blocks_local, threads, 0, stream>>>(d_Ap + row_offset, d_r + row_offset, n_local);
    // Step 2: r = b - r = b - Ap
    axpby_kernel<<<blocks_local, threads, 0, stream>>>(1.0, d_b + row_offset, -1.0, d_r + row_offset, n_local);

    // AllGather r (synchronize across GPUs)
    if (world_size > 1) {
        CHECK_NCCL(ncclGroupStart());
        for (int r = 0; r < world_size; r++) {
            int r_base_rows = n / world_size;
            int r_remainder = n % world_size;
            int r_n_local = r_base_rows + (r < r_remainder ? 1 : 0);
            int r_offset = r * r_base_rows + (r < r_remainder ? r : r_remainder);

            if (r == rank) {
                CHECK_NCCL(ncclBroadcast(d_r + r_offset, d_r + r_offset,
                                         r_n_local, ncclDouble, r, nccl_comm, stream));
            } else {
                CHECK_NCCL(ncclBroadcast(d_r + r_offset, d_r + r_offset,
                                         r_n_local, ncclDouble, r, nccl_comm, stream));
            }
        }
        CHECK_NCCL(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));  // Wait for AllGather to complete
    }

    // p_local = r_local
    copy_kernel<<<blocks_local, threads, 0, stream>>>(d_r + row_offset, d_p + row_offset, n_local);

    // AllGather p (synchronize across GPUs)
    if (world_size > 1) {
        CHECK_NCCL(ncclGroupStart());
        for (int r = 0; r < world_size; r++) {
            int r_base_rows = n / world_size;
            int r_remainder = n % world_size;
            int r_n_local = r_base_rows + (r < r_remainder ? 1 : 0);
            int r_offset = r * r_base_rows + (r < r_remainder ? r : r_remainder);

            if (r == rank) {
                CHECK_NCCL(ncclBroadcast(d_p + r_offset, d_p + r_offset,
                                         r_n_local, ncclDouble, r, nccl_comm, stream));
            } else {
                CHECK_NCCL(ncclBroadcast(d_p + r_offset, d_p + r_offset,
                                         r_n_local, ncclDouble, r, nccl_comm, stream));
            }
        }
        CHECK_NCCL(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));  // Wait for AllGather to complete
    }

    // rs_old = r^T * r (global dot product)
    double rs_local = compute_local_dot(d_r + row_offset, d_r + row_offset, n_local, d_work, stream);
    double rs_old;
    MPI_Allreduce(&rs_local, &rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_norm = sqrt(rs_old);  // Initial residual is b norm (since x0 = 0)
    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", b_norm);
    }

    // CG iteration loop
    int iter;
    for (iter = 0; iter < config.max_iters; iter++) {
        // Ap = A * p (distributed SpMV with multi-GPU kernel)
        stencil5_csr_direct_mgpu_kernel<<<blocks_spmv, threads_spmv, 0, stream>>>(
            d_local_row_ptr, d_local_col_idx, d_local_values,
            d_p, d_Ap_local,
            row_offset, n_local, mat->grid_size, 1.0
        );
        CUDA_CHECK(cudaGetLastError());

        // AllGather Ap segments (NCCL GPU-direct)
        if (world_size > 1) {
            CHECK_NCCL(ncclGroupStart());

            for (int r = 0; r < world_size; r++) {
                int r_base_rows = n / world_size;
                int r_remainder = n % world_size;
                int r_n_local = r_base_rows + (r < r_remainder ? 1 : 0);
                int r_offset = r * r_base_rows + (r < r_remainder ? r : r_remainder);

                if (r == rank) {
                    CHECK_NCCL(ncclBroadcast(d_Ap_local, d_Ap + r_offset,
                                             r_n_local, ncclDouble, r, nccl_comm, stream));
                } else {
                    CHECK_NCCL(ncclBroadcast(d_Ap + r_offset, d_Ap + r_offset,
                                             r_n_local, ncclDouble, r, nccl_comm, stream));
                }
            }

            CHECK_NCCL(ncclGroupEnd());
        } else {
            CUDA_CHECK(cudaMemcpy(d_Ap, d_Ap_local, n_local * sizeof(double), cudaMemcpyDeviceToDevice));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // alpha = rs_old / (p^T * Ap)
        double pAp_local = compute_local_dot(d_p + row_offset, d_Ap + row_offset, n_local, d_work, stream);
        double pAp;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (fabs(pAp) < 1e-20) {
            if (rank == 0) {
                printf("[Rank 0] Warning: pAp too small, breaking\n");
            }
            break;
        }

        double alpha = rs_old / pAp;

        // x_local = x_local + alpha * p_local (update only local segment)
        axpy_kernel<<<blocks_local, threads, 0, stream>>>(alpha, d_p + row_offset, d_x + row_offset, n_local);

        // AllGather x (each rank broadcasts its segment)
        if (world_size > 1) {
            CHECK_NCCL(ncclGroupStart());
            for (int r = 0; r < world_size; r++) {
                int r_base_rows = n / world_size;
                int r_remainder = n % world_size;
                int r_n_local = r_base_rows + (r < r_remainder ? 1 : 0);
                int r_offset = r * r_base_rows + (r < r_remainder ? r : r_remainder);

                if (r == rank) {
                    CHECK_NCCL(ncclBroadcast(d_x + r_offset, d_x + r_offset,
                                             r_n_local, ncclDouble, r, nccl_comm, stream));
                } else {
                    CHECK_NCCL(ncclBroadcast(d_x + r_offset, d_x + r_offset,
                                             r_n_local, ncclDouble, r, nccl_comm, stream));
                }
            }
            CHECK_NCCL(ncclGroupEnd());
            CUDA_CHECK(cudaStreamSynchronize(stream));  // Wait for AllGather to complete
        }

        // r_local = r_local - alpha * Ap_local (update only local segment)
        axpy_kernel<<<blocks_local, threads, 0, stream>>>(-alpha, d_Ap + row_offset, d_r + row_offset, n_local);

        // AllGather r (each rank broadcasts its segment)
        if (world_size > 1) {
            CHECK_NCCL(ncclGroupStart());
            for (int r = 0; r < world_size; r++) {
                int r_base_rows = n / world_size;
                int r_remainder = n % world_size;
                int r_n_local = r_base_rows + (r < r_remainder ? 1 : 0);
                int r_offset = r * r_base_rows + (r < r_remainder ? r : r_remainder);

                if (r == rank) {
                    CHECK_NCCL(ncclBroadcast(d_r + r_offset, d_r + r_offset,
                                             r_n_local, ncclDouble, r, nccl_comm, stream));
                } else {
                    CHECK_NCCL(ncclBroadcast(d_r + r_offset, d_r + r_offset,
                                             r_n_local, ncclDouble, r, nccl_comm, stream));
                }
            }
            CHECK_NCCL(ncclGroupEnd());
            CUDA_CHECK(cudaStreamSynchronize(stream));  // Wait for AllGather to complete
        }

        // rs_new = r^T * r
        double rs_local_new = compute_local_dot(d_r + row_offset, d_r + row_offset, n_local, d_work, stream);
        double rs_new;
        MPI_Allreduce(&rs_local_new, &rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double residual_norm = sqrt(rs_new);
        double rel_residual = residual_norm / b_norm;

        if (rank == 0 && config.verbose >= 2) {
            printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n",
                   iter + 1, residual_norm, rel_residual, alpha);
        }

        // Check convergence (relative criterion)
        if (rel_residual < config.tolerance) {
            iter++;
            if (rank == 0 && config.verbose >= 1) {
                printf("\nConverged in %d iterations (residual: %.6e)\n",
                       iter, residual_norm);
            }

            stats->converged = 1;
            stats->iterations = iter;
            stats->residual_norm = residual_norm;
            break;
        }

        // beta = rs_new / rs_old
        double beta = rs_new / rs_old;

        // p_local = r_local + beta * p_local (update only local segment)
        axpby_kernel<<<blocks_local, threads, 0, stream>>>(1.0, d_r + row_offset, beta, d_p + row_offset, n_local);

        // AllGather p (each rank broadcasts its segment)
        if (world_size > 1) {
            CHECK_NCCL(ncclGroupStart());
            for (int r = 0; r < world_size; r++) {
                int r_base_rows = n / world_size;
                int r_remainder = n % world_size;
                int r_n_local = r_base_rows + (r < r_remainder ? 1 : 0);
                int r_offset = r * r_base_rows + (r < r_remainder ? r : r_remainder);

                if (r == rank) {
                    CHECK_NCCL(ncclBroadcast(d_p + r_offset, d_p + r_offset,
                                             r_n_local, ncclDouble, r, nccl_comm, stream));
                } else {
                    CHECK_NCCL(ncclBroadcast(d_p + r_offset, d_p + r_offset,
                                             r_n_local, ncclDouble, r, nccl_comm, stream));
                }
            }
            CHECK_NCCL(ncclGroupEnd());
            CUDA_CHECK(cudaStreamSynchronize(stream));  // Wait for AllGather to complete
        }

        rs_old = rs_new;
    }

    // Check if max iterations reached
    if (iter == config.max_iters && rank == 0) {
        printf("\nMax iterations reached without convergence\n");
        stats->converged = 0;
        stats->iterations = iter;
        stats->residual_norm = sqrt(rs_old);
    }

    // ========== Timing ==========
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    if (rank == 0) {
        stats->time_total_ms = time_ms;
        if (config.verbose >= 1) {
            printf("Total time: %.2f ms\n", time_ms);
            printf("========================================\n");
        }
    }

    // ========== Copy result back ==========
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // ========== Cleanup ==========
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_Ap_local);
    cudaFree(d_b);
    cudaFree(d_work);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    if (nccl_comm != NULL) {
        ncclCommDestroy(nccl_comm);
    }

    return 0;
}
