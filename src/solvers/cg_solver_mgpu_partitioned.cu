/**
 * @file cg_solver_mgpu_partitioned.cu
 * @brief Multi-GPU CG solver with CSR partitioning and halo exchange
 *
 * Architecture:
 * - Each GPU: Local CSR partition (rows [row_offset : row_offset + n_local))
 * - Halo zones: Ghost cells for stencil boundary dependencies (1 row per neighbor)
 * - Communication: P2P exchange of boundary rows only (not full vectors)
 *
 * For 5-point stencil on 10000×10000 grid with 2 GPUs:
 * - GPU0: rows [0:5000), needs row 5000 from GPU1 (grid_size doubles = 80 KB)
 * - GPU1: rows [5000:10000), needs row 4999 from GPU0 (grid_size doubles = 80 KB)
 * - Total communication: 160 KB per iteration vs 800 MB AllGather
 *
 * Halo layout in memory:
 * - d_p_local[0:n_local]        : Local partition
 * - d_p_halo_prev[0:grid_size]  : Previous rank boundary (if exists)
 * - d_p_halo_next[0:grid_size]  : Next rank boundary (if exists)
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-11
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "spmv.h"
#include "io.h"
#include "solvers/cg_solver_mgpu_partitioned.h"

// NCCL error checking macro
#ifndef CHECK_NCCL
#define CHECK_NCCL(cmd) do {                          \
  ncclResult_t res = cmd;                             \
  if (res != ncclSuccess) {                           \
    fprintf(stderr, "NCCL error %s:%d '%s'\n",        \
        __FILE__,__LINE__,ncclGetErrorString(res));   \
    exit(EXIT_FAILURE);                               \
  }                                                   \
} while(0)
#endif

/**
 * @brief Simple CSR SpMV kernel (non-optimized, standard)
 * @details One thread per row, standard CSR traversal
 */
__global__ void csr_spmv_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    double sum = 0.0;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * x[col_idx[j]];
    }

    y[row] = sum;
}

/**
 * @brief Optimized stencil SpMV kernel for partitioned CSR with halo zones
 * @details Uses direct column indices (no indirection) for interior stencil points
 *          Accesses local + halo data instead of global vector
 *
 * Memory layout:
 * - x_local[0:n_local] : Local partition data
 * - x_halo_prev[0:grid_size] : Halo from previous rank (if exists)
 * - x_halo_next[0:grid_size] : Halo from next rank (if exists)
 *
 * Global index mapping:
 * - [row_offset:row_offset+n_local) → x_local[]
 * - row_offset-grid_size : row_offset-1 → x_halo_prev[] (previous rank boundary row)
 * - row_offset+n_local : row_offset+n_local+grid_size-1 → x_halo_next[] (next rank boundary row)
 */
__global__ void stencil5_csr_partitioned_halo_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x_local,      // Local vector (n_local)
    const double* __restrict__ x_halo_prev,  // Halo from prev rank (grid_size or NULL)
    const double* __restrict__ x_halo_next,  // Halo from next rank (grid_size or NULL)
    double* __restrict__ y,                  // Local output (n_local)
    int n_local,                             // Number of local rows
    int row_offset,                          // Global row offset
    int N,                                   // Global size
    int grid_size
) {
    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_local) return;

    int row = row_offset + local_row;  // Global row for geometry
    int i = row / grid_size;
    int j = row % grid_size;

    // Get CSR row range from row_ptr
    int row_start = row_ptr[local_row];
    int row_end = row_ptr[local_row + 1];

    double sum = 0.0;

    // Interior points: direct column calculation (no col_idx lookup)
    if (i > 0 && i < grid_size - 1 && j > 0 && j < grid_size - 1 && (row_end - row_start) == 5) {
        // Column indices known from stencil structure (global indices)
        int idx_north = row - grid_size;
        int idx_west = row - 1;
        int idx_center = row;
        int idx_east = row + 1;
        int idx_south = row + grid_size;

        // Map global indices to local/halo space
        // North: row - grid_size
        double val_north;
        if (idx_north >= row_offset && idx_north < row_offset + n_local) {
            val_north = x_local[idx_north - row_offset];
        } else if (idx_north >= row_offset - grid_size && idx_north < row_offset) {
            // Previous rank halo (boundary row)
            val_north = x_halo_prev[idx_north - (row_offset - grid_size)];
        } else {
            val_north = 0.0;  // Should never happen for valid stencil
        }

        // West, Center, East: Always in local partition for interior points
        double val_west = x_local[idx_west - row_offset];
        double val_center = x_local[idx_center - row_offset];
        double val_east = x_local[idx_east - row_offset];

        // South: row + grid_size
        double val_south;
        if (idx_south >= row_offset && idx_south < row_offset + n_local) {
            val_south = x_local[idx_south - row_offset];
        } else if (idx_south >= row_offset + n_local && idx_south < row_offset + n_local + grid_size) {
            // Next rank halo (boundary row)
            val_south = x_halo_next[idx_south - (row_offset + n_local)];
        } else {
            val_south = 0.0;  // Should never happen for valid stencil
        }

        // CSR sorted order: [North, West, Center, East, South]
        sum = values[row_start + 0] * val_north
            + values[row_start + 1] * val_west
            + values[row_start + 2] * val_center
            + values[row_start + 3] * val_east
            + values[row_start + 4] * val_south;
    }
    // Boundary/corner: standard CSR traversal with global index mapping
    else {
        for (int k = row_start; k < row_end; k++) {
            int global_col = col_idx[k];
            double val;

            // Map global column to local/halo
            if (global_col >= row_offset && global_col < row_offset + n_local) {
                val = x_local[global_col - row_offset];
            } else if (x_halo_prev != NULL && global_col >= row_offset - grid_size && global_col < row_offset) {
                val = x_halo_prev[global_col - (row_offset - grid_size)];
            } else if (x_halo_next != NULL && global_col >= row_offset + n_local && global_col < row_offset + n_local + grid_size) {
                val = x_halo_next[global_col - (row_offset + n_local)];
            } else {
                val = 0.0;  // Should never happen for valid stencil
            }

            sum += values[k] * val;
        }
    }

    y[local_row] = sum;  // Write to local output
}

/**
 * @brief Original SpMV kernel using global vector (for compatibility)
 */
__global__ void stencil5_csr_direct_partitioned_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,        // Global vector (full N)
    double* __restrict__ y,              // Local output (n_local)
    int n_local,                         // Number of local rows
    int row_offset,                      // Global row offset
    int N,                               // Global size
    int grid_size
) {
    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_local) return;

    int row = row_offset + local_row;  // Global row for geometry
    int i = row / grid_size;
    int j = row % grid_size;

    // Get CSR row range from row_ptr
    int row_start = row_ptr[local_row];
    int row_end = row_ptr[local_row + 1];

    double sum = 0.0;

    // Interior points: direct column calculation (no col_idx lookup)
    if (i > 0 && i < grid_size - 1 && j > 0 && j < grid_size - 1 && (row_end - row_start) == 5) {
        // Column indices known from stencil structure (global indices)
        int idx_north = row - grid_size;
        int idx_west = row - 1;
        int idx_center = row;
        int idx_east = row + 1;
        int idx_south = row + grid_size;

        // Optimized memory access: W-C-E (stride 1), then N-S (stride grid_size)
        // CSR sorted order: [North, West, Center, East, South]
        sum = values[row_start + 1] * x[idx_west]      // West
            + values[row_start + 2] * x[idx_center]    // Center
            + values[row_start + 3] * x[idx_east]      // East
            + values[row_start + 0] * x[idx_north]     // North
            + values[row_start + 4] * x[idx_south];    // South
    }
    // Boundary/corner: standard CSR traversal
    else {
        #pragma unroll 8
        for (int k = row_start; k < row_end; k++) {
            sum += values[k] * x[col_idx[k]];
        }
    }

    y[local_row] = sum;  // Write to local output
}

/**
 * @brief AXPY kernel: y = alpha * x + y
 */
__global__ void axpy_kernel(double alpha, const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}

/**
 * @brief AXPBY kernel: y = alpha * x + beta * y
 */
__global__ void axpby_kernel(double alpha, const double* x, double beta, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

/**
 * @brief Compute local dot product using cuBLAS
 */
static double compute_local_dot(cublasHandle_t cublas_handle, const double* d_x, const double* d_y, int n) {
    double result;
    cublasStatus_t status = cublasDdot(cublas_handle, n, d_x, 1, d_y, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS ddot failed\n");
        exit(EXIT_FAILURE);
    }
    return result;
}

/**
 * @brief Exchange halo zones with neighbors using P2P NCCL communication
 * @details Each rank sends its boundary row to neighbors and receives theirs
 *
 * For 5-point stencil with row-band partitioning:
 * - GPU0: sends last row to GPU1, receives GPU1's first row
 * - GPU1: sends first row to GPU0, receives GPU0's last row
 *
 * Total communication: 2 × grid_size × 8 bytes = 160 KB (vs 800 MB AllGather)
 */
static void exchange_halo_p2p(
    const double* d_local_send_prev,    // Data to send to prev rank (or NULL)
    const double* d_local_send_next,    // Data to send to next rank (or NULL)
    double* d_halo_recv_prev,           // Buffer to receive from prev rank (or NULL)
    double* d_halo_recv_next,           // Buffer to receive from next rank (or NULL)
    int halo_size,                      // Number of elements per halo (grid_size)
    int rank,
    int world_size,
    ncclComm_t nccl_comm,
    cudaStream_t stream
) {
    // P2P communication pattern for row-band decomposition:
    // - Even ranks send down, odd ranks send up (deadlock avoidance)
    // - Then reverse direction

    // Phase 1: Send to next rank, receive from previous rank
    if (rank < world_size - 1) {
        // Send last row to next rank
        CHECK_NCCL(ncclSend(d_local_send_next, halo_size, ncclDouble, rank + 1, nccl_comm, stream));
    }
    if (rank > 0) {
        // Receive from previous rank
        CHECK_NCCL(ncclRecv(d_halo_recv_prev, halo_size, ncclDouble, rank - 1, nccl_comm, stream));
    }

    // Phase 2: Send to previous rank, receive from next rank
    if (rank > 0) {
        // Send first row to previous rank
        CHECK_NCCL(ncclSend(d_local_send_prev, halo_size, ncclDouble, rank - 1, nccl_comm, stream));
    }
    if (rank < world_size - 1) {
        // Receive from next rank
        CHECK_NCCL(ncclRecv(d_halo_recv_next, halo_size, ncclDouble, rank + 1, nccl_comm, stream));
    }
}

/**
 * @brief Multi-GPU CG solver with CSR partitioning
 */
int cg_solve_mgpu_partitioned(SpmvOperator* spmv_op,
                               MatrixData* mat,
                               const double* b,
                               double* x,
                               CGConfigMultiGPU config,
                               CGStatsMultiGPU* stats) {

    // MPI initialization
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = mat->rows;
    int grid_size = mat->grid_size;

    if (rank == 0 && config.verbose >= 1) {
        printf("\n========================================\n");
        printf("Multi-GPU CG Solver (PARTITIONED CSR)\n");
        printf("========================================\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Problem size: %d unknowns\n", n);
        printf("Max iterations: %d\n", config.max_iters);
        printf("Tolerance: %.1e\n", config.tolerance);
        printf("========================================\n\n");
    }

    // Set GPU device
    CUDA_CHECK(cudaSetDevice(rank));

    // Partition: 1D row-band decomposition
    int n_local = n / world_size;
    int row_offset = rank * n_local;

    // Adjust last rank if n not divisible
    if (rank == world_size - 1) {
        n_local = n - row_offset;
    }

    if (config.verbose >= 1) {
        char gpu_name[256];
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, rank));
        snprintf(gpu_name, sizeof(gpu_name), "%s", prop.name);
        printf("[Rank %d] GPU %d: %s (CC %d.%d)\n", rank, rank, gpu_name, prop.major, prop.minor);
        printf("[Rank %d] Rows: [%d:%d) (%d rows)\n", rank, row_offset, row_offset + n_local, n_local);
    }

    // Initialize NCCL
    ncclComm_t nccl_comm;
    ncclUniqueId nccl_id;

    if (rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));

    if (rank == 0 && config.verbose >= 1) {
        printf("NCCL initialized (%d GPUs)\n\n", world_size);
    }

    // Create CUDA stream and cuBLAS handle
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
    cublasSetStream(cublas_handle, stream);

    // Build local CSR partition
    // TODO: For now, we build full CSR and extract partition (temporary)
    // In production, should partition during CSR construction

    if (rank == 0 && config.verbose >= 1) {
        printf("Building local CSR partitions...\n");
    }

    // Build full CSR (temporary - will optimize later)
    build_csr_struct(mat);

    // Extract local CSR partition
    int local_nnz = csr_mat.row_ptr[row_offset + n_local] - csr_mat.row_ptr[row_offset];

    // Allocate local CSR on device
    int *d_row_ptr, *d_col_idx;
    double *d_values;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n_local + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, local_nnz * sizeof(double)));

    // Copy local CSR partition to device
    // Adjust row_ptr offsets to start from 0
    int* local_row_ptr = (int*)malloc((n_local + 1) * sizeof(int));
    int offset = csr_mat.row_ptr[row_offset];
    for (int i = 0; i <= n_local; i++) {
        local_row_ptr[i] = csr_mat.row_ptr[row_offset + i] - offset;
    }

    CUDA_CHECK(cudaMemcpy(d_row_ptr, local_row_ptr, (n_local + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, &csr_mat.col_indices[offset], local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, &csr_mat.values[offset], local_nnz * sizeof(double), cudaMemcpyHostToDevice));

    free(local_row_ptr);

    if (config.verbose >= 1) {
        printf("[Rank %d] Local CSR: %d rows, %d nnz (%.2f MB)\n",
               rank, n_local, local_nnz,
               (n_local * sizeof(int) + local_nnz * (sizeof(int) + sizeof(double))) / 1e6);
    }

    // Allocate vectors - LOCAL ONLY with halo buffers
    // With halo exchange: no need for full-size vectors, only local + halo
    double *d_x_local, *d_r_local, *d_p_local, *d_Ap, *d_b;
    double *d_p_halo_prev, *d_p_halo_next;  // Halo buffers for p vector
    double *d_r_halo_prev, *d_r_halo_next;  // Halo buffers for r vector

    // Local data (owned partition)
    CUDA_CHECK(cudaMalloc(&d_x_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));

    // Halo buffers (boundary rows from neighbors)
    // Each halo is one grid row (grid_size elements)
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_prev, grid_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r_halo_prev, grid_size * sizeof(double)));
    } else {
        d_p_halo_prev = NULL;
        d_r_halo_prev = NULL;
    }

    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_next, grid_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r_halo_next, grid_size * sizeof(double)));
    } else {
        d_p_halo_next = NULL;
        d_r_halo_next = NULL;
    }

    if (config.verbose >= 1) {
        size_t local_mem = n_local * 5 * sizeof(double);  // x, r, p, Ap, b
        size_t halo_mem = 0;
        if (rank > 0) halo_mem += grid_size * 2 * sizeof(double);  // p_prev, r_prev
        if (rank < world_size - 1) halo_mem += grid_size * 2 * sizeof(double);  // p_next, r_next
        printf("[Rank %d] Vector memory: %.2f MB (local) + %.2f KB (halo)\n",
               rank, local_mem / 1e6, halo_mem / 1e3);
    }

    // Initialize vectors (local partition only)
    CUDA_CHECK(cudaMemcpy(d_b, &b[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_local, &x[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_r_local, 0, n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p_local, 0, n_local * sizeof(double)));

    // Start timing
    cudaEvent_t start, stop, timer_start, timer_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&timer_start));
    CUDA_CHECK(cudaEventCreate(&timer_stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Initialize CG statistics
    stats->time_spmv_ms = 0.0;
    stats->time_blas1_ms = 0.0;
    stats->time_reductions_ms = 0.0;
    stats->time_allreduce_ms = 0.0;
    stats->time_allgather_ms = 0.0;

    if (rank == 0 && config.verbose >= 1) {
        printf("\nStarting CG iterations...\n");
    }

    // Compute initial residual: r = b - A*x
    int threads = 256;
    int blocks_local = (n_local + threads - 1) / threads;

    // Initial x halo exchange for SpMV (x0 initial guess)
    // For stencil: send first/last row of x_local to neighbors
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_start, stream));
    }

    // Exchange x halo zones (for initial SpMV only - x doesn't change after this)
    // Send boundary rows: first row to prev, last row to next
    double *d_x_halo_prev, *d_x_halo_next;
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_prev, grid_size * sizeof(double)));
    } else {
        d_x_halo_prev = NULL;
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_next, grid_size * sizeof(double)));
    } else {
        d_x_halo_next = NULL;
    }

    exchange_halo_p2p(
        d_x_local,                                   // First row to send to prev
        d_x_local + (n_local - grid_size),           // Last row to send to next
        d_x_halo_prev,                               // Receive from prev
        d_x_halo_next,                               // Receive from next
        grid_size, rank, world_size, nccl_comm, stream
    );

    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(timer_stop));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
        stats->time_allgather_ms += elapsed_ms;  // Count as communication time
    }

    // Initial SpMV: Ap = A*x (using x0 with halo)
    stencil5_csr_partitioned_halo_kernel<<<blocks_local, threads, 0, stream>>>(
        d_row_ptr, d_col_idx, d_values,
        d_x_local, d_x_halo_prev, d_x_halo_next,
        d_Ap, n_local, row_offset, n, grid_size);

    // r_local = b - Ap
    axpy_kernel<<<blocks_local, threads, 0, stream>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpy(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice));

    // Exchange r halo for initial dot product
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_start, stream));
    }
    exchange_halo_p2p(
        d_r_local,                                   // First row to prev
        d_r_local + (n_local - grid_size),           // Last row to next
        d_r_halo_prev,                               // Receive from prev
        d_r_halo_next,                               // Receive from next
        grid_size, rank, world_size, nccl_comm, stream
    );
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(timer_stop));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
        stats->time_allgather_ms += elapsed_ms;
    }

    // p_local = r_local (local copy)
    CUDA_CHECK(cudaMemcpy(d_p_local, d_r_local, n_local * sizeof(double), cudaMemcpyDeviceToDevice));

    // Copy r halo to p halo
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(d_p_halo_prev, d_r_halo_prev, grid_size * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMemcpy(d_p_halo_next, d_r_halo_next, grid_size * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // rs_old = r^T * r (local dot product + AllReduce)
    double rs_local_old = compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
    double rs_old;
    MPI_Allreduce(&rs_local_old, &rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_norm = sqrt(rs_old);

    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", sqrt(rs_old));
    }

    // CG iteration loop
    int iter;
    for (iter = 0; iter < config.max_iters; iter++) {
        // Ap = A * p (local SpMV) - halo-aware kernel
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        stencil5_csr_partitioned_halo_kernel<<<blocks_local, threads, 0, stream>>>(
            d_row_ptr, d_col_idx, d_values,
            d_p_local, d_p_halo_prev, d_p_halo_next,
            d_Ap, n_local, row_offset, n, grid_size);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_spmv_ms += elapsed_ms;
        }

        // alpha = rs_old / (p^T * Ap) - local dot product
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        double pAp_local = compute_local_dot(cublas_handle, d_p_local, d_Ap, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_reductions_ms += elapsed_ms;
        }

        // AllReduce for pAp
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        double pAp;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allreduce_ms += elapsed_ms;
        }

        double alpha = rs_old / pAp;

        // x_local = x_local + alpha * p_local
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        axpy_kernel<<<blocks_local, threads, 0, stream>>>(alpha, d_p_local, d_x_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
        }

        // r_local = r_local - alpha * Ap_local
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        axpy_kernel<<<blocks_local, threads, 0, stream>>>(-alpha, d_Ap, d_r_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
        }

        // rs_new = r^T * r - local dot product
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        double rs_local_new = compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_reductions_ms += elapsed_ms;
        }

        // AllReduce for rs_new
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        double rs_new;
        MPI_Allreduce(&rs_local_new, &rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allreduce_ms += elapsed_ms;
        }

        double residual_norm = sqrt(rs_new);
        double rel_residual = residual_norm / b_norm;

        if (rank == 0 && config.verbose >= 2) {
            printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n",
                   iter + 1, residual_norm, rel_residual, alpha);
        }

        // Check convergence
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

        // p_local = r_local + beta * p_local
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        axpby_kernel<<<blocks_local, threads, 0, stream>>>(1.0, d_r_local, beta, d_p_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
        }

        // P2P halo exchange for p vector (160 KB vs 800 MB AllGather)
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }
        exchange_halo_p2p(
            d_p_local,                                   // First row to prev
            d_p_local + (n_local - grid_size),           // Last row to next
            d_p_halo_prev,                               // Receive from prev
            d_p_halo_next,                               // Receive from next
            grid_size, rank, world_size, nccl_comm, stream
        );
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allgather_ms += elapsed_ms;  // Keep same name for comparison
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

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    if (rank == 0) {
        stats->time_total_ms = time_ms;
        if (config.verbose >= 1) {
            printf("Total time: %.2f ms\n", time_ms);
            if (config.enable_detailed_timers) {
                printf("\nDetailed Timing Breakdown:\n");
                printf("  SpMV:       %.2f ms (%.1f%%)\n", stats->time_spmv_ms, 100.0 * stats->time_spmv_ms / time_ms);
                printf("  BLAS1:      %.2f ms (%.1f%%)\n", stats->time_blas1_ms, 100.0 * stats->time_blas1_ms / time_ms);
                printf("  Reductions: %.2f ms (%.1f%%)\n", stats->time_reductions_ms, 100.0 * stats->time_reductions_ms / time_ms);
                printf("  AllReduce:  %.2f ms (%.1f%%)\n", stats->time_allreduce_ms, 100.0 * stats->time_allreduce_ms / time_ms);
                printf("  Halo P2P:   %.2f ms (%.1f%%) [was AllGather]\n", stats->time_allgather_ms, 100.0 * stats->time_allgather_ms / time_ms);
            }
            printf("========================================\n");
        }
    }

    // Copy result back (local partition only)
    CUDA_CHECK(cudaMemcpy(&x[row_offset], d_x_local, n_local * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather full solution to rank 0
    MPI_Gather(&x[row_offset], n_local, MPI_DOUBLE, x, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Cleanup
    cudaFree(d_x_local);
    cudaFree(d_r_local);
    cudaFree(d_p_local);
    cudaFree(d_Ap);
    cudaFree(d_b);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);

    // Cleanup halo buffers
    if (d_x_halo_prev) cudaFree(d_x_halo_prev);
    if (d_x_halo_next) cudaFree(d_x_halo_next);
    if (d_p_halo_prev) cudaFree(d_p_halo_prev);
    if (d_p_halo_next) cudaFree(d_p_halo_next);
    if (d_r_halo_prev) cudaFree(d_r_halo_prev);
    if (d_r_halo_next) cudaFree(d_r_halo_next);

    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (config.enable_detailed_timers) {
        cudaEventDestroy(timer_start);
        cudaEventDestroy(timer_stop);
    }

    ncclCommDestroy(nccl_comm);

    return 0;
}
