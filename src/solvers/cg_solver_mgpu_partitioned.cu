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
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#include "spmv.h"
#include "io.h"
#include "solvers/cg_solver_mgpu_partitioned.h"

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
 * @brief Shared kernel - implementation in src/spmv/spmv_stencil_partitioned_halo_kernel.cu
 */
extern __global__ void stencil5_csr_partitioned_halo_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev,
    const double* __restrict__ x_halo_next,
    double* __restrict__ y,
    int n_local,
    int row_offset,
    int N,
    int grid_size
);

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
 * @brief CUDA IPC handles for remote GPU memory access
 */
struct HaloIPCHandles {
    cudaIpcMemHandle_t send_prev_handle;  // IPC handle for data to send to prev rank
    cudaIpcMemHandle_t send_next_handle;  // IPC handle for data to send to next rank
    double* remote_prev_ptr;              // Opened pointer to prev rank's send buffer
    double* remote_next_ptr;              // Opened pointer to next rank's send buffer
    bool has_prev;                        // Whether this rank has a previous neighbor
    bool has_next;                        // Whether this rank has a next neighbor
};

/**
 * @brief Setup CUDA peer access and exchange IPC memory handles via MPI
 *
 * Each rank shares IPC handles for its local boundary rows with neighbors.
 * Neighbors open these handles to get direct GPU pointers for cudaMemcpyPeer.
 *
 * @param d_local Pointer to local partition on device
 * @param n_local Number of local rows
 * @param grid_size Halo size (one grid row)
 * @param rank MPI rank
 * @param world_size Number of MPI ranks
 * @param handles [out] Initialized IPC handles structure
 */
static void setup_halo_ipc(
    const double* d_local,
    int n_local,
    int grid_size,
    int rank,
    int world_size,
    HaloIPCHandles* handles
) {
    handles->has_prev = (rank > 0);
    handles->has_next = (rank < world_size - 1);
    handles->remote_prev_ptr = NULL;
    handles->remote_next_ptr = NULL;

    // Enable peer access to neighbor GPUs
    if (handles->has_prev) {
        int can_access_prev;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_prev, rank, rank - 1));
        if (can_access_prev) {
            cudaError_t peer_err = cudaDeviceEnablePeerAccess(rank - 1, 0);
            if (peer_err != cudaSuccess && peer_err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(peer_err);
            }
        } else {
            fprintf(stderr, "[Rank %d] Cannot enable peer access to GPU %d\n", rank, rank - 1);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (handles->has_next) {
        int can_access_next;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_next, rank, rank + 1));
        if (can_access_next) {
            cudaError_t peer_err = cudaDeviceEnablePeerAccess(rank + 1, 0);
            if (peer_err != cudaSuccess && peer_err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(peer_err);
            }
        } else {
            fprintf(stderr, "[Rank %d] Cannot enable peer access to GPU %d\n", rank, rank + 1);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Create IPC handles for local data to share with neighbors
    if (handles->has_prev) {
        // Share first row with prev rank (they need it as their halo_next)
        CUDA_CHECK(cudaIpcGetMemHandle(&handles->send_prev_handle, (void*)d_local));
    }

    if (handles->has_next) {
        // Share last row with next rank (they need it as their halo_prev)
        const double* last_row_ptr = d_local + (n_local - grid_size);
        CUDA_CHECK(cudaIpcGetMemHandle(&handles->send_next_handle, (void*)last_row_ptr));
    }

    // Exchange IPC handles with neighbors via MPI
    MPI_Request requests[4];
    int req_count = 0;

    cudaIpcMemHandle_t recv_prev_handle, recv_next_handle;

    if (handles->has_prev) {
        // Send my first row handle to prev, receive their last row handle
        MPI_Isend(&handles->send_prev_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                  rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(&recv_prev_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                  rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    if (handles->has_next) {
        // Send my last row handle to next, receive their first row handle
        MPI_Isend(&handles->send_next_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                  rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(&recv_next_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                  rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    // Open received IPC handles to get remote GPU pointers
    if (handles->has_prev) {
        CUDA_CHECK(cudaIpcOpenMemHandle((void**)&handles->remote_prev_ptr,
                                        recv_prev_handle,
                                        cudaIpcMemLazyEnablePeerAccess));
    }

    if (handles->has_next) {
        CUDA_CHECK(cudaIpcOpenMemHandle((void**)&handles->remote_next_ptr,
                                        recv_next_handle,
                                        cudaIpcMemLazyEnablePeerAccess));
    }
}

/**
 * @brief Cleanup CUDA IPC handles
 */
static void cleanup_halo_ipc(HaloIPCHandles* handles) {
    if (handles->remote_prev_ptr != NULL) {
        CUDA_CHECK(cudaIpcCloseMemHandle(handles->remote_prev_ptr));
    }
    if (handles->remote_next_ptr != NULL) {
        CUDA_CHECK(cudaIpcCloseMemHandle(handles->remote_next_ptr));
    }
}

/**
 * @brief Synchronize with neighbor ranks before P2P read
 *
 * Ensures that neighbor ranks have finished updating their local data
 * before this rank reads their memory via CUDA IPC P2P.
 *
 * Without this sync, there's a race condition:
 * - Rank 0 finishes kernel quickly, starts P2P read from Rank 1
 * - Rank 1 hasn't finished kernel yet → Rank 0 reads partial/stale data
 * - Result: Numerical variability, non-deterministic convergence
 *
 * This is why MPI staging is stable (MPI_Waitall provides implicit sync)
 * but P2P direct needs explicit inter-rank synchronization.
 */
static void sync_with_neighbors(int rank, int world_size) {
    char dummy = 0;
    MPI_Request send_reqs[2];
    int send_count = 0;

    // Send "ready" signal to neighbors
    if (rank > 0) {
        MPI_Isend(&dummy, 1, MPI_CHAR, rank - 1, 99, MPI_COMM_WORLD, &send_reqs[send_count++]);
    }
    if (rank < world_size - 1) {
        MPI_Isend(&dummy, 1, MPI_CHAR, rank + 1, 99, MPI_COMM_WORLD, &send_reqs[send_count++]);
    }

    // Wait for "ready" signal from neighbors
    if (rank > 0) {
        MPI_Recv(&dummy, 1, MPI_CHAR, rank - 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < world_size - 1) {
        MPI_Recv(&dummy, 1, MPI_CHAR, rank + 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Wait for my sends to complete
    if (send_count > 0) {
        MPI_Waitall(send_count, send_reqs, MPI_STATUSES_IGNORE);
    }
}

/**
 * @brief Direct GPU-to-GPU halo exchange using cudaMemcpyPeerAsync
 *
 * Uses cudaMemcpyPeerAsync for proper GPU cache coherency (automatic L2 flush/invalidate).
 * Zero CPU staging overhead - transfers directly via NVLink/PCIe.
 * Requires CUDA IPC setup (see setup_halo_ipc).
 *
 * IMPORTANT: Must call sync_with_neighbors() BEFORE this function to ensure
 * neighbor ranks have finished updating their local data.
 *
 * For 5-point stencil with row-band partitioning:
 * - GPU0: copy last row to GPU1's halo_prev, receive GPU1's first row to halo_next
 * - GPU1: copy first row to GPU0's halo_next, receive GPU0's last row to halo_prev
 *
 * Total communication: 2 × grid_size × 8 bytes = 160 KB
 * Expected latency: ~1 ms (vs 2 ms MPI staging, vs 3.5-5.6 ms NCCL)
 */
static void exchange_halo_p2p_direct(
    double* d_halo_recv_prev,           // Local buffer to receive from prev rank
    double* d_halo_recv_next,           // Local buffer to receive from next rank
    const HaloIPCHandles* handles,      // IPC handles with remote pointers
    int halo_size,                      // Number of elements per halo (grid_size)
    cudaStream_t stream,
    int rank,                           // Local GPU device ID
    int world_size                      // Total number of GPUs
) {
    // Copy from remote GPU memory to local halo buffers using cudaMemcpyPeerAsync
    // This ensures proper cache coherency (automatic L2 flush/invalidate)
    // Transfers execute in parallel via NVLink/PCIe

    if (handles->has_prev && handles->remote_prev_ptr != NULL) {
        // Copy prev rank's last row → my halo_prev
        int src_device = rank - 1;  // Previous GPU
        int dst_device = rank;      // Local GPU
        CUDA_CHECK(cudaMemcpyPeerAsync(
            d_halo_recv_prev,               // Destination: local halo buffer
            dst_device,                     // Destination GPU ID
            handles->remote_prev_ptr,       // Source: remote GPU's last row
            src_device,                     // Source GPU ID
            halo_size * sizeof(double),
            stream
        ));
    }

    if (handles->has_next && handles->remote_next_ptr != NULL) {
        // Copy next rank's first row → my halo_next
        int src_device = rank + 1;  // Next GPU
        int dst_device = rank;      // Local GPU
        CUDA_CHECK(cudaMemcpyPeerAsync(
            d_halo_recv_next,               // Destination: local halo buffer
            dst_device,                     // Destination GPU ID
            handles->remote_next_ptr,       // Source: remote GPU's first row
            src_device,                     // Source GPU ID
            halo_size * sizeof(double),
            stream
        ));
    }

    // Ensure transfers complete before kernel uses halo data
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Exchange halo zones with neighbors using MPI with explicit staging
 * @details Each rank sends its boundary row to neighbors and receives theirs
 *
 * Implementation: Explicit staging (D2H, MPI, H2D) to avoid CUDA-aware MPI requirement:
 * 1. cudaMemcpyAsync D2H (device send → host send buffers)
 * 2. MPI_Isend/Irecv (non-blocking host communication)
 * 3. MPI_Waitall (ensure MPI operations complete)
 * 4. cudaMemcpyAsync H2D (host recv → device halo buffers)
 * 5. cudaStreamSynchronize (ensure data ready before kernel launch)
 *
 * For 5-point stencil with row-band partitioning:
 * - GPU0: sends last row to GPU1, receives GPU1's first row
 * - GPU1: sends first row to GPU0, receives GPU0's last row
 *
 * Total communication: 2 × grid_size × 8 bytes = 160 KB (vs 800 MB AllGather)
 */
static void exchange_halo_mpi(
    const double* d_local_send_prev,    // Data to send to prev rank (or NULL)
    const double* d_local_send_next,    // Data to send to next rank (or NULL)
    double* d_halo_recv_prev,           // Buffer to receive from prev rank (or NULL)
    double* d_halo_recv_next,           // Buffer to receive from next rank (or NULL)
    double* h_send_prev,                // Host pinned buffer for send to prev
    double* h_send_next,                // Host pinned buffer for send to next
    double* h_recv_prev,                // Host pinned buffer for recv from prev
    double* h_recv_next,                // Host pinned buffer for recv from next
    int halo_size,                      // Number of elements per halo (grid_size)
    int rank,
    int world_size,
    cudaStream_t stream
) {
    MPI_Request requests[4];
    int req_count = 0;

    // Step 1: Copy device send buffers to host asynchronously
    if (rank > 0 && d_local_send_prev != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_prev, d_local_send_prev,
                                    halo_size * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    }
    if (rank < world_size - 1 && d_local_send_next != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_next, d_local_send_next,
                                    halo_size * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    }

    // Wait for D2H copies to complete before MPI can access host buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Step 2: Launch all MPI_Isend/Irecv operations
    if (rank > 0) {
        MPI_Isend(h_send_prev, halo_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(h_recv_prev, halo_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    if (rank < world_size - 1) {
        MPI_Isend(h_send_next, halo_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(h_recv_next, halo_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Step 3: Wait for all MPI operations to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    // Step 4: Copy host recv buffers to device halo zones asynchronously
    if (rank > 0 && d_halo_recv_prev != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_halo_recv_prev, h_recv_prev,
                                    halo_size * sizeof(double),
                                    cudaMemcpyHostToDevice, stream));
    }
    if (rank < world_size - 1 && d_halo_recv_next != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_halo_recv_next, h_recv_next,
                                    halo_size * sizeof(double),
                                    cudaMemcpyHostToDevice, stream));
    }

    // Step 5: Ensure H2D copies complete before kernel uses halo data
    CUDA_CHECK(cudaStreamSynchronize(stream));
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

    // Note: Using MPI with explicit staging (no NCCL required)

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

    // Setup CUDA IPC for direct GPU-to-GPU halo exchange
    // Create handles for x, r, and p vectors
    HaloIPCHandles ipc_handles_x, ipc_handles_r, ipc_handles_p;

    // Initialize vectors (local partition only)
    CUDA_CHECK(cudaMemcpy(d_b, &b[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_local, &x[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_r_local, 0, n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p_local, 0, n_local * sizeof(double)));

    // Setup IPC handles AFTER vector initialization (need valid device pointers)
    setup_halo_ipc(d_x_local, n_local, grid_size, rank, world_size, &ipc_handles_x);
    setup_halo_ipc(d_r_local, n_local, grid_size, rank, world_size, &ipc_handles_r);
    setup_halo_ipc(d_p_local, n_local, grid_size, rank, world_size, &ipc_handles_p);

    // Synchronize all ranks before timing to avoid measuring rank arrival skew
    MPI_Barrier(MPI_COMM_WORLD);

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

    // Sync with neighbors before P2P read
    sync_with_neighbors(rank, world_size);

    exchange_halo_p2p_direct(
        d_x_halo_prev,                               // Receive from prev
        d_x_halo_next,                               // Receive from next
        &ipc_handles_x,                              // IPC handles for x vector
        grid_size, stream,
        rank, world_size                             // GPU device IDs for cudaMemcpyPeerAsync
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
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_start, stream));
    }
    axpy_kernel<<<blocks_local, threads, 0, stream>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpy(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice));
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(timer_stop));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
        stats->time_initial_r_ms = elapsed_ms;
    }

    // Exchange r halo for initial dot product
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_start, stream));
    }

    // Ensure r_local kernel is done before neighbors read it
    CUDA_CHECK(cudaStreamSynchronize(stream));
    sync_with_neighbors(rank, world_size);

    exchange_halo_p2p_direct(
        d_r_halo_prev,                               // Receive from prev
        d_r_halo_next,                               // Receive from next
        &ipc_handles_r,                              // IPC handles for r vector
        grid_size, stream,
        rank, world_size                             // GPU device IDs for cudaMemcpyPeerAsync
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
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_start, stream));
    }
    double rs_local_old = compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventRecord(timer_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(timer_stop));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
        stats->time_dot_rs_initial_ms = elapsed_ms;
    }
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
        nvtxRangePush("SpMV");
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
        nvtxRangePop();

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
            stats->time_dot_pAp_ms += elapsed_ms;  // Granular timer
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

        nvtxRangePush("BLAS_AXPY");
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
            stats->time_axpy_update_x_ms += elapsed_ms;  // Granular timer
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
            stats->time_axpy_update_r_ms += elapsed_ms;  // Granular timer
        }
        nvtxRangePop();

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
            stats->time_dot_rs_new_ms += elapsed_ms;  // Granular timer
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
            stats->time_axpby_update_p_ms += elapsed_ms;  // Granular timer
        }

        // P2P halo exchange for p vector (160 KB vs 800 MB AllGather)
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream));
        }

        // Ensure p_local kernel is done before neighbors read it
        CUDA_CHECK(cudaStreamSynchronize(stream));
        sync_with_neighbors(rank, world_size);

        nvtxRangePush("Halo_Exchange_P2P");
        exchange_halo_p2p_direct(
            d_p_halo_prev,                               // Receive from prev
            d_p_halo_next,                               // Receive from next
            &ipc_handles_p,                              // IPC handles for p vector
            grid_size, stream,
            rank, world_size                             // GPU device IDs for cudaMemcpyPeerAsync
        );
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allgather_ms += elapsed_ms;  // Keep same name for comparison
        }
        nvtxRangePop();

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

    // Normalize granular timers by number of iterations (compute averages)
    if (config.enable_detailed_timers && stats->iterations > 0) {
        stats->time_dot_pAp_ms /= stats->iterations;
        stats->time_dot_rs_new_ms /= stats->iterations;
        stats->time_axpy_update_x_ms /= stats->iterations;
        stats->time_axpy_update_r_ms /= stats->iterations;
        stats->time_axpby_update_p_ms /= stats->iterations;
    }

    // All ranks fill their local stats first
    stats->time_total_ms = time_ms;

    // ========== MPI Stats Aggregation ==========
    // Collect max times from all ranks (slowest rank = bottleneck for wall-time)
    // Also collect min to compute load imbalance
    if (world_size > 1) {
        double local_times[6], max_times[6], min_times[6];

        local_times[0] = stats->time_total_ms;
        local_times[1] = stats->time_spmv_ms;
        local_times[2] = stats->time_blas1_ms;
        local_times[3] = stats->time_reductions_ms;
        local_times[4] = stats->time_allreduce_ms;
        local_times[5] = stats->time_allgather_ms;  // Halo P2P time

        MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_times, min_times, 6, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Store max times (bottleneck = what matters for wall-time)
            stats->time_total_ms = max_times[0];
            stats->time_spmv_ms = max_times[1];
            stats->time_blas1_ms = max_times[2];
            stats->time_reductions_ms = max_times[3];
            stats->time_allreduce_ms = max_times[4];
            stats->time_allgather_ms = max_times[5];

            // Display load imbalance (always show for multi-GPU)
            double imbalance_pct = 100.0 * (max_times[0] - min_times[0]) / max_times[0];
            printf("Total time: %.2f ms (max), %.2f ms (min) - Load imbalance: %.1f%%\n",
                   max_times[0], min_times[0], imbalance_pct);

            if (config.verbose >= 1 && config.enable_detailed_timers) {
                printf("\nDetailed Timing Breakdown:\n");
                printf("  SpMV:       %.2f ms (%.1f%%)\n", stats->time_spmv_ms, 100.0 * stats->time_spmv_ms / stats->time_total_ms);
                printf("  BLAS1:      %.2f ms (%.1f%%)\n", stats->time_blas1_ms, 100.0 * stats->time_blas1_ms / stats->time_total_ms);
                printf("  Reductions: %.2f ms (%.1f%%)\n", stats->time_reductions_ms, 100.0 * stats->time_reductions_ms / stats->time_total_ms);
                printf("  AllReduce:  %.2f ms (%.1f%%)\n", stats->time_allreduce_ms, 100.0 * stats->time_allreduce_ms / stats->time_total_ms);
                printf("  Halo P2P:   %.2f ms (%.1f%%) [was AllGather]\n", stats->time_allgather_ms, 100.0 * stats->time_allgather_ms / stats->time_total_ms);

                printf("\nGranular BLAS1 Breakdown (per-iteration avg):\n");
                printf("  Initial r=b-Ax0:  %.3f ms\n", stats->time_initial_r_ms);
                printf("  AXPY update_x:    %.3f ms\n", stats->time_axpy_update_x_ms);
                printf("  AXPY update_r:    %.3f ms\n", stats->time_axpy_update_r_ms);
                printf("  AXPBY update_p:   %.3f ms\n", stats->time_axpby_update_p_ms);

                printf("\nGranular Reductions Breakdown (per-iteration avg):\n");
                printf("  Dot rs_initial:   %.3f ms\n", stats->time_dot_rs_initial_ms);
                printf("  Dot pAp:          %.3f ms\n", stats->time_dot_pAp_ms);
                printf("  Dot rs_new:       %.3f ms\n", stats->time_dot_rs_new_ms);
            }
            printf("========================================\n");
        }
    } else if (rank == 0 && config.verbose >= 1) {
        printf("Total time: %.2f ms\n", stats->time_total_ms);
        if (config.enable_detailed_timers) {
            printf("\nDetailed Timing Breakdown:\n");
            printf("  SpMV:       %.2f ms (%.1f%%)\n", stats->time_spmv_ms, 100.0 * stats->time_spmv_ms / time_ms);
            printf("  BLAS1:      %.2f ms (%.1f%%)\n", stats->time_blas1_ms, 100.0 * stats->time_blas1_ms / time_ms);
            printf("  Reductions: %.2f ms (%.1f%%)\n", stats->time_reductions_ms, 100.0 * stats->time_reductions_ms / time_ms);
            printf("  AllReduce:  %.2f ms (%.1f%%)\n", stats->time_allreduce_ms, 100.0 * stats->time_allreduce_ms / time_ms);
            printf("  Halo P2P:   %.2f ms (%.1f%%) [was AllGather]\n", stats->time_allgather_ms, 100.0 * stats->time_allgather_ms / time_ms);

            printf("\nGranular BLAS1 Breakdown (per-iteration avg):\n");
            printf("  Initial r=b-Ax0:  %.3f ms\n", stats->time_initial_r_ms);
            printf("  AXPY update_x:    %.3f ms\n", stats->time_axpy_update_x_ms);
            printf("  AXPY update_r:    %.3f ms\n", stats->time_axpy_update_r_ms);
            printf("  AXPBY update_p:   %.3f ms\n", stats->time_axpby_update_p_ms);

            printf("\nGranular Reductions Breakdown (per-iteration avg):\n");
            printf("  Dot rs_initial:   %.3f ms\n", stats->time_dot_rs_initial_ms);
            printf("  Dot pAp:          %.3f ms\n", stats->time_dot_pAp_ms);
            printf("  Dot rs_new:       %.3f ms\n", stats->time_dot_rs_new_ms);
        }
        printf("========================================\n");
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

    // Cleanup IPC handles
    cleanup_halo_ipc(&ipc_handles_x);
    cleanup_halo_ipc(&ipc_handles_r);
    cleanup_halo_ipc(&ipc_handles_p);

    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (config.enable_detailed_timers) {
        cudaEventDestroy(timer_start);
        cudaEventDestroy(timer_stop);
    }

    return 0;
}
