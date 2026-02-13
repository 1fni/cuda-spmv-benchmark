/**
 * @file spmv_stencil_3d_partitioned_halo_kernel.cu
 * @brief Optimized 3D 7-point stencil SpMV kernel with Z-slab partitioning
 *
 * @details
 * Z-slab partitioning: each GPU owns contiguous Z-planes.
 * Halo contains one full XY-plane (N² elements) from neighbors.
 *
 * Author: Bouhrour Stephane
 */

/**
 * @brief Optimized 3D 7-point stencil SpMV kernel for partitioned CSR with Z-slab halo
 *
 * @param[in] row_ptr CSR row pointers
 * @param[in] col_idx CSR column indices
 * @param[in] values CSR values
 * @param[in] x_local Local vector partition
 * @param[in] x_halo_prev Previous Z-plane halo (NULL if rank==0)
 * @param[in] x_halo_next Next Z-plane halo (NULL if rank==world_size-1)
 * @param[out] y Output vector partition
 * @param[in] n_local Number of local rows
 * @param[in] row_offset Global row offset for this partition
 * @param[in] N_total Total grid dimension (NxNxN grid)
 * @param[in] grid_size N (used for stencil pattern)
 */
__global__ void stencil7_csr_partitioned_halo_kernel_3d(
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size) {

    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_local)
        return;

    int global_row = row_offset + local_row;
    int N = grid_size;  // Grid dimension

    // Decompose global row to 3D coordinates: (i, j, k)
    // Global index = i*N² + j*N + k
    int i = global_row / (N * N);
    int j = (global_row / N) % N;
    int k = global_row % N;

    // Decompose local row to Z-plane information
    int local_nz = n_local / (N * N);  // Number of local Z-planes
    int local_z = local_row / (N * N);

    int row_start = row_ptr[local_row];
    int row_end = row_ptr[local_row + 1];
    double sum = 0.0;

    // Interior stencil path (all 6 neighbors are accessible)
    // Interior check: interior in X, Y, Z and within local partition
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1 && local_z > 0 &&
        local_z < local_nz - 1 && (row_end - row_start) == 7) {

        // Compute global indices of the 7-point stencil
        int idx_center = global_row;
        int idx_west = global_row - 1;        // (i, j, k-1) in terms of linear index within plane
        int idx_east = global_row + 1;        // (i, j, k+1)
        int idx_north = global_row - N;       // (i-1, j, k) in terms of Y direction
        int idx_south = global_row + N;       // (i+1, j, k)
        int idx_up = global_row - (N * N);    // (i, j-1, k) in terms of Z direction
        int idx_down = global_row + (N * N);  // (i, j+1, k)

        // All neighbors are within local partition (interior check ensures this)
        double val_center = x_local[idx_center - row_offset];
        double val_west = x_local[idx_west - row_offset];
        double val_east = x_local[idx_east - row_offset];
        double val_north = x_local[idx_north - row_offset];
        double val_south = x_local[idx_south - row_offset];
        double val_up = x_local[idx_up - row_offset];
        double val_down = x_local[idx_down - row_offset];

        // Stencil pattern: center=6.0, neighbors=-1.0
        sum = 6.0 * val_center - val_west - val_east - val_north - val_south - val_up - val_down;
    }
    // Boundary/corner: CSR traversal with halo mapping
    else {
        for (int jj = row_start; jj < row_end; jj++) {
            int global_col = col_idx[jj];
            double val;

            // Check if column is in local partition
            if (global_col >= row_offset && global_col < row_offset + n_local) {
                val = x_local[global_col - row_offset];
            }
            // Check if column is in previous Z-plane halo
            else if (x_halo_prev != NULL && global_col >= row_offset - (N * N) &&
                     global_col < row_offset) {
                // Halo stores one full XY-plane = N² elements
                // Offset within the plane: local_index_in_plane from the previous Z-plane
                int halo_offset = global_col - (row_offset - (N * N));
                val = x_halo_prev[halo_offset];
            }
            // Check if column is in next Z-plane halo
            else if (x_halo_next != NULL && global_col >= row_offset + n_local &&
                     global_col < row_offset + n_local + (N * N)) {
                // Halo stores one full XY-plane = N² elements
                int halo_offset = global_col - (row_offset + n_local);
                val = x_halo_next[halo_offset];
            }
            // Column is outside known regions (boundary of domain)
            else {
                val = 0.0;
            }

            sum += values[jj] * val;
        }
    }

    y[local_row] = sum;
}
