/**
 * @file spmv_stencil_3d_27pt_partitioned_halo_kernel.cu
 * @brief Optimized 3D 27-point stencil SpMV kernel with Z-slab partitioning
 *
 * @details
 * Z-slab partitioning: each GPU owns contiguous Z-planes.
 * Halo contains one full XY-plane (N² elements) from neighbors.
 * 27-point stencil: center + 6 face + 12 edge + 8 corner neighbors.
 *
 * Author: Bouhrour Stephane
 */

/**
 * @brief Optimized 3D 27-point stencil SpMV kernel for partitioned CSR with Z-slab halo
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
__global__ void stencil27_csr_partitioned_halo_kernel_3d(
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size) {

    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_local)
        return;

    int global_row = row_offset + local_row;
    int N = grid_size;

    // Decompose global row to 3D coordinates: (i, j, k)
    int i = global_row / (N * N);
    int j = (global_row / N) % N;
    int k = global_row % N;

    // Decompose local row to Z-plane information
    int local_nz = n_local / (N * N);
    int local_z = local_row / (N * N);

    int row_start = row_ptr[local_row];
    int row_end = row_ptr[local_row + 1];
    double sum = 0.0;

    // Interior stencil path (all 26 neighbors are within local partition)
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1 && local_z > 0 &&
        local_z < local_nz - 1 && (row_end - row_start) == 27) {

        double val_center = x_local[local_row];

        // 6 face neighbors
        double val_xm = x_local[local_row - N * N];  // (i-1, j, k)
        double val_xp = x_local[local_row + N * N];  // (i+1, j, k)
        double val_ym = x_local[local_row - N];      // (i, j-1, k)
        double val_yp = x_local[local_row + N];      // (i, j+1, k)
        double val_zm = x_local[local_row - 1];      // (i, j, k-1)
        double val_zp = x_local[local_row + 1];      // (i, j, k+1)

        // 12 edge neighbors
        double val_xm_ym = x_local[local_row - N * N - N];  // (i-1, j-1, k)
        double val_xm_yp = x_local[local_row - N * N + N];  // (i-1, j+1, k)
        double val_xp_ym = x_local[local_row + N * N - N];  // (i+1, j-1, k)
        double val_xp_yp = x_local[local_row + N * N + N];  // (i+1, j+1, k)
        double val_xm_zm = x_local[local_row - N * N - 1];  // (i-1, j, k-1)
        double val_xm_zp = x_local[local_row - N * N + 1];  // (i-1, j, k+1)
        double val_xp_zm = x_local[local_row + N * N - 1];  // (i+1, j, k-1)
        double val_xp_zp = x_local[local_row + N * N + 1];  // (i+1, j, k+1)
        double val_ym_zm = x_local[local_row - N - 1];      // (i, j-1, k-1)
        double val_ym_zp = x_local[local_row - N + 1];      // (i, j-1, k+1)
        double val_yp_zm = x_local[local_row + N - 1];      // (i, j+1, k-1)
        double val_yp_zp = x_local[local_row + N + 1];      // (i, j+1, k+1)

        // 8 corner neighbors
        double val_xm_ym_zm = x_local[local_row - N * N - N - 1];  // (i-1, j-1, k-1)
        double val_xm_ym_zp = x_local[local_row - N * N - N + 1];  // (i-1, j-1, k+1)
        double val_xm_yp_zm = x_local[local_row - N * N + N - 1];  // (i-1, j+1, k-1)
        double val_xm_yp_zp = x_local[local_row - N * N + N + 1];  // (i-1, j+1, k+1)
        double val_xp_ym_zm = x_local[local_row + N * N - N - 1];  // (i+1, j-1, k-1)
        double val_xp_ym_zp = x_local[local_row + N * N - N + 1];  // (i+1, j-1, k+1)
        double val_xp_yp_zm = x_local[local_row + N * N + N - 1];  // (i+1, j+1, k-1)
        double val_xp_yp_zp = x_local[local_row + N * N + N + 1];  // (i+1, j+1, k+1)

        // Stencil: center=26.0, all 26 neighbors=-1.0
        sum = 26.0 * val_center - val_xm - val_xp - val_ym - val_yp - val_zm - val_zp - val_xm_ym -
              val_xm_yp - val_xp_ym - val_xp_yp - val_xm_zm - val_xm_zp - val_xp_zm - val_xp_zp -
              val_ym_zm - val_ym_zp - val_yp_zm - val_yp_zp - val_xm_ym_zm - val_xm_ym_zp -
              val_xm_yp_zm - val_xm_yp_zp - val_xp_ym_zm - val_xp_ym_zp - val_xp_yp_zm -
              val_xp_yp_zp;
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
                int halo_offset = global_col - (row_offset - (N * N));
                val = x_halo_prev[halo_offset];
            }
            // Check if column is in next Z-plane halo
            else if (x_halo_next != NULL && global_col >= row_offset + n_local &&
                     global_col < row_offset + n_local + (N * N)) {
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
