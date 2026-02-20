/**
 * @file spmv_stencil_3d_27pt_coarsened_kernel.cu
 * @brief Thread-coarsened 27-point stencil SpMV kernel for 3D Z-slab partitioning
 *
 * Each thread computes ITEMS_PER_THREAD consecutive points in the k-direction
 * (stride-1 dimension). Register-only approach — no shared memory, no __syncthreads.
 *
 * The interior fast-path uses geometric position checks and direct neighbor indexing.
 * Boundary points fall back to the standard CSR traversal with halo mapping.
 *
 * Author: Bouhrour Stephane
 */

/**
 * @brief Thread-coarsened 27-point stencil kernel for interior subrange
 *
 * @tparam ITEMS_PER_THREAD Number of consecutive k-points per thread
 *
 * Each thread processes ITEMS_PER_THREAD consecutive rows in the k-dimension.
 * For the interior path, loads 3 Z-planes x 3 Y-rows x (ITEMS+2) X-positions
 * into registers, then computes ITEMS output values from the register array.
 *
 * @param[in] row_ptr      CSR row pointers (for coefficient access + boundary fallback)
 * @param[in] col_idx      CSR column indices (boundary fallback only)
 * @param[in] values       CSR values (coefficients)
 * @param[in] x_local      Local vector partition (including halo-accessible range)
 * @param[in] x_halo_prev  Previous Z-plane halo (boundary fallback)
 * @param[in] x_halo_next  Next Z-plane halo (boundary fallback)
 * @param[out] y            Output vector partition
 * @param[in] n_local      Number of local rows
 * @param[in] row_offset   Global row offset for this partition
 * @param[in] N_total      Total matrix dimension (unused, kept for interface compat)
 * @param[in] grid_size    N (grid dimension, NxNxN)
 * @param[in] subrange_start  First local row in subrange
 * @param[in] subrange_count  Number of rows in subrange (must be divisible by ITEMS)
 */
template <int ITEMS_PER_THREAD>
__global__ void
stencil27_coarsened_kernel_3d(const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                              const double* __restrict__ values, const double* __restrict__ x_local,
                              const double* __restrict__ x_halo_prev,
                              const double* __restrict__ x_halo_next, double* __restrict__ y,
                              int n_local, int row_offset, int N_total, int grid_size,
                              int subrange_start, int subrange_count) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base_local_row = subrange_start + tid * ITEMS_PER_THREAD;

    // Early exit if this thread is entirely out of range
    if (base_local_row >= subrange_start + subrange_count)
        return;

    int N = grid_size;
    int N2 = N * N;

    // Determine how many items this thread actually processes
    int items_this_thread = ITEMS_PER_THREAD;
    if (base_local_row + ITEMS_PER_THREAD > subrange_start + subrange_count)
        items_this_thread = subrange_start + subrange_count - base_local_row;

    // Decode 3D coordinates of the base point
    int base_global_row = row_offset + base_local_row;
    int i = base_global_row / N2;
    int j = (base_global_row / N) % N;
    int k = base_global_row % N;

    // Local Z-plane info
    int local_nz = n_local / N2;
    int local_z = base_local_row / N2;

    // Check if ALL points in this thread's range are interior:
    // Same (i, j) for all ITEMS points, k ranges from k to k + items_this_thread - 1
    // All must be strictly interior in the grid AND in the local partition
    bool all_interior =
        (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && (k + items_this_thread - 1) < N - 1 &&
         local_z > 0 && local_z < local_nz - 1);

    if (all_interior) {
        // Register arrays: 3 Z-planes x 3 Y-rows x (ITEMS+2) X-positions
        // Plane indexing: 0 = i-1, 1 = i, 2 = i+1
        // Row indexing:   0 = j-1, 1 = j, 2 = j+1
        // Col indexing:   0 = k-1, 1..ITEMS = k..k+ITEMS-1, ITEMS+1 = k+ITEMS
        double plane[3][3][ITEMS_PER_THREAD + 2];

        // Load all values into register array
        // Iterate Z-planes outer, Y-rows middle, X-positions inner (coalesced)
        int base = base_local_row;

#pragma unroll
        for (int pz = 0; pz < 3; pz++) {
            int z_offset = (pz - 1) * N2;  // -N2, 0, +N2
#pragma unroll
            for (int py = 0; py < 3; py++) {
                int y_offset = (py - 1) * N;                    // -N, 0, +N
                int row_base = base + z_offset + y_offset - 1;  // start at k-1
#pragma unroll
                for (int px = 0; px < ITEMS_PER_THREAD + 2; px++) {
                    plane[pz][py][px] = x_local[row_base + px];
                }
            }
        }

// Compute ITEMS output values
#pragma unroll
        for (int m = 0; m < ITEMS_PER_THREAD; m++) {
            if (base_local_row + m >= subrange_start + subrange_count)
                break;

            int local_row = base_local_row + m;
            int csr_offset = row_ptr[local_row];

            // px index: m+1 is center (k), m is k-1, m+2 is k+1
            double sum = 0.0;

            // Z-plane i-1 (offsets 0-8)
            sum = values[csr_offset + 0] * plane[0][0][m];       // (i-1,j-1,k-1)
            sum += values[csr_offset + 1] * plane[0][0][m + 1];  // (i-1,j-1,k)
            sum += values[csr_offset + 2] * plane[0][0][m + 2];  // (i-1,j-1,k+1)
            sum += values[csr_offset + 3] * plane[0][1][m];      // (i-1,j,k-1)
            sum += values[csr_offset + 4] * plane[0][1][m + 1];  // (i-1,j,k)
            sum += values[csr_offset + 5] * plane[0][1][m + 2];  // (i-1,j,k+1)
            sum += values[csr_offset + 6] * plane[0][2][m];      // (i-1,j+1,k-1)
            sum += values[csr_offset + 7] * plane[0][2][m + 1];  // (i-1,j+1,k)
            sum += values[csr_offset + 8] * plane[0][2][m + 2];  // (i-1,j+1,k+1)

            // Z-plane i (offsets 9-17)
            sum += values[csr_offset + 9] * plane[1][0][m];       // (i,j-1,k-1)
            sum += values[csr_offset + 10] * plane[1][0][m + 1];  // (i,j-1,k)
            sum += values[csr_offset + 11] * plane[1][0][m + 2];  // (i,j-1,k+1)
            sum += values[csr_offset + 12] * plane[1][1][m];      // (i,j,k-1)
            sum += values[csr_offset + 13] * plane[1][1][m + 1];  // (i,j,k) center
            sum += values[csr_offset + 14] * plane[1][1][m + 2];  // (i,j,k+1)
            sum += values[csr_offset + 15] * plane[1][2][m];      // (i,j+1,k-1)
            sum += values[csr_offset + 16] * plane[1][2][m + 1];  // (i,j+1,k)
            sum += values[csr_offset + 17] * plane[1][2][m + 2];  // (i,j+1,k+1)

            // Z-plane i+1 (offsets 18-26)
            sum += values[csr_offset + 18] * plane[2][0][m];      // (i+1,j-1,k-1)
            sum += values[csr_offset + 19] * plane[2][0][m + 1];  // (i+1,j-1,k)
            sum += values[csr_offset + 20] * plane[2][0][m + 2];  // (i+1,j-1,k+1)
            sum += values[csr_offset + 21] * plane[2][1][m];      // (i+1,j,k-1)
            sum += values[csr_offset + 22] * plane[2][1][m + 1];  // (i+1,j,k)
            sum += values[csr_offset + 23] * plane[2][1][m + 2];  // (i+1,j,k+1)
            sum += values[csr_offset + 24] * plane[2][2][m];      // (i+1,j+1,k-1)
            sum += values[csr_offset + 25] * plane[2][2][m + 1];  // (i+1,j+1,k)
            sum += values[csr_offset + 26] * plane[2][2][m + 2];  // (i+1,j+1,k+1)

            y[local_row] = sum;
        }
    }
    // Boundary fallback: per-point CSR traversal
    else {
        for (int m = 0; m < items_this_thread; m++) {
            int local_row = base_local_row + m;
            if (local_row >= n_local)
                break;

            int row_start = row_ptr[local_row];
            int row_end = row_ptr[local_row + 1];
            double sum = 0.0;

            for (int jj = row_start; jj < row_end; jj++) {
                int global_col = col_idx[jj];
                double val;

                if (global_col >= row_offset && global_col < row_offset + n_local) {
                    val = x_local[global_col - row_offset];
                } else if (x_halo_prev != NULL && global_col >= row_offset - N2 &&
                           global_col < row_offset) {
                    val = x_halo_prev[global_col - (row_offset - N2)];
                } else if (x_halo_next != NULL && global_col >= row_offset + n_local &&
                           global_col < row_offset + n_local + N2) {
                    val = x_halo_next[global_col - (row_offset + n_local)];
                } else {
                    val = 0.0;
                }

                sum += values[jj] * val;
            }

            y[local_row] = sum;
        }
    }
}

// Explicit template instantiations
template __global__ void stencil27_coarsened_kernel_3d<2>(const int*, const int*, const double*,
                                                          const double*, const double*,
                                                          const double*, double*, int, int, int,
                                                          int, int, int);

template __global__ void stencil27_coarsened_kernel_3d<4>(const int*, const int*, const double*,
                                                          const double*, const double*,
                                                          const double*, double*, int, int, int,
                                                          int, int, int);

template __global__ void stencil27_coarsened_kernel_3d<8>(const int*, const int*, const double*,
                                                          const double*, const double*,
                                                          const double*, double*, int, int, int,
                                                          int, int, int);
