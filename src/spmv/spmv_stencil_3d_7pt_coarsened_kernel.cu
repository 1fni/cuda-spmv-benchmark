/**
 * @file spmv_stencil_3d_7pt_coarsened_kernel.cu
 * @brief Thread-coarsened 7-point stencil SpMV kernel for 3D Z-slab partitioning
 *
 * Each thread computes ITEMS_PER_THREAD consecutive points in the k-direction
 * (stride-1 dimension). Register-only approach — no shared memory, no __syncthreads.
 *
 * For 7-point stencil, each point needs 5 rows:
 *   Z-plane i-1: center column only (j, k-1..k+ITEMS)
 *   Z-plane i:   3 rows (j-1, j, j+1, each k-1..k+ITEMS)
 *   Z-plane i+1: center column only (j, k-1..k+ITEMS)
 * Total: 5 x (ITEMS+2) loads for ITEMS output points
 *
 * Author: Bouhrour Stephane
 */

/**
 * @brief Thread-coarsened 7-point stencil kernel for interior subrange
 *
 * @tparam ITEMS_PER_THREAD Number of consecutive k-points per thread
 */
template <int ITEMS_PER_THREAD>
__global__ void
stencil7_coarsened_kernel_3d(const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                             const double* __restrict__ values, const double* __restrict__ x_local,
                             const double* __restrict__ x_halo_prev,
                             const double* __restrict__ x_halo_next, double* __restrict__ y,
                             int n_local, int row_offset, int N_total, int grid_size,
                             int subrange_start, int subrange_count) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base_local_row = subrange_start + tid * ITEMS_PER_THREAD;

    if (base_local_row >= subrange_start + subrange_count)
        return;

    int N = grid_size;
    int N2 = N * N;

    int items_this_thread = ITEMS_PER_THREAD;
    if (base_local_row + ITEMS_PER_THREAD > subrange_start + subrange_count)
        items_this_thread = subrange_start + subrange_count - base_local_row;

    int base_global_row = row_offset + base_local_row;
    int i = base_global_row / N2;
    int j = (base_global_row / N) % N;
    int k = base_global_row % N;

    int local_nz = n_local / N2;
    int local_z = base_local_row / N2;

    bool all_interior =
        (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && (k + items_this_thread - 1) < N - 1 &&
         local_z > 0 && local_z < local_nz - 1);

    if (all_interior) {
        // 5 rows x (ITEMS+2) positions in registers
        // row_zm1: Z-plane i-1, column j   (only center column needed for 7pt)
        // row_ym1: Z-plane i,   column j-1
        // row_ctr: Z-plane i,   column j   (center row)
        // row_yp1: Z-plane i,   column j+1
        // row_zp1: Z-plane i+1, column j   (only center column needed for 7pt)
        double row_zm1[ITEMS_PER_THREAD + 2];  // x[local_row - N2 + (k-1..k+ITEMS)]
        double row_ym1[ITEMS_PER_THREAD + 2];  // x[local_row - N + (k-1..k+ITEMS)]
        double row_ctr[ITEMS_PER_THREAD + 2];  // x[local_row + (k-1..k+ITEMS)]
        double row_yp1[ITEMS_PER_THREAD + 2];  // x[local_row + N + (k-1..k+ITEMS)]
        double row_zp1[ITEMS_PER_THREAD + 2];  // x[local_row + N2 + (k-1..k+ITEMS)]

        int base = base_local_row;

// Load stride-1 segments (coalesced reads)
#pragma unroll
        for (int px = 0; px < ITEMS_PER_THREAD + 2; px++) {
            row_zm1[px] = x_local[base - N2 - 1 + px];
            row_ym1[px] = x_local[base - N - 1 + px];
            row_ctr[px] = x_local[base - 1 + px];
            row_yp1[px] = x_local[base + N - 1 + px];
            row_zp1[px] = x_local[base + N2 - 1 + px];
        }

// Compute ITEMS output values
#pragma unroll
        for (int m = 0; m < ITEMS_PER_THREAD; m++) {
            if (base_local_row + m >= subrange_start + subrange_count)
                break;

            int local_row = base_local_row + m;
            int csr_offset = row_ptr[local_row];

            // CSR order for 7pt interior: (i-1,j,k), (i,j-1,k), (i,j,k-1),
            //   (i,j,k) center, (i,j,k+1), (i,j+1,k), (i+1,j,k)
            double sum = 0.0;
            sum = values[csr_offset + 0] * row_zm1[m + 1];   // (i-1,j,k)
            sum += values[csr_offset + 1] * row_ym1[m + 1];  // (i,j-1,k)
            sum += values[csr_offset + 2] * row_ctr[m];      // (i,j,k-1)
            sum += values[csr_offset + 3] * row_ctr[m + 1];  // (i,j,k) center
            sum += values[csr_offset + 4] * row_ctr[m + 2];  // (i,j,k+1)
            sum += values[csr_offset + 5] * row_yp1[m + 1];  // (i,j+1,k)
            sum += values[csr_offset + 6] * row_zp1[m + 1];  // (i+1,j,k)

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
template __global__ void stencil7_coarsened_kernel_3d<2>(const int*, const int*, const double*,
                                                         const double*, const double*,
                                                         const double*, double*, int, int, int, int,
                                                         int, int);

template __global__ void stencil7_coarsened_kernel_3d<4>(const int*, const int*, const double*,
                                                         const double*, const double*,
                                                         const double*, double*, int, int, int, int,
                                                         int, int);

template __global__ void stencil7_coarsened_kernel_3d<8>(const int*, const int*, const double*,
                                                         const double*, const double*,
                                                         const double*, double*, int, int, int, int,
                                                         int, int);
