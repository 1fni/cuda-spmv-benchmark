/**
 * @file spmv_stencil.h
 * @brief Header for stencil-specific CUDA kernels and functions.
 *
 * @details
 * This header provides declarations for:
 *  - CUDA kernels optimized for 5-point stencil patterns
 *  - Stencil-specific SpMV operators and utilities
 *  - Multi-GPU stencil implementations
 *
 * Author: Bouhrour Stephane
 * Date: 2025-09-25
 */

#ifndef SPMV_STENCIL_H
#define SPMV_STENCIL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Optimized CUDA kernel for 5-point stencil SpMV using ELLPACK format.
 *
 * @details
 * Kernel optimized for 5-point stencil patterns with separate handling for:
 * - Interior points: direct indexing for optimal performance
 * - Boundary points: general ELLPACK processing with boundary conditions
 *
 * @param values ELLPACK matrix values array
 * @param col_indices ELLPACK column indices array
 * @param x Input vector
 * @param y Output vector (result of A*x)
 * @param num_rows Number of matrix rows to process
 * @param width ELLPACK width (max non-zeros per row)
 * @param alpha Scalar multiplier for matrix-vector product
 * @param beta Scalar multiplier for existing result vector
 * @param grid_size 2D grid dimension (N for N×N stencil grid)
 */
__global__ void stencil5_ellpack_kernel(const double* values, const int* col_indices,
                                        const double* x, double* y, int num_rows, int width,
                                        const double alpha, const double beta, int grid_size);

#ifdef __cplusplus
}
#endif

#endif  // SPMV_STENCIL_H