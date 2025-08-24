/**
 * @file spmv_stencil.cu
 * @brief Implements SpMV using custom CUDA kernels optimized for 5-point stencil matrices.
 *
 * @details
 * This file provides specialized SpMV implementation for 5-point stencil patterns commonly
 * found in finite difference computations. The implementation uses ELLPACK format converted
 * from CSR to optimize memory access patterns on GPU architectures:
 * - Converts CSR matrix to ELLPACK format for better memory coalescing
 * - Provides two kernel variants: generic and pattern-optimized for diffusion problems
 * - Implements timing measurements using CUDA events
 * - Handles both interior points (regular 5-point pattern) and boundary conditions
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#include <stdio.h>
#include "spmv.h"
#include "io.h"

ELLPACKMatrix ellpack_matrix; ///< Global ELLPACK matrix structure used by stencil operator

// GPU device memory pointers
static double *d_values = nullptr;  ///< Device memory for ELLPACK matrix values
static int *d_indices = nullptr;    ///< Device memory for ELLPACK column indices  
static double *dX = nullptr;        ///< Device memory for input vector x
static double *dY = nullptr;        ///< Device memory for output vector y

// SpMV computation parameters
static const double alpha = 1.0;    ///< Alpha coefficient for SpMV operation (y = alpha*A*x + beta*y)
static const double beta = 0.0;     ///< Beta coefficient for SpMV operation (y = alpha*A*x + beta*y)

/**
 * @brief CUDA kernel for SpMV optimized for 5-point stencil patterns with separate handling for interior and boundary points.
 * @details This kernel distinguishes between interior grid points (which follow regular 5-point stencil pattern)
 * and boundary/corner points (which require general ELLPACK processing). Interior points use direct indexing
 * for optimal performance, while boundary points use loop-based processing.
 * @param data ELLPACK matrix values array
 * @param col_indices ELLPACK column indices array  
 * @param vec Input vector x
 * @param result Output vector y (result of A*x)
 * @param num_rows Number of matrix rows
 * @param max_nonzero_per_row Maximum non-zeros per row (ELLPACK width)
 * @param alpha Scalar multiplier for matrix-vector product
 * @param beta Scalar multiplier for existing result vector (not used in current implementation)
 */
__global__ void ellpack_matvec_optimized_diffusion_pattern_middle_and_else(const double * data, const int* col_indices, const double * vec, double * result, int num_rows, int max_nonzero_per_row, const double alpha,  const double beta, int grid_size) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < num_rows){

		// Separate the interior (middle) stencil convolution from borders/corners:
		// on sufficiently large grids the interior pattern represents by far the heaviest workload,
		// and handling it separately minimizes warp divergence.
		// Further splitting the minor patterns (corners/boundaries) yields negligible benefit.
		// Follow logic with grid flatten and represent by x vector
		// Where the sparse matrice stores the convolution pattern value 

		// Applies the 5-point stencil to interior rows of a structured grid, performing coalesced memory accesses.
		// The interior (middle) stencil convolution is isolated from borders and corners because on sufficiently
		// large grids it represents by far the heaviest workload; handling it separately minimizes warp divergence
		// on the critical path. Further subdividing the minor border/corner patterns yields negligible performance benefit.
		
		// Convert 1D row index to 2D grid coordinates
		int i = row / grid_size;  // row in 2D grid
		int j = row % grid_size;  // column in 2D grid
		
		// Check if interior point (not on boundaries)
		if (i > 0 && i < grid_size-1 && j > 0 && j < grid_size-1) {
																//if ((row < num_rows - num_rows) && (row > num_rows- 1) && (row % num_rows!= 0) && (row % (num_rows- 1) != 0)) {//milieu
																//if ((row < num_rows - size_arcfile) && (row > size_arcfile - 1) && (row % size_arcfile != 0) && (row % (size_arcfile - 1) != 0)) {//milieu
			double sum = 0.0f;
			int offset = row * max_nonzero_per_row;
			sum += data[offset] * vec[row - grid_size];  // North neighbor
			sum += data[offset + 1] * vec[row - 1];      // West neighbor
			sum += data[offset + 2] * vec[row];          // Center point
			sum += data[offset + 3] * vec[row + 1];      // East neighbor
			sum += data[offset + 4] * vec[row + grid_size]; // South neighbor
			result[row] = alpha*sum;
			//printf("row %d %lf %lf, %lf %lf, %lf %lf, %lf %lf, %lf %lf\n", row, data[offset] , vec[row - size_grid],data[offset+1] , vec[row - 1],data[offset+2] , vec[row],data[offset+3] , vec[row + 1],data[offset+4] , vec[row + size_grid]);
		}
		else{
			//else if((row == 0) || (row == num_rows- 1) || (row == num_rows - num_rows) || ((row == num_rows - 1)) || ((row < num_rows- 1) && (row > 0)) || ((row > (num_rows - num_rows)) && (row < (num_rows - 1))) || ((row != 0) && (row != (num_rows - num_rows)) && ((row % num_rows) == 0)) || ((row != (num_rows- 1)) && (row != (num_rows - 1)) && ((row % (num_rows- 1)) == 0)) ) {//others, edge and corners
			//else if((row == 0) || (row == size_arcfile - 1) || (row == num_rows - size_arcfile) || ((row == num_rows - 1)) || ((row < size_arcfile - 1) && (row > 0)) || ((row > (num_rows - size_arcfile)) && (row < (num_rows - 1))) || ((row != 0) && (row != (num_rows - size_arcfile)) && ((row % size_arcfile) == 0)) || ((row != (size_arcfile - 1)) && (row != (num_rows - 1)) && ((row % (size_arcfile - 1)) == 0)) ) {//others, edge and corners
			//printf("elserow %d\n", row);
			double sum = 0.0f;
			int offset = row * max_nonzero_per_row;

#pragma unroll
			for (int i = 0; i < max_nonzero_per_row; ++i) {
				int col = col_indices[row * max_nonzero_per_row + i];
				if (col >= 0) {  // Vérifie si l'indice de colonne est valide
					sum += data[offset + i] * vec[col];
				}
			}

			// Stocke le résultat final
			result[row] = alpha*sum;
		}
		}
		}

/**
 * @brief Generic CUDA kernel for ELLPACK SpMV computation.
 * @details This kernel performs standard ELLPACK format SpMV using loop-based processing
 * for all matrix rows. Each thread processes one matrix row by iterating through its
 * non-zero elements stored in ELLPACK format.
 * @param data ELLPACK matrix values array
 * @param col_indices ELLPACK column indices array
 * @param vec Input vector x  
 * @param result Output vector y (result of A*x)
 * @param num_rows Number of matrix rows
 * @param max_nonzero_per_row Maximum non-zeros per row (ELLPACK width)
 * @param alpha Scalar multiplier for matrix-vector product
 * @param beta Scalar multiplier for existing result vector
 */
__global__ void ellpack_matvec_optimized_diffusion(const double * data, const int* col_indices, const double * vec, double * result, int num_rows, int max_nonzero_per_row, const double alpha,  const double beta) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	// Vérifie si le thread traite une ligne valide
	if (row < num_rows) {
		double sum = 0.0f;

		// Utilise des accès directs au lieu d'une boucle pour les colonnes non nulles
#pragma unroll
		for (int i = 0; i < max_nonzero_per_row; ++i) {
			int col = col_indices[row * max_nonzero_per_row + i];
			if (col >= 0) {  // Vérifie si l'indice de colonne est valide
				sum += alpha * data[row * max_nonzero_per_row + i] + beta * vec[col];
			}
		}

		// Stocke le résultat final
		result[row] = sum;
	}
}


/**
 * @brief Converts CSR matrix format to ELLPACK format for optimized GPU processing.
 * @details This function transforms a CSR (Compressed Sparse Row) matrix into ELLPACK format
 * by determining the maximum row width and creating padded arrays for values and indices.
 * ELLPACK format enables better memory coalescing on GPU by storing matrix data in
 * row-major order with uniform row lengths.
 * @param csr_matrix Pointer to the source CSR matrix structure
 * @return int 0 on success, non-zero on failure
 */
int build_ellpack_from_csr_struct(CSRMatrix *csr_matrix){
	int max_nonzeros = 0;
	for (int i = 0; i < csr_matrix->nb_rows; ++i) {
		int row_nonzeros = csr_matrix->row_ptr[i + 1] - csr_matrix->row_ptr[i];
		if (row_nonzeros > max_nonzeros) {
			max_nonzeros = row_nonzeros;
		}
	}
	ellpack_matrix.ell_width = (max_nonzeros > MAX_WIDTH) ? MAX_WIDTH : max_nonzeros;
	printf("ELL WIDTH %d\n", ellpack_matrix.ell_width);

	ellpack_matrix.nb_rows = csr_matrix->nb_rows;
	ellpack_matrix.nb_cols = csr_matrix->nb_cols;
	ellpack_matrix.nb_nonzeros = csr_matrix->nb_nonzeros;
	// grid_size will be assigned in stencil5_init

	int total_ell_elements = csr_matrix->nb_rows * ellpack_matrix.ell_width;
	ellpack_matrix.indices = (int *)calloc(total_ell_elements, sizeof(int));
	ellpack_matrix.values = (double *)calloc(total_ell_elements, sizeof(double));

	// Debug: afficher la structure CSR avant conversion
	//printf("CSR Matrix debug:\n");
	//for (int i = 0; i < csr_matrix->nb_rows; ++i) {
	//	printf("Row %d CSR: ", i);
	//	for (int j = csr_matrix->row_ptr[i]; j < csr_matrix->row_ptr[i + 1]; ++j) {
	//		printf("(col:%d val:%lf) ", csr_matrix->col_indices[j], csr_matrix->values[j]);
	//	}
	//	printf("\n");
	//}
	
	//populate ELLPACK format
	for (int i = 0; i < csr_matrix->nb_rows; ++i) {
		int ell_index = 0;
		for (int j = csr_matrix->row_ptr[i]; j < csr_matrix->row_ptr[i + 1]; ++j) {
			if (ell_index < ellpack_matrix.ell_width) {
				ellpack_matrix.indices[i * ellpack_matrix.ell_width + ell_index] = csr_matrix->col_indices[j];
				ellpack_matrix.values[i * ellpack_matrix.ell_width + ell_index] = csr_matrix->values[j];
				ell_index++;
			} else {
				break;
			}
		}
	}
	// Debug: afficher la conversion pour toutes les lignes
	//for (int i = 0; i < ellpack_matrix.nb_rows; i++) {
	//	printf("Row %d ELLPACK: ", i);
	//	for (int j = 0; j < ellpack_matrix.ell_width; j++) {
	//		printf("(col:%d val:%lf) ", ellpack_matrix.indices[ellpack_matrix.ell_width*i + j], ellpack_matrix.values[ellpack_matrix.ell_width*i + j]);
	//	}
	//	printf("\n");
	//}
	return 0;
}

/**
 * @brief Initializes the stencil5 SpMV operator with matrix data and GPU memory allocation.
 * @details This function performs the complete initialization sequence for stencil-based SpMV:
 * 1. Converts input MatrixData to CSR format using global csr_mat structure
 * 2. Converts CSR to ELLPACK format for optimized GPU access patterns
 * 3. Allocates GPU device memory for matrix values, indices, and input/output vectors
 * 4. Transfers matrix data from host to device memory
 * @param mat Pointer to MatrixData structure containing the sparse matrix
 * @return int 0 on successful initialization, non-zero on failure
 */
int stencil5_init(MatrixData* mat) {

	//build CSR from MatrixData* mat then convert in ELLPACK
	build_csr_struct(mat);
	build_ellpack_from_csr_struct(&csr_mat);
	
	// Store grid_size from MatrixData for stencil operations
	ellpack_matrix.grid_size = mat->grid_size;

	size_t size_values = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
	size_t size_indices = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int);
	size_t size_vec = ellpack_matrix.nb_rows * sizeof(double);

	// Allocation GPU
	CUDA_CHECK(cudaMalloc((void**)&d_values, size_values));
	CUDA_CHECK(cudaMalloc((void**)&d_indices, size_indices));
	CUDA_CHECK(cudaMalloc((void**)&dX, size_vec));
	CUDA_CHECK(cudaMalloc((void**)&dY, size_vec));

	// Transfert H2D
	CUDA_CHECK(cudaMemcpy(d_values, ellpack_matrix.values, size_values, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_indices, ellpack_matrix.indices, size_indices, cudaMemcpyHostToDevice));
	
	// Synchronisation pour s'assurer que le transfert est terminé
	CUDA_CHECK(cudaDeviceSynchronize());
	
	// Debug: vérifier les données sur GPU après transfert
	//printf("Verification GPU data after H2D transfer:\n");
	//double *h_verify_values = (double*)malloc(size_values);
	//int *h_verify_indices = (int*)malloc(size_indices);
	//
	//CUDA_CHECK(cudaMemcpy(h_verify_values, d_values, size_values, cudaMemcpyDeviceToHost));
	//CUDA_CHECK(cudaMemcpy(h_verify_indices, d_indices, size_indices, cudaMemcpyDeviceToHost));
	//
	//for (int i = 0; i < ellpack_matrix.nb_rows; i++) {
	//	printf("GPU Row %d: ", i);
	//	for (int j = 0; j < ellpack_matrix.ell_width; j++) {
	//		int idx = i * ellpack_matrix.ell_width + j;
	//		printf("(col:%d val:%lf) ", h_verify_indices[idx], h_verify_values[idx]);
	//	}
	//	printf("\n");
	//}
	//
	//free(h_verify_values);
	//free(h_verify_indices);
	return 0;
}

/**
 * @brief Executes the stencil5 SpMV computation with precise kernel timing measurement.
 * @details This function performs the complete SpMV execution workflow:
 * 1. Transfers input vector from host to device memory
 * 2. Launches optimized CUDA kernel for 5-point stencil pattern
 * 3. Measures precise kernel execution time using CUDA events
 * 4. Transfers result vector back to host memory
 * 5. Computes and displays checksum for verification
 * @param x Input vector (host memory)
 * @param y Output vector (host memory) - will contain result of A*x
 * @param kernel_time_ms Output parameter for kernel execution time in milliseconds
 * @return int 0 on successful execution, non-zero on failure
 */
int stencil5_run_timed(const double* x, double* y, double* kernel_time_ms) {
	size_t size_vec = ellpack_matrix.nb_rows * sizeof(double);
	
	CUDA_CHECK(cudaMemcpy(dX, x, size_vec, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel SpMV
	cudaEventRecord(start);
	int threads = 32;
	int blocks = (ellpack_matrix.nb_rows + threads - 1) / threads;
	//ellpack_matvec_optimized_diffusion<<<blocks, threads>>>(d_values, d_indices, dX, dY,ellpack_matrix.nb_rows, ellpack_matrix.ell_width, alpha, beta);
	printf("Matrix rows: %d, Grid size: %d\n", ellpack_matrix.nb_rows, ellpack_matrix.grid_size);
	ellpack_matvec_optimized_diffusion_pattern_middle_and_else<<<blocks, threads>>>(d_values, d_indices, dX, dY,ellpack_matrix.nb_rows, ellpack_matrix.ell_width, alpha, beta, ellpack_matrix.grid_size);

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float computeTime;
	cudaEventElapsedTime(&computeTime, start, stop);
	
	// Return precise kernel timing for metrics calculation
	*kernel_time_ms = (double)computeTime;
	printf("[Stencil5] Kernel time: %.3f ms\n", computeTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy result
	CUDA_CHECK(cudaMemcpy(y, dY, size_vec, cudaMemcpyDeviceToHost));

	// Print result
	printf("Result:\n");
	double check_sum = 0.0;
	for (int i = 0; i < ellpack_matrix.nb_rows; i++) {
		check_sum += y[i];
	}
	printf("check_sum %le\n", check_sum);

	return 0;
}

/**
 * @brief Frees all GPU device memory allocated by the stencil5 operator.
 * @details This cleanup function releases all CUDA device memory allocations:
 * - Matrix values and indices arrays
 * - Input and output vector device memory
 * Should be called when the stencil5 operator is no longer needed.
 */
void stencil5_free() {
	printf("[STENCIL5] Cleaning up\n");
	
	// Free GPU memory  
	CUDA_CHECK(cudaFree(d_values));
	CUDA_CHECK(cudaFree(d_indices));
	CUDA_CHECK(cudaFree(dX));
	CUDA_CHECK(cudaFree(dY));
	
	// Free host ELLPACK arrays
	if (ellpack_matrix.indices) {
		free(ellpack_matrix.indices);
		ellpack_matrix.indices = NULL;
	}
	if (ellpack_matrix.values) {
		free(ellpack_matrix.values);
		ellpack_matrix.values = NULL;
	}
	
	// Free host CSR arrays used to build ELLPACK
	if (csr_mat.row_ptr) {
		free(csr_mat.row_ptr);
		csr_mat.row_ptr = NULL;
	}
	if (csr_mat.col_indices) {
		free(csr_mat.col_indices);
		csr_mat.col_indices = NULL;
	}
	if (csr_mat.values) {
		free(csr_mat.values);
		csr_mat.values = NULL;
	}
}

/**
 * @brief Global SpmvOperator structure for stencil5 implementation.
 * @details This operator provides the interface for 5-point stencil SpMV computations:
 * - name: "stencil5" identifier for operator selection
 * - init: stencil5_init function for GPU memory setup and matrix conversion
 * - run: stencil5_run function for kernel execution and timing
 * - free: stencil5_free function for resource cleanup
 */
SpmvOperator SPMV_STENCIL5 = {
	.name = "stencil5",
	.init = stencil5_init,
	.run_timed = stencil5_run_timed,
	.free = stencil5_free
};

