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
static bool ellpack_structure_built = false; ///< Flag to avoid redundant ELLPACK reconstruction

// GPU device memory pointers
static double *d_values = nullptr;  ///< Device memory for ELLPACK matrix values
static int *d_indices = nullptr;    ///< Device memory for ELLPACK column indices  
static double *dX = nullptr;        ///< Device memory for input vector x
static double *dY = nullptr;        ///< Device memory for output vector y

// SpMV computation parameters
static const double alpha = 1.0;    ///< Alpha coefficient for SpMV operation (y = alpha*A*x + beta*y)
static const double beta = 0.0;     ///< Beta coefficient for SpMV operation (y = alpha*A*x + beta*y)

/**
 * @brief Ensures ELLPACK structure is built from matrix data (shared across all kernels)
 * @param mat Matrix data to convert to ELLPACK format
 * @return 0 on success, non-zero on failure
 */
int ensure_ellpack_structure_built(MatrixData* mat) {
    if (ellpack_structure_built) {
        // Verify grid_size consistency
        if (ellpack_matrix.grid_size != mat->grid_size) {
            printf("Warning: Grid size mismatch (%d vs %d), rebuilding ELLPACK\n", 
                   ellpack_matrix.grid_size, mat->grid_size);
            ellpack_structure_built = false;
        } else {
            return 0;  // Already built and consistent
        }
    }
    
    printf("Building ELLPACK structure (CSR conversion)...\n");
    build_csr_struct(mat);
    build_ellpack_from_csr_local(&csr_mat);
    ellpack_matrix.grid_size = mat->grid_size;
    ellpack_structure_built = true;
    printf("ELLPACK structure built: %d rows, width %d\n", ellpack_matrix.nb_rows, ellpack_matrix.ell_width);
    
    return 0;
}

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
extern "C" __global__ void stencil5_ellpack_kernel(const double * data, const int* col_indices, const double * vec, double * result, int num_rows, int max_nonzero_per_row, const double alpha,  const double beta, int grid_size) {
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
			// Optimized memory access order: group spatially adjacent vec[] accesses first
			sum += data[offset + 1] * vec[row - 1];      // West neighbor (stride -1)
			sum += data[offset + 2] * vec[row];          // Center point (stride 0)
			sum += data[offset + 3] * vec[row + 1];      // East neighbor (stride +1)
			sum += data[offset] * vec[row - grid_size];  // North neighbor (stride -grid_size)
			sum += data[offset + 4] * vec[row + grid_size]; // South neighbor (stride +grid_size)
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

__global__ void stencil5_optimized_ellpack_kernel(const double* __restrict__ data, 
                                                    const int* __restrict__ col_indices,
                                                    const double* __restrict__ vec, 
                                                    double* __restrict__ result,
                                                    int num_rows, int max_nonzero_per_row, 
                                                    double alpha, double beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= num_rows) return;
    
    double sum = 0.0;
    int base_idx = row * max_nonzero_per_row;
    
    // Unroll the loop for 5-point stencil (max_nonzero_per_row should be 5 for stencil matrices)
    #pragma unroll 5
    for (int j = 0; j < max_nonzero_per_row; j++) {
        int col = col_indices[base_idx + j];
        if (col >= 0 && col < num_rows) {  // Valid column index
            sum += data[base_idx + j] * vec[col];
        }
    }
    
    result[row] = alpha * sum + beta * result[row];
}

__global__ void stencil5_shared_memory_ellpack_kernel(const double* __restrict__ data,
                                                      const int* __restrict__ col_indices,
                                                      const double* __restrict__ vec,
                                                      double* __restrict__ result,
                                                      int num_rows, int max_nonzero_per_row,
                                                      double alpha, double beta, int grid_size) {
    const int TILE_SIZE = 32;
    __shared__ double tile[TILE_SIZE + 2][TILE_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int global_x = bx * TILE_SIZE + tx;
    int global_y = by * TILE_SIZE + ty;
    int global_idx = global_y * grid_size + global_x;
    
    if (global_x >= grid_size || global_y >= grid_size) return;
    
    // Load tile center
    tile[ty + 1][tx + 1] = vec[global_idx];
    
    // Load halo regions cooperatively
    if (ty == 0 && global_y > 0) 
        tile[0][tx + 1] = vec[(global_y - 1) * grid_size + global_x];
    if (ty == TILE_SIZE - 1 && global_y < grid_size - 1)
        tile[TILE_SIZE + 1][tx + 1] = vec[(global_y + 1) * grid_size + global_x];
    if (tx == 0 && global_x > 0)
        tile[ty + 1][0] = vec[global_y * grid_size + global_x - 1];
    if (tx == TILE_SIZE - 1 && global_x < grid_size - 1)
        tile[ty + 1][TILE_SIZE + 1] = vec[global_y * grid_size + global_x + 1];
    
    __syncthreads();
    
    // Read stencil coefficients from ELLPACK data for this row
    int base_idx = global_idx * max_nonzero_per_row;
    double sum = 0.0;
    
    // Find center coefficient (should be at column = global_idx)
    double center_val = 0.0;
    for (int j = 0; j < max_nonzero_per_row; j++) {
        int col = col_indices[base_idx + j];
        if (col == global_idx) {
            center_val = data[base_idx + j];
            break;
        }
    }
    
    // Use shared memory for computation with real center coefficient
    sum = center_val * tile[ty + 1][tx + 1];
    
    // Add neighbors using ELLPACK data
    for (int j = 0; j < max_nonzero_per_row; j++) {
        int col = col_indices[base_idx + j];
        if (col != global_idx && col >= 0 && col < num_rows) {
            // Determine if neighbor is available in shared memory
            int neighbor_y = col / grid_size;
            int neighbor_x = col % grid_size;
            
            // Check if neighbor is within shared memory tile
            if (abs(neighbor_y - global_y) <= 1 && abs(neighbor_x - global_x) <= 1) {
                int tile_y = neighbor_y - (by * TILE_SIZE) + 1;
                int tile_x = neighbor_x - (bx * TILE_SIZE) + 1;
                if (tile_y >= 0 && tile_y < TILE_SIZE + 2 && tile_x >= 0 && tile_x < TILE_SIZE + 2) {
                    sum += data[base_idx + j] * tile[tile_y][tile_x];
                } else {
                    sum += data[base_idx + j] * vec[col];  // Fallback to global memory
                }
            } else {
                sum += data[base_idx + j] * vec[col];  // Fallback to global memory
            }
        }
    }
    
    result[global_idx] = alpha * sum + beta * result[global_idx];
}

__global__ void stencil5_coarsened_ellpack_kernel(const double* __restrict__ data,
                                                   const int* __restrict__ col_indices,
                                                   const double* __restrict__ vec,
                                                   double* __restrict__ result,
                                                   int num_rows, int max_nonzero_per_row,
                                                   double alpha, double beta) {
    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple rows per thread (thread coarsening for better throughput)
    for (int offset = 0; offset < 4 && base_idx + offset * blockDim.x * gridDim.x < num_rows; offset++) {
        int row = base_idx + offset * blockDim.x * gridDim.x;
        
        if (row >= num_rows) continue;
        
        double sum = 0.0;
        int ellpack_base = row * max_nonzero_per_row;
        
        // Process all nonzeros in this row using real ELLPACK data
        for (int j = 0; j < max_nonzero_per_row; j++) {
            int col = col_indices[ellpack_base + j];
            if (col >= 0 && col < num_rows) {  // Valid column index
                sum += data[ellpack_base + j] * vec[col];
            }
        }
        
        result[row] = alpha * sum + beta * result[row];
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
int build_ellpack_from_csr_local(CSRMatrix *csr_matrix){
	printf("🔄 Converting CSR to ELLPACK format...\n");
	fflush(stdout);
	
	printf("   ➤ Finding maximum row width...\n");
	fflush(stdout);
	
	int max_nonzeros = 0;
	for (int i = 0; i < csr_matrix->nb_rows; ++i) {
		int row_nonzeros = csr_matrix->row_ptr[i + 1] - csr_matrix->row_ptr[i];
		if (row_nonzeros > max_nonzeros) {
			max_nonzeros = row_nonzeros;
		}
	}
	ellpack_matrix.ell_width = (max_nonzeros > MAX_WIDTH) ? MAX_WIDTH : max_nonzeros;
	printf("   ➤ ELLPACK width determined: %d\n", ellpack_matrix.ell_width);
	fflush(stdout);

	ellpack_matrix.nb_rows = csr_matrix->nb_rows;
	ellpack_matrix.nb_cols = csr_matrix->nb_cols;
	ellpack_matrix.nb_nonzeros = csr_matrix->nb_nonzeros;
	// grid_size will be assigned in stencil5_init

	printf("   ➤ Allocating ELLPACK arrays (%d x %d elements)...\n", 
		   csr_matrix->nb_rows, ellpack_matrix.ell_width);
	fflush(stdout);
	
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
	
	printf("   ➤ Populating ELLPACK format from CSR data...\n");
	fflush(stdout);
	
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
	printf("✅ ELLPACK conversion completed successfully\n");
	fflush(stdout);
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
	// Ensure ELLPACK structure is built (shared across all kernels)
	ensure_ellpack_structure_built(mat);

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

int stencil5_optimized_init(MatrixData* mat) {
	// Ensure ELLPACK structure is built (shared across all kernels)
	ensure_ellpack_structure_built(mat);

	size_t size_values = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
	size_t size_indices = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int);
	size_t size_vec = ellpack_matrix.nb_rows * sizeof(double);

	// Allocate GPU memory for ELLPACK data
	CUDA_CHECK(cudaMalloc((void**)&d_values, size_values));
	CUDA_CHECK(cudaMalloc((void**)&d_indices, size_indices));
	CUDA_CHECK(cudaMalloc((void**)&dX, size_vec));
	CUDA_CHECK(cudaMalloc((void**)&dY, size_vec));

	// Transfer ELLPACK data to GPU
	CUDA_CHECK(cudaMemcpy(d_values, ellpack_matrix.values, size_values, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_indices, ellpack_matrix.indices, size_indices, cudaMemcpyHostToDevice));
	
	CUDA_CHECK(cudaDeviceSynchronize());
	
	return 0;
}

int stencil5_optimized_run_timed(const double* x, double* y, double* kernel_time_ms) {
	size_t size_vec = ellpack_matrix.grid_size * ellpack_matrix.grid_size * sizeof(double);
	
	CUDA_CHECK(cudaMemset(dY, 0, size_vec));
	CUDA_CHECK(cudaMemcpy(dX, x, size_vec, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Use optimized ELLPACK kernel with loop unrolling
	cudaEventRecord(start);
	
	dim3 block_size(256);
	dim3 grid_size((ellpack_matrix.nb_rows + block_size.x - 1) / block_size.x);
	
	stencil5_optimized_ellpack_kernel<<<grid_size, block_size>>>(d_values, d_indices, dX, dY, 
	                                                              ellpack_matrix.nb_rows, 
	                                                              ellpack_matrix.ell_width, 
	                                                              alpha, beta);

	CUDA_CHECK(cudaDeviceSynchronize());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float computeTime;
	cudaEventElapsedTime(&computeTime, start, stop);
	*kernel_time_ms = (double)computeTime;
	
	printf("[Stencil5-Optimized] Kernel time: %.3f ms\n", computeTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CUDA_CHECK(cudaMemcpy(y, dY, size_vec, cudaMemcpyDeviceToHost));

	double check_sum = 0.0;
	for (int i = 0; i < ellpack_matrix.grid_size * ellpack_matrix.grid_size; i++) {
		check_sum += y[i];
	}
	printf("check_sum %le\n", check_sum);

	return 0;
}

void stencil5_optimized_free() {
	printf("[STENCIL5-OPTIMIZED] Cleaning up\n");
	CUDA_CHECK(cudaFree(d_values));
	CUDA_CHECK(cudaFree(d_indices));
	CUDA_CHECK(cudaFree(dX));
	CUDA_CHECK(cudaFree(dY));
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
	
	// Reset output vector and copy input vector
	CUDA_CHECK(cudaMemset(dY, 0, size_vec));
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
	stencil5_ellpack_kernel<<<blocks, threads>>>(d_values, d_indices, dX, dY,ellpack_matrix.nb_rows, ellpack_matrix.ell_width, alpha, beta, ellpack_matrix.grid_size);

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
	.run_device = NULL,

	.free = stencil5_free
};

SpmvOperator SPMV_STENCIL5_OPTIMIZED = {
	.name = "stencil5-opt",
	.init = stencil5_optimized_init,
	.run_timed = stencil5_optimized_run_timed,
	.run_device = NULL,

	.free = stencil5_optimized_free
};

// Shared memory variables (separate from regular stencil)
static double *d_values_shared = nullptr;
static int *d_indices_shared = nullptr; 
static double *dX_shared = nullptr;
static double *dY_shared = nullptr;

int stencil5_shared_init(MatrixData* mat) {
	// Ensure ELLPACK structure is built (shared across all kernels)
	ensure_ellpack_structure_built(mat);

	size_t size_values = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
	size_t size_indices = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int);
	size_t size_vec = ellpack_matrix.nb_rows * sizeof(double);

	CUDA_CHECK(cudaMalloc((void**)&d_values_shared, size_values));
	CUDA_CHECK(cudaMalloc((void**)&d_indices_shared, size_indices));
	CUDA_CHECK(cudaMalloc((void**)&dX_shared, size_vec));
	CUDA_CHECK(cudaMalloc((void**)&dY_shared, size_vec));

	CUDA_CHECK(cudaMemcpy(d_values_shared, ellpack_matrix.values, size_values, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_indices_shared, ellpack_matrix.indices, size_indices, cudaMemcpyHostToDevice));
	
	CUDA_CHECK(cudaDeviceSynchronize());
	
	return 0;
}

int stencil5_shared_run_timed(const double* x, double* y, double* kernel_time_ms) {
	size_t size_vec = ellpack_matrix.grid_size * ellpack_matrix.grid_size * sizeof(double);
	
	CUDA_CHECK(cudaMemset(dY_shared, 0, size_vec));
	CUDA_CHECK(cudaMemcpy(dX_shared, x, size_vec, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Use shared memory kernel with 2D blocks
	cudaEventRecord(start);
	
	dim3 block_size(32, 32);
	dim3 grid_size((ellpack_matrix.grid_size + 31) / 32, (ellpack_matrix.grid_size + 31) / 32);
	
	stencil5_shared_memory_ellpack_kernel<<<grid_size, block_size>>>(d_values_shared, d_indices_shared, dX_shared, dY_shared, 
	                                                                  ellpack_matrix.nb_rows, 
	                                                                  ellpack_matrix.ell_width, 
	                                                                  alpha, beta, ellpack_matrix.grid_size);

	CUDA_CHECK(cudaDeviceSynchronize());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float computeTime;
	cudaEventElapsedTime(&computeTime, start, stop);
	*kernel_time_ms = (double)computeTime;
	
	printf("[Stencil5-Shared] Kernel time: %.3f ms\n", computeTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CUDA_CHECK(cudaMemcpy(y, dY_shared, size_vec, cudaMemcpyDeviceToHost));

	double check_sum = 0.0;
	for (int i = 0; i < ellpack_matrix.grid_size * ellpack_matrix.grid_size; i++) {
		check_sum += y[i];
	}
	printf("check_sum %e\n", check_sum);

	return 0;
}

void stencil5_shared_free() {
	printf("[STENCIL5-SHARED] Cleaning up\n");
	CUDA_CHECK(cudaFree(d_values_shared));
	CUDA_CHECK(cudaFree(d_indices_shared));
	CUDA_CHECK(cudaFree(dX_shared));
	CUDA_CHECK(cudaFree(dY_shared));
}

SpmvOperator SPMV_STENCIL5_SHARED = {
	.name = "stencil5-shared",
	.init = stencil5_shared_init,
	.run_timed = stencil5_shared_run_timed,
	.run_device = NULL,

	.free = stencil5_shared_free
};

// Thread coarsening variables
static double *d_values_coarsened = nullptr;
static int *d_indices_coarsened = nullptr; 
static double *dX_coarsened = nullptr;
static double *dY_coarsened = nullptr;

// Naive ELLPACK variables
static double *d_values_naive = nullptr;
static int *d_indices_naive = nullptr; 
static double *dX_naive = nullptr;
static double *dY_naive = nullptr;

__global__ void ellpack_naive_kernel(const double* __restrict__ data,
                                      const int* __restrict__ col_indices,
                                      const double* __restrict__ vec,
                                      double* __restrict__ result,
                                      int num_rows, int max_nonzero_per_row,
                                      double alpha, double beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= num_rows) return;
    
    double sum = 0.0;
    int base_idx = row * max_nonzero_per_row;
    
    // Simple ELLPACK implementation - no optimizations
    for (int j = 0; j < max_nonzero_per_row; j++) {
        int col = col_indices[base_idx + j];
        if (col >= 0 && col < num_rows) {
            sum += data[base_idx + j] * vec[col];
        }
    }
    
    result[row] = alpha * sum + beta * result[row];
}

int stencil5_coarsened_init(MatrixData* mat) {
	// Ensure ELLPACK structure is built (shared across all kernels)
	ensure_ellpack_structure_built(mat);

	size_t size_values = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
	size_t size_indices = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int);
	size_t size_vec = ellpack_matrix.nb_rows * sizeof(double);

	CUDA_CHECK(cudaMalloc((void**)&d_values_coarsened, size_values));
	CUDA_CHECK(cudaMalloc((void**)&d_indices_coarsened, size_indices));
	CUDA_CHECK(cudaMalloc((void**)&dX_coarsened, size_vec));
	CUDA_CHECK(cudaMalloc((void**)&dY_coarsened, size_vec));

	CUDA_CHECK(cudaMemcpy(d_values_coarsened, ellpack_matrix.values, size_values, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_indices_coarsened, ellpack_matrix.indices, size_indices, cudaMemcpyHostToDevice));
	
	CUDA_CHECK(cudaDeviceSynchronize());
	
	return 0;
}

int stencil5_coarsened_run_timed(const double* x, double* y, double* kernel_time_ms) {
	size_t size_vec = ellpack_matrix.grid_size * ellpack_matrix.grid_size * sizeof(double);
	
	CUDA_CHECK(cudaMemset(dY_coarsened, 0, size_vec));
	CUDA_CHECK(cudaMemcpy(dX_coarsened, x, size_vec, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	
	dim3 block_size(256);
	dim3 grid_size((ellpack_matrix.nb_rows + 4 * block_size.x - 1) / (4 * block_size.x));
	
	stencil5_coarsened_ellpack_kernel<<<grid_size, block_size>>>(d_values_coarsened, d_indices_coarsened, dX_coarsened, dY_coarsened, 
	                                                             ellpack_matrix.nb_rows, ellpack_matrix.ell_width, 
	                                                             alpha, beta);

	CUDA_CHECK(cudaDeviceSynchronize());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float computeTime;
	cudaEventElapsedTime(&computeTime, start, stop);
	*kernel_time_ms = (double)computeTime;
	
	printf("[Stencil5-Coarsened] Kernel time: %.3f ms\n", computeTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CUDA_CHECK(cudaMemcpy(y, dY_coarsened, size_vec, cudaMemcpyDeviceToHost));

	double check_sum = 0.0;
	for (int i = 0; i < ellpack_matrix.grid_size * ellpack_matrix.grid_size; i++) {
		check_sum += y[i];
	}
	printf("check_sum %e\n", check_sum);

	return 0;
}

void stencil5_coarsened_free() {
	printf("[STENCIL5-COARSENED] Cleaning up\n");
	CUDA_CHECK(cudaFree(d_values_coarsened));
	CUDA_CHECK(cudaFree(d_indices_coarsened));
	CUDA_CHECK(cudaFree(dX_coarsened));
	CUDA_CHECK(cudaFree(dY_coarsened));
}

int ellpack_naive_init(MatrixData* mat) {
	// Ensure ELLPACK structure is built (shared across all kernels)
	ensure_ellpack_structure_built(mat);

	size_t size_values = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
	size_t size_indices = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int);
	size_t size_vec = ellpack_matrix.nb_rows * sizeof(double);

	CUDA_CHECK(cudaMalloc((void**)&d_values_naive, size_values));
	CUDA_CHECK(cudaMalloc((void**)&d_indices_naive, size_indices));
	CUDA_CHECK(cudaMalloc((void**)&dX_naive, size_vec));
	CUDA_CHECK(cudaMalloc((void**)&dY_naive, size_vec));

	CUDA_CHECK(cudaMemcpy(d_values_naive, ellpack_matrix.values, size_values, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_indices_naive, ellpack_matrix.indices, size_indices, cudaMemcpyHostToDevice));
	
	CUDA_CHECK(cudaDeviceSynchronize());
	
	return 0;
}

int ellpack_naive_run_timed(const double* x, double* y, double* kernel_time_ms) {
	size_t size_vec = ellpack_matrix.grid_size * ellpack_matrix.grid_size * sizeof(double);
	
	CUDA_CHECK(cudaMemset(dY_naive, 0, size_vec));
	CUDA_CHECK(cudaMemcpy(dX_naive, x, size_vec, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	
	dim3 block_size(256);
	dim3 grid_size((ellpack_matrix.nb_rows + block_size.x - 1) / block_size.x);
	
	ellpack_naive_kernel<<<grid_size, block_size>>>(d_values_naive, d_indices_naive, dX_naive, dY_naive, 
	                                                 ellpack_matrix.nb_rows, ellpack_matrix.ell_width, 
	                                                 alpha, beta);

	CUDA_CHECK(cudaDeviceSynchronize());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float computeTime;
	cudaEventElapsedTime(&computeTime, start, stop);
	*kernel_time_ms = (double)computeTime;
	
	printf("[ELLPACK-Naive] Kernel time: %.3f ms\n", computeTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CUDA_CHECK(cudaMemcpy(y, dY_naive, size_vec, cudaMemcpyDeviceToHost));

	double check_sum = 0.0;
	for (int i = 0; i < ellpack_matrix.grid_size * ellpack_matrix.grid_size; i++) {
		check_sum += y[i];
	}
	printf("check_sum %e\n", check_sum);

	return 0;
}

void ellpack_naive_free() {
	printf("[ELLPACK-NAIVE] Cleaning up\n");
	CUDA_CHECK(cudaFree(d_values_naive));
	CUDA_CHECK(cudaFree(d_indices_naive));
	CUDA_CHECK(cudaFree(dX_naive));
	CUDA_CHECK(cudaFree(dY_naive));
}

SpmvOperator SPMV_STENCIL5_COARSENED = {
	.name = "stencil5-coarsened",
	.init = stencil5_coarsened_init,
	.run_timed = stencil5_coarsened_run_timed,
	.run_device = NULL,

	.free = stencil5_coarsened_free
};

SpmvOperator SPMV_ELLPACK_NAIVE = {
	.name = "ellpack-naive",
	.init = ellpack_naive_init,
	.run_timed = ellpack_naive_run_timed,
	.run_device = NULL,

	.free = ellpack_naive_free
};

// Variables for stencil without column indices (standard ELLPACK layout)
static double *dX_no_colindices = nullptr;
static double *dY_no_colindices = nullptr;

// Variables for stencil without column indices (optimized layout)  
static double *d_values_optimized = nullptr;
static double *dX_no_colindices_opt = nullptr;
static double *dY_no_colindices_opt = nullptr;

/**
 * @brief CUDA kernel for stencil without column indices using standard ELLPACK layout.
 * Uses existing ELLPACK values order but eliminates column indices array.
 */
__global__ void stencil5_no_colindices_standard_kernel(
    const double* ellpack_values, const double* x, double* y, 
    int num_rows, int N, int max_nonzero_per_row) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int i = row / N;  // row in 2D grid
        int j = row % N;  // column in 2D grid
        
        // Check if interior point (same logic as existing kernel)
        if (i > 0 && i < N-1 && j > 0 && j < N-1) {
            double sum = 0.0;
            int offset = row * max_nonzero_per_row;
            
            // Use existing ELLPACK order (from your optimized kernel)
            sum += ellpack_values[offset + 1] * x[row - 1];      // West neighbor
            sum += ellpack_values[offset + 2] * x[row];          // Center point  
            sum += ellpack_values[offset + 3] * x[row + 1];      // East neighbor
            sum += ellpack_values[offset + 0] * x[row - N];      // North neighbor
            sum += ellpack_values[offset + 4] * x[row + N];      // South neighbor
            
            y[row] = sum;
        } else {
            // Boundary points - use general loop but without col_indices
            double sum = 0.0;
            int offset = row * max_nonzero_per_row;
            
            // Hardcode the stencil pattern for boundaries
            if (j > 0) sum += ellpack_values[offset + 1] * x[row - 1];      // West
            sum += ellpack_values[offset + 2] * x[row];                     // Center
            if (j < N - 1) sum += ellpack_values[offset + 3] * x[row + 1];  // East  
            if (i > 0) sum += ellpack_values[offset + 0] * x[row - N];      // North
            if (i < N - 1) sum += ellpack_values[offset + 4] * x[row + N];  // South
            
            y[row] = sum;
        }
    }
}

/**
 * @brief CUDA kernel for stencil without column indices using optimized values layout.
 * Values are reordered for sequential access aligned with x vector accesses.
 */
__global__ void stencil5_no_colindices_optimized_kernel(
    const double* optimized_values, const double* x, double* y, 
    int num_rows, int N, int max_nonzero_per_row) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int i = row / N;  // row in 2D grid
        int j = row % N;  // column in 2D grid
        
        double sum = 0.0;
        int offset = row * max_nonzero_per_row;
        
        // Optimized order: values aligned with sequential x accesses
        if (j > 0) sum += optimized_values[offset + 0] * x[row - 1];      // West
        sum += optimized_values[offset + 1] * x[row];                     // Center
        if (j < N - 1) sum += optimized_values[offset + 2] * x[row + 1];  // East
        if (i > 0) sum += optimized_values[offset + 3] * x[row - N];      // North  
        if (i < N - 1) sum += optimized_values[offset + 4] * x[row + N];  // South
        
        y[row] = sum;
    }
}

/**
 * @brief Creates optimized values layout from standard ELLPACK format.
 * Reorders: [North, West, Center, East, South] → [West, Center, East, North, South]
 */
static int create_optimized_values_layout() {
    printf("   ➤ Creating optimized values layout for SpMV...\n");
    
    size_t values_size = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
    double *h_values_optimized = (double*)malloc(values_size);
    if (!h_values_optimized) {
        fprintf(stderr, "[ERROR] Failed to allocate optimized values array\n");
        return EXIT_FAILURE;
    }
    
    // Reorder values: standard [0,1,2,3,4] = [North,West,Center,East,South]
    //                optimized [0,1,2,3,4] = [West,Center,East,North,South]
    for (int row = 0; row < ellpack_matrix.nb_rows; row++) {
        int src_offset = row * ellpack_matrix.ell_width;
        int dst_offset = row * ellpack_matrix.ell_width;
        
        h_values_optimized[dst_offset + 0] = ellpack_matrix.values[src_offset + 1]; // West
        h_values_optimized[dst_offset + 1] = ellpack_matrix.values[src_offset + 2]; // Center
        h_values_optimized[dst_offset + 2] = ellpack_matrix.values[src_offset + 3]; // East
        h_values_optimized[dst_offset + 3] = ellpack_matrix.values[src_offset + 0]; // North
        h_values_optimized[dst_offset + 4] = ellpack_matrix.values[src_offset + 4]; // South
    }
    
    // Copy optimized layout to GPU
    CUDA_CHECK(cudaMalloc(&d_values_optimized, values_size));
    CUDA_CHECK(cudaMemcpy(d_values_optimized, h_values_optimized, values_size, cudaMemcpyHostToDevice));
    
    free(h_values_optimized);
    printf("   ➤ Optimized layout ready (reordered for sequential x access)\n");
    
    return EXIT_SUCCESS;
}

// Standard ELLPACK layout (no column indices)
int stencil5_no_colindices_init(MatrixData* mat) {
    if (!mat) {
        fprintf(stderr, "[ERROR] Invalid matrix data\n");
        return EXIT_FAILURE;
    }
    
    printf("🔧 Initializing stencil5_no_colindices (standard ELLPACK layout)...\n");
    printf("   ➤ Matrix: %dx%d\n", mat->rows, mat->cols);
    fflush(stdout);
    
    // Ensure ELLPACK structure is built
    if (ensure_ellpack_structure_built(mat) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to build ELLPACK structure\n");
        return EXIT_FAILURE;
    }
    
    // Allocate vectors only (reuse existing ELLPACK values)
    size_t vec_size = ellpack_matrix.nb_rows * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dX_no_colindices, vec_size));
    CUDA_CHECK(cudaMalloc(&dY_no_colindices, vec_size));
    
    printf("✅ Standard layout initialized (reuses ELLPACK values, eliminates col_indices)\n");
    printf("   ➤ Memory saved: %.2f MB (no column indices array)\n",
           (ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int)) / 1024.0 / 1024.0);
    fflush(stdout);
    
    return EXIT_SUCCESS;
}

int stencil5_no_colindices_run_timed(const double* h_x, double* h_y, double* kernel_time_ms) {
    if (!dX_no_colindices || !dY_no_colindices || !h_x || !h_y || !kernel_time_ms) {
        fprintf(stderr, "[ERROR] Invalid parameters or uninitialized operator\n");
        return EXIT_FAILURE;
    }
    
    size_t vec_size = ellpack_matrix.nb_rows * sizeof(double);
    
    // Copy input and initialize output
    CUDA_CHECK(cudaMemcpy(dX_no_colindices, h_x, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dY_no_colindices, 0, vec_size));
    
    // Configure kernel
    int block_size = 256;
    int grid_size = (ellpack_matrix.nb_rows + block_size - 1) / block_size;
    
    // Warm-up run
    stencil5_no_colindices_standard_kernel<<<grid_size, block_size>>>(
        ellpack_matrix.values, dX_no_colindices, dY_no_colindices,
        ellpack_matrix.nb_rows, ellpack_matrix.grid_size, ellpack_matrix.ell_width);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed execution
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    stencil5_no_colindices_standard_kernel<<<grid_size, block_size>>>(
        ellpack_matrix.values, dX_no_colindices, dY_no_colindices,
        ellpack_matrix.nb_rows, ellpack_matrix.grid_size, ellpack_matrix.ell_width);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    *kernel_time_ms = (double)elapsed_ms;
    
    printf("[Stencil5_no_colindices] Kernel time: %.3f ms\n", elapsed_ms);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y, dY_no_colindices, vec_size, cudaMemcpyDeviceToHost));
    
    // Checksum validation
    double checksum = 0.0;
    for (int i = 0; i < ellpack_matrix.nb_rows; i++) {
        checksum += h_y[i];
    }
    printf("[Stencil5_no_colindices] checksum: %e\n", checksum);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return EXIT_SUCCESS;
}

void stencil5_no_colindices_free() {
    printf("[STENCIL5-NO-COLINDICES] Cleaning up\n");
    if (dX_no_colindices) { cudaFree(dX_no_colindices); dX_no_colindices = nullptr; }
    if (dY_no_colindices) { cudaFree(dY_no_colindices); dY_no_colindices = nullptr; }
}

// Optimized layout (reordered values)
int stencil5_no_colindices_optimized_init(MatrixData* mat) {
    if (!mat) {
        fprintf(stderr, "[ERROR] Invalid matrix data\n");
        return EXIT_FAILURE;
    }
    
    printf("🔧 Initializing stencil5_no_colindices_optimized (reordered layout)...\n");
    printf("   ➤ Matrix: %dx%d\n", mat->rows, mat->cols);
    fflush(stdout);
    
    // Ensure ELLPACK structure is built
    if (ensure_ellpack_structure_built(mat) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to build ELLPACK structure\n");
        return EXIT_FAILURE;
    }
    
    // Create optimized values layout
    if (create_optimized_values_layout() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    
    // Allocate vectors
    size_t vec_size = ellpack_matrix.nb_rows * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dX_no_colindices_opt, vec_size));
    CUDA_CHECK(cudaMalloc(&dY_no_colindices_opt, vec_size));
    
    printf("✅ Optimized layout initialized (values reordered for SpMV performance)\n");
    fflush(stdout);
    
    return EXIT_SUCCESS;
}

int stencil5_no_colindices_optimized_run_timed(const double* h_x, double* h_y, double* kernel_time_ms) {
    if (!d_values_optimized || !dX_no_colindices_opt || !dY_no_colindices_opt || !h_x || !h_y || !kernel_time_ms) {
        fprintf(stderr, "[ERROR] Invalid parameters or uninitialized operator\n");
        return EXIT_FAILURE;
    }
    
    size_t vec_size = ellpack_matrix.nb_rows * sizeof(double);
    
    // Copy input and initialize output
    CUDA_CHECK(cudaMemcpy(dX_no_colindices_opt, h_x, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dY_no_colindices_opt, 0, vec_size));
    
    // Configure kernel
    int block_size = 256;
    int grid_size = (ellpack_matrix.nb_rows + block_size - 1) / block_size;
    
    // Warm-up run
    stencil5_no_colindices_optimized_kernel<<<grid_size, block_size>>>(
        d_values_optimized, dX_no_colindices_opt, dY_no_colindices_opt,
        ellpack_matrix.nb_rows, ellpack_matrix.grid_size, ellpack_matrix.ell_width);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed execution
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    stencil5_no_colindices_optimized_kernel<<<grid_size, block_size>>>(
        d_values_optimized, dX_no_colindices_opt, dY_no_colindices_opt,
        ellpack_matrix.nb_rows, ellpack_matrix.grid_size, ellpack_matrix.ell_width);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    *kernel_time_ms = (double)elapsed_ms;
    
    printf("[Stencil5_no_colindices_opt] Kernel time: %.3f ms\n", elapsed_ms);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y, dY_no_colindices_opt, vec_size, cudaMemcpyDeviceToHost));
    
    // Checksum validation
    double checksum = 0.0;
    for (int i = 0; i < ellpack_matrix.nb_rows; i++) {
        checksum += h_y[i];
    }
    printf("[Stencil5_no_colindices_opt] checksum: %e\n", checksum);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return EXIT_SUCCESS;
}

void stencil5_no_colindices_optimized_free() {
    printf("[STENCIL5-NO-COLINDICES-OPT] Cleaning up\n");
    if (d_values_optimized) { cudaFree(d_values_optimized); d_values_optimized = nullptr; }
    if (dX_no_colindices_opt) { cudaFree(dX_no_colindices_opt); dX_no_colindices_opt = nullptr; }
    if (dY_no_colindices_opt) { cudaFree(dY_no_colindices_opt); dY_no_colindices_opt = nullptr; }
}

// Operator definitions
SpmvOperator SPMV_STENCIL5_NO_COLINDICES = {
	.name = "stencil5-no-colindices",
	.init = stencil5_no_colindices_init,
	.run_timed = stencil5_no_colindices_run_timed,
	.run_device = NULL,

	.free = stencil5_no_colindices_free
};

SpmvOperator SPMV_STENCIL5_NO_COLINDICES_OPTIMIZED = {
	.name = "stencil5-no-colindices-opt",
	.init = stencil5_no_colindices_optimized_init,
	.run_timed = stencil5_no_colindices_optimized_run_timed,
	.run_device = NULL,

	.free = stencil5_no_colindices_optimized_free
};


