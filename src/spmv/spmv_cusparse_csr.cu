/**
 * @file spmv_cusparse_csr.cu
 * @brief Implements SpMV (Sparse Matrix-Vector multiplication) using cuSPARSE CSR format.
 *
 * @details
 * Responsibilities:
 *  - Allocate device memory for CSR components (row_ptr, col_indices, values)
 *  - Create cuSPARSE descriptors for CSR matrix and dense vectors
 *  - Perform SpMV using cusparseSpMV()
 *  - Measure kernel execution time
 *  - Free GPU and cuSPARSE resources
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#include <stdio.h>
#include <cusparse.h>
#include "spmv.h"
#include "io.h"

/// Host-side CSR matrix structure holding row_ptr, col_indices, values, and dimensions
CSRMatrix csr_mat; ///< CSR matrix used by GPU routines

/// cuSPARSE handle for library context
static cusparseHandle_t handle = nullptr; ///< cuSPARSE context handle

/// cuSPARSE descriptors for sparse matrix A and dense vectors X, Y
static cusparseSpMatDescr_t matA; ///< Descriptor for CSR sparse matrix
static cusparseDnVecDescr_t vecX, vecY; ///< Descriptors for dense vectors

/// Device workspace buffer for cuSPARSE operations
static void* dBuffer = nullptr;       ///< Pointer to temporary workspace buffer
static size_t bufferSize = 0;         ///< Size of workspace buffer in bytes

/// Device pointers for CSR data and input/output vectors
static int    *dA_csrOffsets = nullptr; ///< Device array of row pointers (CSR)
static int    *dA_columns    = nullptr; ///< Device array of column indices
static double *dA_values     = nullptr; ///< Device array of non-zero values
static double *dX            = nullptr; ///< Device input vector X
static double *dY            = nullptr; ///< Device output vector Y

/// Scalars for SpMV: Y = alpha * A * X + beta * Y
static double alpha = 1.0;           ///< Weight for A*X
static double beta  = 0.0;           ///< Weight for existing Y

/**
 * @brief Builds CSR arrays from host-side MatrixData entries.
 * @details
 * Converts COO-like format (mat->entries) into CSR storage:
 *   - row_ptr: (rows + 1) length, prefix-sums of non-zero counts per row
 *   - col_indices: flattened column indices
 *   - values: flattened non-zero values
 * Allocates and populates host memory, then stores into global csr_mat.
 * @param mat Pointer to MatrixData with fields:
 *            - mat->rows    : number of rows
 *            - mat->cols    : number of columns
 *            - mat->nnz     : number of non-zero entries
 *            - mat->entries : array of {row, col, value} entries (length nnz)
 * @return EXIT_SUCCESS (0) on success, EXIT_FAILURE (non-zero) on allocation error.
 */
int build_csr_struct(MatrixData* mat)
{
    // Allocate row pointer array
    int *row_ptr = (int*)calloc(mat->rows + 1, sizeof(int));
    if (!row_ptr) {
        fprintf(stderr, "[ERROR] calloc failed for row_ptr\n");
        return EXIT_FAILURE;
    }

    // Count non-zeros per row
    for (int i = 0; i < mat->nnz; ++i) {
        int r = mat->entries[i].row;
        row_ptr[r + 1]++;
    }

    // Build prefix sum for row offsets
    for (int i = 1; i <= mat->rows; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }

    // Allocate column indices and values arrays
    int *col_indices = (int*)malloc(mat->nnz * sizeof(int));
    if (!col_indices) { free(row_ptr); return EXIT_FAILURE; }
    double *values   = (double*)malloc(mat->nnz * sizeof(double));
    if (!values) { free(row_ptr); free(col_indices); return EXIT_FAILURE; }

    // Temporary counters per row
    int *local_count = (int*)calloc(mat->rows, sizeof(int));
    if (!local_count) { free(row_ptr); free(col_indices); free(values); return EXIT_FAILURE; }

    // Populate CSR arrays
    for (int i = 0; i < mat->nnz; ++i) {
        int r = mat->entries[i].row;
        int dst = row_ptr[r] + local_count[r]++;
        col_indices[dst] = mat->entries[i].col;
        values[dst]      = mat->entries[i].value;
    }

    // Store into global CSRMatrix
    csr_mat.row_ptr     = row_ptr;
    csr_mat.col_indices = col_indices;
    csr_mat.values      = values;
    csr_mat.nb_rows     = mat->rows;
    csr_mat.nb_cols     = mat->cols;
    csr_mat.nb_nonzeros = mat->nnz;

    free(local_count);
    return EXIT_SUCCESS;
}

/**
 * @brief Initializes cuSPARSE and allocates GPU memory for CSR SpMV.
 * @details
 *   - Builds host CSR arrays via build_csr_struct()
 *   - Creates cuSPARSE handle and descriptors
 *   - Allocates and copies CSR arrays and vectors to GPU
 *   - Queries and allocates workspace buffer
 * @param mat Pointer to MatrixData for initialization
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
int csr_init(MatrixData* mat)
{
    if (build_csr_struct(mat) != EXIT_SUCCESS) return EXIT_FAILURE;

    // Create cuSPARSE context
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Allocate GPU memory for CSR arrays and vectors
    CUDA_CHECK(cudaMalloc((void**)&dA_csrOffsets, (csr_mat.nb_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dA_columns,    csr_mat.nb_nonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dA_values,     csr_mat.nb_nonzeros * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dX,            csr_mat.nb_cols     * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dY,            csr_mat.nb_rows     * sizeof(double)));

    // Copy CSR data to GPU
    CUDA_CHECK(cudaMemcpy(dA_csrOffsets, csr_mat.row_ptr,
                          (csr_mat.nb_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_columns,    csr_mat.col_indices,
                          csr_mat.nb_nonzeros  * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_values,     csr_mat.values,
                          csr_mat.nb_nonzeros  * sizeof(double), cudaMemcpyHostToDevice));

    // Create sparse matrix descriptor
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
        csr_mat.nb_rows, csr_mat.nb_cols, csr_mat.nb_nonzeros,
        dA_csrOffsets, dA_columns, dA_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // Create dense vector descriptors
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, csr_mat.nb_cols, dX, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, csr_mat.nb_rows, dY, CUDA_R_64F));

    // Query workspace size and allocate
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    return EXIT_SUCCESS;
}

/**
 * @brief Executes CSR SpMV kernel and reports result.
 * @details
 *   - Copies input vector to GPU
 *   - Launches cusparseSpMV and measures execution time
 *   - Copies output vector back to host and computes checksum
 * @param x Host input vector (length nb_cols)
 * @param y Host output vector (length nb_rows)
 * @return EXIT_SUCCESS on success
 */
int csr_run(const double* x, double* y)
{
    // Copy input vector
    CUDA_CHECK(cudaMemcpy(dX, x, csr_mat.nb_cols * sizeof(double), cudaMemcpyHostToDevice));

    // Record and perform SpMV
    cudaEvent_t start, stop;
    float time_ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("[cuSPARSE CSR] Compute SpMV: %.3f ms\n", time_ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy and verify result
    CUDA_CHECK(cudaMemcpy(y, dY, csr_mat.nb_rows * sizeof(double), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (int i = 0; i < csr_mat.nb_rows; ++i) checksum += y[i];
    printf("[CSR] checksum: %le\n", checksum);

    return EXIT_SUCCESS;
}

/**
 * @brief Frees GPU memory and destroys cuSPARSE resources.
 * @details
 *   - Releases device buffers and workspace
 *   - Destroys cuSPARSE descriptors and handle
 */
void csr_free()
{
    printf("[CSR] Cleaning up\n");
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(dA_values));
    CUDA_CHECK(cudaFree(dA_columns));
    CUDA_CHECK(cudaFree(dA_csrOffsets));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dBuffer));

    // Free cuSPARSE objects
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);
    if (matA) cusparseDestroySpMat(matA);
    if (handle) cusparseDestroy(handle);
    
    // Free host CSR arrays
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
 * @brief Registers the CSR SpMV operator in the benchmark suite.
 * @details
 *   Provides function pointers for init, run, and free routines.
 */
SpmvOperator SPMV_CSR = {
    .name = "csr",
    .init = csr_init,
    .run  = csr_run,
    .free = csr_free
};
