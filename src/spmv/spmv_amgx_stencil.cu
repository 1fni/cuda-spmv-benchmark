/**
 * @file spmv_amgx_stencil.cu  
 * @brief Implements SpMV using NVIDIA AmgX library for stencil matrix comparison.
 *
 * @details
 * This operator serves as a reference implementation using NVIDIA's AmgX library
 * to provide baseline performance comparison for our custom stencil kernels.
 * Uses AmgX's internal multiply() function for pure SpMV operations, not full solving.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-09-24
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "spmv.h" 
#include "io.h"

#ifdef WITH_AMGX
// AmgX C++ headers
#include <amgx_c.h>

// AmgX resources and matrix structures
static AMGX_config_handle cfg = NULL;
static AMGX_resources_handle rsrc = NULL;
static AMGX_matrix_handle A = NULL;
static AMGX_vector_handle x = NULL;
static AMGX_vector_handle y = NULL;

// GPU memory for vectors
static double *d_x = NULL;
static double *d_y = NULL;
static int vector_size = 0;

/**
 * @brief Initializes AmgX structures and converts MatrixData to AmgX format.
 * @param mat Input matrix data in Matrix Market format
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
static int amgx_stencil_init(MatrixData* mat) {
    if (!mat || mat->nnz <= 0) {
        fprintf(stderr, "[ERROR] Invalid matrix data for AmgX init\n");
        return EXIT_FAILURE;
    }

    printf("🔧 Initializing AmgX stencil operator...\n");
    printf("   ➤ Matrix: %dx%d, nnz=%d\n", mat->rows, mat->cols, mat->nnz);
    fflush(stdout);

    // Initialize AmgX
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());

    // Create config for SpMV only (no solving)
    const char* config_string = "config_version=2, determinism_flag=1, exception_handling=1";
    
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, config_string));

    // Create resources  
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

    // Create matrix and vectors
    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&y, rsrc, AMGX_mode_dDDI));

    printf("   ➤ Converting MatrixData to CSR format...\n");
    fflush(stdout);

    // Convert MatrixData to CSR format for AmgX
    int *row_ptr = (int*)calloc(mat->rows + 1, sizeof(int));
    if (!row_ptr) {
        fprintf(stderr, "[ERROR] Failed to allocate row_ptr for AmgX\n");
        return EXIT_FAILURE;
    }

    // Count non-zeros per row
    for (int i = 0; i < mat->nnz; ++i) {
        int r = mat->entries[i].row;
        row_ptr[r + 1]++;
    }

    // Build prefix sum
    for (int i = 1; i <= mat->rows; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }

    // Allocate column indices and values
    int *col_indices = (int*)malloc(mat->nnz * sizeof(int));
    double *values = (double*)malloc(mat->nnz * sizeof(double));
    if (!col_indices || !values) {
        free(row_ptr); free(col_indices); free(values);
        fprintf(stderr, "[ERROR] Failed to allocate CSR arrays for AmgX\n");
        return EXIT_FAILURE;
    }

    // Populate CSR arrays
    int *local_count = (int*)calloc(mat->rows, sizeof(int));
    for (int i = 0; i < mat->nnz; ++i) {
        int r = mat->entries[i].row;
        int dst = row_ptr[r] + local_count[r]++;
        col_indices[dst] = mat->entries[i].col;
        values[dst] = mat->entries[i].value;
    }
    free(local_count);

    printf("   ➤ Uploading matrix to AmgX...\n");
    fflush(stdout);

    // Upload matrix to AmgX
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, mat->rows, mat->nnz, 1, 1, 
                                          row_ptr, col_indices, values, NULL));

    // Allocate and initialize vectors
    vector_size = mat->rows;
    CUDA_CHECK(cudaMalloc(&d_x, vector_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, vector_size * sizeof(double)));

    // Initialize vectors in AmgX
    AMGX_SAFE_CALL(AMGX_vector_upload(x, vector_size, 1, d_x));
    AMGX_SAFE_CALL(AMGX_vector_upload(y, vector_size, 1, d_y));

    // Cleanup temporary host arrays
    free(row_ptr);
    free(col_indices); 
    free(values);

    printf("✅ AmgX stencil operator initialized successfully\n");
    fflush(stdout);
    return EXIT_SUCCESS;
}

/**
 * @brief Executes SpMV using AmgX multiply function with kernel timing.
 * @param h_x Input vector (host)
 * @param h_y Output vector (host)  
 * @param kernel_time_ms Output: kernel execution time in milliseconds
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
static int amgx_stencil_run_timed(const double* h_x, double* h_y, double* kernel_time_ms) {
    if (!h_x || !h_y || !kernel_time_ms) {
        fprintf(stderr, "[ERROR] NULL pointers in amgx_stencil_run_timed\n");
        return EXIT_FAILURE;
    }

    // Copy input vector to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, vector_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, vector_size * sizeof(double)));

    // Update AmgX vectors
    AMGX_SAFE_CALL(AMGX_vector_upload(x, vector_size, 1, d_x));
    AMGX_SAFE_CALL(AMGX_vector_upload(y, vector_size, 1, d_y));

    // Warm-up run
    AMGX_SAFE_CALL(AMGX_matrix_vector_multiply(A, x, y));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed execution
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    AMGX_SAFE_CALL(AMGX_matrix_vector_multiply(A, x, y));
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    *kernel_time_ms = (double)elapsed_ms;

    // Download result
    AMGX_SAFE_CALL(AMGX_vector_download(y, d_y));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, vector_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Calculate and display checksum for result validation
    double checksum = 0.0;
    for (int i = 0; i < vector_size; i++) {
        checksum += h_y[i];
    }
    printf("[AmgX] checksum: %e\n", checksum);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}

/**
 * @brief Cleans up AmgX resources and GPU memory.
 */
static void amgx_stencil_free() {
    printf("🧹 Cleaning up AmgX resources...\n");
    fflush(stdout);

    if (d_x) { cudaFree(d_x); d_x = NULL; }
    if (d_y) { cudaFree(d_y); d_y = NULL; }

    if (y) { AMGX_vector_destroy(y); y = NULL; }
    if (x) { AMGX_vector_destroy(x); x = NULL; }  
    if (A) { AMGX_matrix_destroy(A); A = NULL; }
    if (rsrc) { AMGX_resources_destroy(rsrc); rsrc = NULL; }
    if (cfg) { AMGX_config_destroy(cfg); cfg = NULL; }

    AMGX_finalize_plugins();
    AMGX_finalize();

    printf("✅ AmgX cleanup completed\n");
    fflush(stdout);
}

// AmgX operator definition following SpmvOperator interface
SpmvOperator SPMV_AMGX_STENCIL = {
    .name = "amgx-stencil",
    .init = amgx_stencil_init,
    .run_timed = amgx_stencil_run_timed,
    .free = amgx_stencil_free
};

#else
// Stub implementation when AmgX not available
static int amgx_stencil_init_stub(MatrixData* mat) {
    printf("AmgX not available - compile with AmgX support to use amgx-stencil mode\n");
    return EXIT_FAILURE;
}

static int amgx_stencil_run_timed_stub(const double* h_x, double* h_y, double* kernel_time_ms) {
    return EXIT_FAILURE;
}

static void amgx_stencil_free_stub() {
    // Nothing to free
}

SpmvOperator SPMV_AMGX_STENCIL = {
    .name = "amgx-stencil",
    .init = amgx_stencil_init_stub,
    .run_timed = amgx_stencil_run_timed_stub,
    .free = amgx_stencil_free_stub
};
#endif