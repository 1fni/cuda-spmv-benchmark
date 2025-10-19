/**
 * @file spmv_csr_mgpu.cu
 * @brief Multi-GPU implementation of CSR SpMV with 1D row-band decomposition.
 *
 * @details
 * Multi-GPU reference implementation using cuSPARSE CSR on each GPU.
 * Each GPU processes a contiguous block of matrix rows.
 * Uses full input vector on each GPU for simplicity (same pattern as stencil).
 *
 * Architecture:
 *  - Row-band partitioning: GPU i processes rows [start_row[i] : end_row[i]]
 *  - Full vector replication: each GPU has complete x vector
 *  - cuSPARSE CSR: industry standard reference for comparison
 *
 * Author: Bouhrour Stephane
 * Date: 2025-09-26
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "spmv.h"
#include "io.h"

// Multi-GPU CSR context structure
typedef struct {
    // GPU configuration
    int gpu_count;
    int gpu_ids[8];                   // Physical GPU IDs to use
    
    // Global problem size
    int global_matrix_size;           // Total matrix rows/cols
    
    // 1D domain decomposition (row bands)
    int start_row[8];                 // Start row for each GPU
    int end_row[8];                   // End row (exclusive) for each GPU
    
    // CSR matrix partitioning
    double *d_local_values[8];        // Local CSR values per GPU
    int *d_local_col_indices[8];      // Local CSR column indices per GPU
    int *d_local_row_ptr[8];          // Local CSR row pointers per GPU
    
    // Vector storage
    double *d_full_x[8];              // Full input vector on each GPU
    double *d_local_y[8];             // Local output vectors per GPU
    
    // cuSPARSE resources
    cusparseHandle_t cusparse_handles[8];
    cusparseSpMatDescr_t mat_descr[8];
    cusparseDnVecDescr_t vec_x_descr[8];
    cusparseDnVecDescr_t vec_y_descr[8];
    void *d_buffers[8];                   // Pre-allocated buffers per GPU
    size_t buffer_sizes[8];               // Buffer sizes per GPU
    
    // CUDA resources
    cudaStream_t compute_streams[8];
} MultiGpuCSRBaseline;

// Global context
static MultiGpuCSRBaseline mgpu_csr_ctx;
static bool csr_context_initialized = false;

/**
 * @brief Partitions matrix rows among GPUs with load balancing.
 * @param ctx Multi-GPU context to setup
 */
static void setup_csr_1d_domain_decomposition(MultiGpuCSRBaseline* ctx) {
    int base_rows = ctx->global_matrix_size / ctx->gpu_count;
    int remainder_rows = ctx->global_matrix_size % ctx->gpu_count;
    
    printf("   ➤ Domain decomposition (1D row bands):\\n");
    
    for (int gpu = 0; gpu < ctx->gpu_count; gpu++) {
        ctx->gpu_ids[gpu] = gpu;  // Use GPUs 0,1,2,... 
        
        // Distribute remainder rows to first GPUs
        int local_rows = base_rows + ((gpu < remainder_rows) ? 1 : 0);
        
        if (gpu == 0) {
            ctx->start_row[gpu] = 0;
        } else {
            ctx->start_row[gpu] = ctx->end_row[gpu-1];
        }
        ctx->end_row[gpu] = ctx->start_row[gpu] + local_rows;
        
        printf("     GPU %d: rows [%d:%d) (%d rows)\\n",
               gpu, ctx->start_row[gpu], ctx->end_row[gpu], local_rows);
    }
}

/**
 * @brief Initializes multi-GPU CSR with 1D row-band decomposition.
 * @param mat Matrix data containing the global CSR matrix
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
static int multi_gpu_csr_init(MatrixData* mat) {
    if (!mat || csr_context_initialized) {
        fprintf(stderr, "[ERROR] Invalid matrix data or context already initialized\\n");
        return EXIT_FAILURE;
    }
    
    // Detect available GPUs
    int available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    
    if (available_gpus < 2) {
        fprintf(stderr, "[ERROR] Multi-GPU requires at least 2 GPUs, found %d\\n", available_gpus);
        return EXIT_FAILURE;
    }
    
    // Use all available GPUs (up to 8)
    mgpu_csr_ctx.gpu_count = (available_gpus > 8) ? 8 : available_gpus;
    mgpu_csr_ctx.global_matrix_size = mat->rows;
    
    printf("🔧 Initializing multi-GPU CSR (baseline: full vector per GPU)...\\n");
    printf("   ➤ GPUs: %d available, using %d\\n", available_gpus, mgpu_csr_ctx.gpu_count);
    printf("   ➤ Global: matrix %dx%d\\n", mat->rows, mat->cols);
    fflush(stdout);
    
    // Setup 1D domain decomposition
    setup_csr_1d_domain_decomposition(&mgpu_csr_ctx);
    
    // Build global CSR structure (reuse existing infrastructure)
    printf("   ➤ Building CSR structure...\\n");
    fflush(stdout);
    
    if (build_csr_struct(mat) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to build CSR structure\\n");
        return EXIT_FAILURE;
    }
    
    printf("   ➤ CSR built: %d rows, %d nnz\\n", csr_mat.nb_rows, csr_mat.nb_nonzeros);
    
    // Allocate GPU memory and setup cuSPARSE
    for (int gpu = 0; gpu < mgpu_csr_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_csr_ctx.gpu_ids[gpu]));
        
        int local_rows = mgpu_csr_ctx.end_row[gpu] - mgpu_csr_ctx.start_row[gpu];
        
        // Calculate local CSR size
        int local_nnz = 0;
        for (int row = mgpu_csr_ctx.start_row[gpu]; row < mgpu_csr_ctx.end_row[gpu]; row++) {
            local_nnz += csr_mat.row_ptr[row + 1] - csr_mat.row_ptr[row];
        }
        
        // Allocate local CSR matrix portions
        CUDA_CHECK(cudaMalloc(&mgpu_csr_ctx.d_local_values[gpu], local_nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&mgpu_csr_ctx.d_local_col_indices[gpu], local_nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&mgpu_csr_ctx.d_local_row_ptr[gpu], (local_rows + 1) * sizeof(int)));
        
        // Allocate full input vector (baseline approach)
        CUDA_CHECK(cudaMalloc(&mgpu_csr_ctx.d_full_x[gpu], 
                             mgpu_csr_ctx.global_matrix_size * sizeof(double)));
        
        // Allocate local output vector
        CUDA_CHECK(cudaMalloc(&mgpu_csr_ctx.d_local_y[gpu], local_rows * sizeof(double)));
        
        // Create cuSPARSE handle and stream
        CHECK_CUSPARSE(cusparseCreate(&mgpu_csr_ctx.cusparse_handles[gpu]));
        CUDA_CHECK(cudaStreamCreate(&mgpu_csr_ctx.compute_streams[gpu]));
        CHECK_CUSPARSE(cusparseSetStream(mgpu_csr_ctx.cusparse_handles[gpu], mgpu_csr_ctx.compute_streams[gpu]));
        
        printf("   ➤ GPU %d: allocated %.1f MB CSR, %.1f MB vectors, %d local nnz\\n",
               gpu, 
               (local_nnz * (sizeof(double) + sizeof(int)) + (local_rows + 1) * sizeof(int)) / 1024.0 / 1024.0,
               (mgpu_csr_ctx.global_matrix_size + local_rows) * sizeof(double) / 1024.0 / 1024.0,
               local_nnz);
    }
    
    // Copy CSR data to GPUs and setup cuSPARSE descriptors
    printf("   ➤ Copying CSR data to GPUs and setting up cuSPARSE...\\n");
    fflush(stdout);
    
    for (int gpu = 0; gpu < mgpu_csr_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_csr_ctx.gpu_ids[gpu]));
        
        int local_rows = mgpu_csr_ctx.end_row[gpu] - mgpu_csr_ctx.start_row[gpu];
        int local_start_row = mgpu_csr_ctx.start_row[gpu];
        
        // Build local CSR arrays
        int local_nnz = 0;
        int *h_local_row_ptr = (int*)malloc((local_rows + 1) * sizeof(int));
        h_local_row_ptr[0] = 0;
        
        for (int i = 0; i < local_rows; i++) {
            int global_row = local_start_row + i;
            int row_nnz = csr_mat.row_ptr[global_row + 1] - csr_mat.row_ptr[global_row];
            local_nnz += row_nnz;
            h_local_row_ptr[i + 1] = local_nnz;
        }
        
        // Copy local portion of CSR matrix
        int local_values_offset = csr_mat.row_ptr[local_start_row];
        
        CUDA_CHECK(cudaMemcpy(mgpu_csr_ctx.d_local_values[gpu], 
                             &csr_mat.values[local_values_offset],
                             local_nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(mgpu_csr_ctx.d_local_col_indices[gpu],
                             &csr_mat.col_indices[local_values_offset], 
                             local_nnz * sizeof(int),
                             cudaMemcpyHostToDevice));
                             
        CUDA_CHECK(cudaMemcpy(mgpu_csr_ctx.d_local_row_ptr[gpu],
                             h_local_row_ptr, 
                             (local_rows + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));
        
        free(h_local_row_ptr);
        
        // Create cuSPARSE matrix descriptor
        CHECK_CUSPARSE(cusparseCreateCsr(&mgpu_csr_ctx.mat_descr[gpu],
                                        local_rows, mgpu_csr_ctx.global_matrix_size, local_nnz,
                                        mgpu_csr_ctx.d_local_row_ptr[gpu],
                                        mgpu_csr_ctx.d_local_col_indices[gpu],
                                        mgpu_csr_ctx.d_local_values[gpu],
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        
        // Create cuSPARSE vector descriptors
        CHECK_CUSPARSE(cusparseCreateDnVec(&mgpu_csr_ctx.vec_x_descr[gpu],
                                          mgpu_csr_ctx.global_matrix_size,
                                          mgpu_csr_ctx.d_full_x[gpu], CUDA_R_64F));
                                          
        CHECK_CUSPARSE(cusparseCreateDnVec(&mgpu_csr_ctx.vec_y_descr[gpu],
                                          local_rows,
                                          mgpu_csr_ctx.d_local_y[gpu], CUDA_R_64F));
                                          
        // Pre-allocate cuSPARSE buffer for this GPU
        const double alpha = 1.0, beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(mgpu_csr_ctx.cusparse_handles[gpu],
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha,
                                              mgpu_csr_ctx.mat_descr[gpu],
                                              mgpu_csr_ctx.vec_x_descr[gpu],
                                              &beta,
                                              mgpu_csr_ctx.vec_y_descr[gpu],
                                              CUDA_R_64F,
                                              CUSPARSE_SPMV_ALG_DEFAULT,
                                              &mgpu_csr_ctx.buffer_sizes[gpu]));
        
        if (mgpu_csr_ctx.buffer_sizes[gpu] > 0) {
            CUDA_CHECK(cudaMalloc(&mgpu_csr_ctx.d_buffers[gpu], mgpu_csr_ctx.buffer_sizes[gpu]));
        } else {
            mgpu_csr_ctx.d_buffers[gpu] = NULL;
        }
    }
    
    csr_context_initialized = true;
    printf("✅ Multi-GPU CSR initialized successfully\\n");
    fflush(stdout);
    
    return EXIT_SUCCESS;
}

/**
 * @brief Executes multi-GPU CSR SpMV with timing.
 * @param h_x Input vector (host)
 * @param h_y Output vector (host)
 * @param kernel_time_ms Output: total execution time in milliseconds
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
static int multi_gpu_csr_run_timed(const double* h_x, double* h_y, double* kernel_time_ms) {
    if (!csr_context_initialized || !h_x || !h_y || !kernel_time_ms) {
        fprintf(stderr, "[ERROR] Invalid parameters or context not initialized\\n");
        return EXIT_FAILURE;
    }
    
    // Create timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Copy full input vector to all GPUs (baseline approach)
    printf("[Multi-GPU CSR] Copying input vector to %d GPUs...\\n", mgpu_csr_ctx.gpu_count);
    for (int gpu = 0; gpu < mgpu_csr_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_csr_ctx.gpu_ids[gpu]));
        CUDA_CHECK(cudaMemcpy(mgpu_csr_ctx.d_full_x[gpu], h_x,
                             mgpu_csr_ctx.global_matrix_size * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
    
    // Start timing (after data transfer)
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch cuSPARSE SpMV on all GPUs in parallel
    const double alpha = 1.0, beta = 0.0;
    
    for (int gpu = 0; gpu < mgpu_csr_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_csr_ctx.gpu_ids[gpu]));
        
        // Execute SpMV using pre-allocated buffer: y = alpha * A * x + beta * y
        CHECK_CUSPARSE(cusparseSpMV(mgpu_csr_ctx.cusparse_handles[gpu],
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha,
                                   mgpu_csr_ctx.mat_descr[gpu],
                                   mgpu_csr_ctx.vec_x_descr[gpu],
                                   &beta,
                                   mgpu_csr_ctx.vec_y_descr[gpu],
                                   CUDA_R_64F,
                                   CUSPARSE_SPMV_ALG_DEFAULT,
                                   mgpu_csr_ctx.d_buffers[gpu]));
    }
    
    // Synchronize all compute streams
    for (int gpu = 0; gpu < mgpu_csr_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_csr_ctx.gpu_ids[gpu]));
        CUDA_CHECK(cudaStreamSynchronize(mgpu_csr_ctx.compute_streams[gpu]));
    }
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    *kernel_time_ms = (double)elapsed_ms;
    
    // Gather results from all GPUs
    printf("[Multi-GPU CSR] Gathering results from %d GPUs...\\n", mgpu_csr_ctx.gpu_count);
    for (int gpu = 0; gpu < mgpu_csr_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_csr_ctx.gpu_ids[gpu]));
        
        int local_rows = mgpu_csr_ctx.end_row[gpu] - mgpu_csr_ctx.start_row[gpu];
        
        CUDA_CHECK(cudaMemcpy(&h_y[mgpu_csr_ctx.start_row[gpu]], mgpu_csr_ctx.d_local_y[gpu],
                             local_rows * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Calculate and display checksum for validation
    double checksum = 0.0;
    for (int i = 0; i < mgpu_csr_ctx.global_matrix_size; i++) {
        checksum += h_y[i];
    }
    printf("[Multi-GPU CSR] checksum: %e\\n", checksum);
    printf("[Multi-GPU CSR] Kernel time: %.3f ms\\n", elapsed_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return EXIT_SUCCESS;
}

/**
 * @brief Cleans up multi-GPU CSR resources.
 */
static void multi_gpu_csr_free() {
    if (!csr_context_initialized) {
        return;
    }
    
    printf("🧹 Cleaning up multi-GPU CSR resources...\\n");
    fflush(stdout);
    
    for (int gpu = 0; gpu < mgpu_csr_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_csr_ctx.gpu_ids[gpu]));
        
        // Free GPU memory
        if (mgpu_csr_ctx.d_local_values[gpu]) cudaFree(mgpu_csr_ctx.d_local_values[gpu]);
        if (mgpu_csr_ctx.d_local_col_indices[gpu]) cudaFree(mgpu_csr_ctx.d_local_col_indices[gpu]);
        if (mgpu_csr_ctx.d_local_row_ptr[gpu]) cudaFree(mgpu_csr_ctx.d_local_row_ptr[gpu]);
        if (mgpu_csr_ctx.d_full_x[gpu]) cudaFree(mgpu_csr_ctx.d_full_x[gpu]);
        if (mgpu_csr_ctx.d_local_y[gpu]) cudaFree(mgpu_csr_ctx.d_local_y[gpu]);
        if (mgpu_csr_ctx.d_buffers[gpu]) cudaFree(mgpu_csr_ctx.d_buffers[gpu]);
        
        // Destroy cuSPARSE descriptors
        if (mgpu_csr_ctx.mat_descr[gpu]) cusparseDestroySpMat(mgpu_csr_ctx.mat_descr[gpu]);
        if (mgpu_csr_ctx.vec_x_descr[gpu]) cusparseDestroyDnVec(mgpu_csr_ctx.vec_x_descr[gpu]);
        if (mgpu_csr_ctx.vec_y_descr[gpu]) cusparseDestroyDnVec(mgpu_csr_ctx.vec_y_descr[gpu]);
        
        // Destroy cuSPARSE handle and streams
        if (mgpu_csr_ctx.cusparse_handles[gpu]) cusparseDestroy(mgpu_csr_ctx.cusparse_handles[gpu]);
        if (mgpu_csr_ctx.compute_streams[gpu]) cudaStreamDestroy(mgpu_csr_ctx.compute_streams[gpu]);
    }
    
    // Reset context
    memset(&mgpu_csr_ctx, 0, sizeof(MultiGpuCSRBaseline));
    csr_context_initialized = false;
    
    printf("✅ Multi-GPU CSR cleanup completed\\n");
    fflush(stdout);
}

// Multi-GPU CSR operator definition
SpmvOperator SPMV_CSR_MULTI_GPU = {
    .name = "csr-mgpu",
    .init = multi_gpu_csr_init,
    .run_timed = multi_gpu_csr_run_timed,
    .run_device = NULL,
    .free = multi_gpu_csr_free
};