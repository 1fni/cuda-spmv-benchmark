/**
 * @file spmv_stencil_mgpu.cu
 * @brief Multi-GPU implementation of stencil SpMV with 1D row-band decomposition.
 *
 * @details
 * Baseline multi-GPU implementation using existing ELLPACK kernel.
 * Each GPU processes a contiguous block of matrix rows.
 * Uses full input vector on each GPU for simplicity (same pattern as AmgX).
 *
 * Architecture:
 *  - Row-band partitioning: GPU i processes rows [start_row[i] : end_row[i]]
 *  - Full vector replication: each GPU has complete x vector
 *  - Existing kernel: ellpack_matvec_optimized_diffusion_pattern_middle_and_else
 *
 * Author: Bouhrour Stephane
 * Date: 2025-09-25
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "spmv.h"
#include "spmv_ellpack.h"
#include "spmv_stencil.h"
#include "io.h"

// Multi-GPU context structure
typedef struct {
    // GPU configuration
    int gpu_count;
    int gpu_ids[8];                   // Physical GPU IDs to use
    
    // Global problem size
    int global_N;                     // 2D grid size (sqrt of matrix dimension)  
    int global_matrix_size;           // Total matrix rows/cols
    
    // 1D domain decomposition (row bands)
    int start_row[8];                 // Start row for each GPU
    int end_row[8];                   // End row (exclusive) for each GPU
    
    // ELLPACK matrix partitioning
    int ellpack_width;                // ELLPACK width (shared across GPUs)
    double *d_local_values[8];        // Local ELLPACK values per GPU
    int *d_local_col_indices[8];      // Local ELLPACK column indices per GPU
    
    // Vector storage
    double *d_full_x[8];              // Full input vector on each GPU
    double *d_local_y[8];             // Local output vectors per GPU
    
    // CUDA resources
    cudaStream_t compute_streams[8];
} MultiGpuStencilBaseline;

// Global context
static MultiGpuStencilBaseline mgpu_ctx;
static bool context_initialized = false;


/**
 * @brief Partitions matrix rows among GPUs with load balancing.
 * @param ctx Multi-GPU context to setup
 */
static void setup_1d_domain_decomposition(MultiGpuStencilBaseline* ctx) {
    int base_rows = ctx->global_matrix_size / ctx->gpu_count;
    int remainder_rows = ctx->global_matrix_size % ctx->gpu_count;
    
    printf("   ➤ Domain decomposition (1D row bands):\n");
    
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
        
        printf("     GPU %d: rows [%d:%d) (%d rows)\n",
               gpu, ctx->start_row[gpu], ctx->end_row[gpu], local_rows);
    }
}

/**
 * @brief Initializes multi-GPU stencil with 1D row-band decomposition.
 * @param mat Matrix data containing the global stencil matrix
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
static int multi_gpu_stencil_init(MatrixData* mat) {
    if (!mat || context_initialized) {
        fprintf(stderr, "[ERROR] Invalid matrix data or context already initialized\n");
        return EXIT_FAILURE;
    }
    
    // Detect available GPUs
    int available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    
    if (available_gpus < 2) {
        fprintf(stderr, "[ERROR] Multi-GPU requires at least 2 GPUs, found %d\n", available_gpus);
        return EXIT_FAILURE;
    }
    
    // Use all available GPUs (up to 8)
    mgpu_ctx.gpu_count = (available_gpus > 8) ? 8 : available_gpus;
    mgpu_ctx.global_matrix_size = mat->rows;
    mgpu_ctx.global_N = (int)sqrt(mat->rows);
    
    if (mgpu_ctx.global_N * mgpu_ctx.global_N != mat->rows) {
        fprintf(stderr, "[ERROR] Matrix must be square N²×N² for 2D stencil, got %d×%d\n", 
                mat->rows, mat->cols);
        return EXIT_FAILURE;
    }
    
    printf("🔧 Initializing multi-GPU stencil (baseline: full vector per GPU)...\n");
    printf("   ➤ GPUs: %d available, using %d\n", available_gpus, mgpu_ctx.gpu_count);
    printf("   ➤ Global: %dx%d grid (matrix %dx%d)\n", 
           mgpu_ctx.global_N, mgpu_ctx.global_N, mat->rows, mat->cols);
    fflush(stdout);
    
    // Setup 1D domain decomposition
    setup_1d_domain_decomposition(&mgpu_ctx);
    
    // Build ELLPACK structure (reuse existing infrastructure)
    printf("   ➤ Building ELLPACK structure...\n");
    fflush(stdout);
    
    if (ensure_ellpack_structure_built(mat) != EXIT_SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to build ELLPACK structure\n");
        return EXIT_FAILURE;
    }
    
    mgpu_ctx.ellpack_width = ellpack_matrix.ell_width;
    printf("   ➤ ELLPACK width: %d\n", mgpu_ctx.ellpack_width);
    
    // Allocate GPU memory 
    for (int gpu = 0; gpu < mgpu_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_ctx.gpu_ids[gpu]));
        
        int local_rows = mgpu_ctx.end_row[gpu] - mgpu_ctx.start_row[gpu];
        size_t local_ellpack_size = local_rows * mgpu_ctx.ellpack_width;
        
        // Allocate local ELLPACK matrix portions
        CUDA_CHECK(cudaMalloc(&mgpu_ctx.d_local_values[gpu], 
                             local_ellpack_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&mgpu_ctx.d_local_col_indices[gpu], 
                             local_ellpack_size * sizeof(int)));
        
        // Allocate full input vector (baseline approach)
        CUDA_CHECK(cudaMalloc(&mgpu_ctx.d_full_x[gpu], 
                             mgpu_ctx.global_matrix_size * sizeof(double)));
        
        // Allocate local output vector
        CUDA_CHECK(cudaMalloc(&mgpu_ctx.d_local_y[gpu], local_rows * sizeof(double)));
        
        // Create compute stream
        CUDA_CHECK(cudaStreamCreate(&mgpu_ctx.compute_streams[gpu]));
        
        printf("   ➤ GPU %d: allocated %.1f MB ELLPACK, %.1f MB vectors\n",
               gpu, 
               local_ellpack_size * (sizeof(double) + sizeof(int)) / 1024.0 / 1024.0,
               (mgpu_ctx.global_matrix_size + local_rows) * sizeof(double) / 1024.0 / 1024.0);
    }
    
    // Copy ELLPACK data to GPUs
    printf("   ➤ Copying ELLPACK data to GPUs...\n");
    fflush(stdout);
    
    for (int gpu = 0; gpu < mgpu_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_ctx.gpu_ids[gpu]));
        
        int local_rows = mgpu_ctx.end_row[gpu] - mgpu_ctx.start_row[gpu];
        size_t local_ellpack_size = local_rows * mgpu_ctx.ellpack_width;
        size_t start_offset = mgpu_ctx.start_row[gpu] * mgpu_ctx.ellpack_width;
        
        // Copy local portion of ELLPACK matrix
        CUDA_CHECK(cudaMemcpy(mgpu_ctx.d_local_values[gpu], 
                             &ellpack_matrix.values[start_offset],
                             local_ellpack_size * sizeof(double),
                             cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(mgpu_ctx.d_local_col_indices[gpu],
                             &ellpack_matrix.indices[start_offset], 
                             local_ellpack_size * sizeof(int),
                             cudaMemcpyHostToDevice));
    }
    
    context_initialized = true;
    printf("✅ Multi-GPU stencil initialized successfully\n");
    fflush(stdout);
    
    return EXIT_SUCCESS;
}

/**
 * @brief Executes multi-GPU stencil SpMV with timing.
 * @param h_x Input vector (host)
 * @param h_y Output vector (host)
 * @param kernel_time_ms Output: total execution time in milliseconds
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
static int multi_gpu_stencil_run_timed(const double* h_x, double* h_y, double* kernel_time_ms) {
    if (!context_initialized || !h_x || !h_y || !kernel_time_ms) {
        fprintf(stderr, "[ERROR] Invalid parameters or context not initialized\n");
        return EXIT_FAILURE;
    }
    
    // Create timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Copy full input vector to all GPUs (baseline approach)
    printf("[Multi-GPU] Copying input vector to %d GPUs...\n", mgpu_ctx.gpu_count);
    for (int gpu = 0; gpu < mgpu_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_ctx.gpu_ids[gpu]));
        CUDA_CHECK(cudaMemcpy(mgpu_ctx.d_full_x[gpu], h_x,
                             mgpu_ctx.global_matrix_size * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
    
    // Start timing (after data transfer)
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch ELLPACK kernels on all GPUs in parallel
    for (int gpu = 0; gpu < mgpu_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_ctx.gpu_ids[gpu]));
        
        int local_rows = mgpu_ctx.end_row[gpu] - mgpu_ctx.start_row[gpu];
        
        // Configure kernel launch parameters (same as single-GPU version)
        int threads = 32;
        int blocks = (local_rows + threads - 1) / threads;
        
        // Launch existing ELLPACK kernel on local rows
        stencil5_ellpack_kernel<<<
            blocks, threads, 0, mgpu_ctx.compute_streams[gpu]>>>(
            mgpu_ctx.d_local_values[gpu],      // Local matrix values
            mgpu_ctx.d_local_col_indices[gpu], // Local matrix indices
            mgpu_ctx.d_full_x[gpu],            // Full input vector
            mgpu_ctx.d_local_y[gpu],           // Local output
            local_rows,                        // Number of local rows
            mgpu_ctx.ellpack_width,            // ELLPACK width
            1.0, 0.0,                         // alpha=1, beta=0
            mgpu_ctx.global_N                 // Global 2D grid size
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Synchronize all compute streams
    for (int gpu = 0; gpu < mgpu_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_ctx.gpu_ids[gpu]));
        CUDA_CHECK(cudaStreamSynchronize(mgpu_ctx.compute_streams[gpu]));
    }
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    *kernel_time_ms = (double)elapsed_ms;
    
    // Gather results from all GPUs
    printf("[Multi-GPU] Gathering results from %d GPUs...\n", mgpu_ctx.gpu_count);
    for (int gpu = 0; gpu < mgpu_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_ctx.gpu_ids[gpu]));
        
        int local_rows = mgpu_ctx.end_row[gpu] - mgpu_ctx.start_row[gpu];
        
        CUDA_CHECK(cudaMemcpy(&h_y[mgpu_ctx.start_row[gpu]], mgpu_ctx.d_local_y[gpu],
                             local_rows * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Calculate and display checksum for validation
    double checksum = 0.0;
    for (int i = 0; i < mgpu_ctx.global_matrix_size; i++) {
        checksum += h_y[i];
    }
    printf("[Multi-GPU Stencil] checksum: %e\n", checksum);
    printf("[Multi-GPU Stencil] Kernel time: %.3f ms\n", elapsed_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return EXIT_SUCCESS;
}

/**
 * @brief Cleans up multi-GPU stencil resources.
 */
static void multi_gpu_stencil_free() {
    if (!context_initialized) {
        return;
    }
    
    printf("🧹 Cleaning up multi-GPU stencil resources...\n");
    fflush(stdout);
    
    for (int gpu = 0; gpu < mgpu_ctx.gpu_count; gpu++) {
        CUDA_CHECK(cudaSetDevice(mgpu_ctx.gpu_ids[gpu]));
        
        // Free GPU memory
        if (mgpu_ctx.d_local_values[gpu]) cudaFree(mgpu_ctx.d_local_values[gpu]);
        if (mgpu_ctx.d_local_col_indices[gpu]) cudaFree(mgpu_ctx.d_local_col_indices[gpu]);
        if (mgpu_ctx.d_full_x[gpu]) cudaFree(mgpu_ctx.d_full_x[gpu]);
        if (mgpu_ctx.d_local_y[gpu]) cudaFree(mgpu_ctx.d_local_y[gpu]);
        
        // Destroy streams
        if (mgpu_ctx.compute_streams[gpu]) cudaStreamDestroy(mgpu_ctx.compute_streams[gpu]);
    }
    
    // Reset context
    memset(&mgpu_ctx, 0, sizeof(MultiGpuStencilBaseline));
    context_initialized = false;
    
    printf("✅ Multi-GPU stencil cleanup completed\n");
    fflush(stdout);
}

// Multi-GPU stencil operator definition
SpmvOperator SPMV_STENCIL5_MULTI_GPU = {
    .name = "stencil5-mgpu",
    .init = multi_gpu_stencil_init,
    .run_timed = multi_gpu_stencil_run_timed,
    .run_device = NULL,

    .free = multi_gpu_stencil_free
};