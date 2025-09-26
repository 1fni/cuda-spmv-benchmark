/**
 * @file spmv.h
 * @brief Central header for Sparse Matrix-Vector Multiplication (SpMV) operators and shared global structures.
 *
 * @details
 * Responsibilities:
 *  - Declare global variables for CSR and ELLPACK structures used across multiple SpMV implementations.
 *  - Provide helper macros for CUDA and cuSPARSE error handling.
 *  - Define the SpmvOperator structure, which acts as a function dispatch table for different SpMV implementations.
 *
 * Key Components:
 *  - `CSRMatrix` and `ELLPACKMatrix` global instances for storing matrix formats.
 *  - Function pointer-based `SpmvOperator` for modular algorithm selection.
 *
 * Author: Bouhrour Stephane  
 * Date: 2025-07-15
 */

#ifndef SPMV_H
#define SPMV_H

#include <stdio.h>
#include "spmv_csr.h"
#include "spmv_ellpack.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Global CSR and ELLPACK matrix structures used by operators. */
extern CSRMatrix csr_mat;

// Local ELLPACK utilities
int build_ellpack_from_csr_local(CSRMatrix *csr_matrix);
int ensure_ellpack_structure_built(MatrixData* mat);
extern ELLPACKMatrix ellpack_matrix;

#ifdef __cplusplus
}
#endif

/** @brief CUDA error checking macro. */
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

/** @brief cuSPARSE error checking macro. */
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

/** @brief AmgX error checking macro (use built-in if available) */

/**
 * @struct BenchmarkMetrics
 * @brief Performance metrics collected during SpMV benchmarking.
 *
 * Metrics for analyzing SpMV performance:
 *  - Execution time measurements
 *  - FLOPS (Floating Point Operations Per Second) calculations
 *  - Memory bandwidth utilization
 *  - Matrix characteristics for analysis context
 */
typedef struct {
    double execution_time_ms;        ///< Total execution time in milliseconds
    double gflops;                  ///< GFLOPS (Giga Floating Point Operations Per Second)
    double bandwidth_gb_s;          ///< Memory bandwidth in GB/s
    int matrix_rows;               ///< Number of matrix rows (N² for stencil)
    int matrix_cols;               ///< Number of matrix columns (N² for stencil)
    int matrix_nnz;                ///< Number of non-zero elements
    int grid_size;                 ///< Original 2D grid dimension (N for NxN stencil)
    double sparsity_ratio;         ///< Sparsity ratio (nnz / (rows * cols))
    const char* operator_name;     ///< Name of the SpMV operator used
    
    struct {
        char name[128];
        int memory_mb;
        char compute_capability[16];
        int multiprocessor_count;
        int max_threads_per_block;
        int memory_clock_khz;
        int graphics_clock_mhz;
        int cuda_runtime_version;
        int cuda_driver_version;
        int cusparse_version;
        int current_temp_c;
        int max_temp_c;
        int power_draw_w;
        int power_limit_w;
        char persistence_mode[16];
        char cpu_model[128];
        int system_ram_gb;
        char pcie_generation[16];
        int pcie_link_width;
    } gpu_info;
} BenchmarkMetrics;

/**
 * @struct SpmvOperator
 * @brief Defines the interface for different SpMV implementations.
 *
 * Each operator provides:
 *  - `name`: Identifier of the method (e.g., CSR, ELLPACK, stencil).
 *  - `init`: Function pointer for initialization with a given matrix.
 *  - `run_timed`: Function pointer to execute SpMV with kernel-level timing.
 *  - `free`: Function pointer to release associated resources.
 */
typedef struct {
    const char* name;                                      ///< Operator name (e.g., "csr", "ellpack").
    int (*init)(MatrixData* mat);                          ///< Initialization function for matrix data.
    int (*run_timed)(const double* x, double* y, double* kernel_time_ms); ///< SpMV with kernel timing.
    void (*free)();                                        ///< Resource cleanup function.
} SpmvOperator;

/** @brief External operator declarations for different formats. */
extern SpmvOperator SPMV_CSR;
extern SpmvOperator SPMV_STENCIL5;
extern SpmvOperator SPMV_STENCIL5_OPTIMIZED;
extern SpmvOperator SPMV_STENCIL5_SHARED;
extern SpmvOperator SPMV_STENCIL5_COARSENED;
extern SpmvOperator SPMV_ELLPACK_NAIVE;
extern SpmvOperator SPMV_ELLPACK;
extern SpmvOperator SPMV_STENCIL5_NO_COLINDICES;
extern SpmvOperator SPMV_STENCIL5_NO_COLINDICES_OPTIMIZED;
extern SpmvOperator SPMV_STENCIL5_MULTI_GPU;
extern SpmvOperator SPMV_AMGX_STENCIL;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Retrieves the appropriate SpMV operator by name.
 * @param mode Name of the operator (e.g., "csr", "ellpack").
 * @return Pointer to the matching SpmvOperator, or NULL if not found.
 */
SpmvOperator* get_operator(const char* mode);

/**
 * @brief Calculates performance metrics for SpMV operations with format-specific memory analysis.
 * @param execution_time_ms Measured execution time in milliseconds
 * @param mat Matrix data structure containing matrix characteristics  
 * @param operator_name Name of the SpMV operator used
 * @param metrics Output structure to store calculated metrics
 */
void calculate_spmv_metrics(double execution_time_ms, const MatrixData* mat, 
                           const char* operator_name, BenchmarkMetrics* metrics);

int get_gpu_properties(BenchmarkMetrics* metrics);

/**
 * @brief Prints detailed performance metrics in human-readable format.
 * @param metrics Benchmark metrics structure to display
 * @param output_file Output file pointer (use stdout if NULL)
 */
void print_benchmark_metrics(const BenchmarkMetrics* metrics, FILE* output_file);

/**
 * @brief Exports performance metrics in JSON format for automated analysis.
 * @param metrics Benchmark metrics structure to export
 * @param output_file Output file pointer (use stdout if NULL)
 */
void print_metrics_json(const BenchmarkMetrics* metrics, FILE* output_file);

/**
 * @brief Exports performance metrics in CSV format for spreadsheet analysis.
 * @param metrics Benchmark metrics structure to export
 * @param output_file Output file pointer (use stdout if NULL)
 */
void print_metrics_csv(const BenchmarkMetrics* metrics, FILE* output_file);

#ifdef __cplusplus
}
#endif

#endif // SPMV_H
