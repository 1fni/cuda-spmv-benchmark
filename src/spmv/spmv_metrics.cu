/**
 * @file spmv_metrics.cu
 * @brief Performance metrics calculation and display for SpMV operations.
 *
 * @details
 * This module provides comprehensive performance analysis capabilities for SpMV operations:
 * - FLOPS (Floating Point Operations Per Second) calculations
 * - Memory bandwidth measurements and utilization analysis
 * - Sparsity pattern analysis and characteristics
 * - Human-readable performance reporting
 *
 * The metrics calculations follow standard HPC benchmarking practices:
 * - FLOPS: Based on 2*nnz operations (one multiply + one add per non-zero)
 * - Bandwidth: Total memory traffic divided by execution time
 * - Memory traffic includes matrix data, input vector, and output vector transfers
 *
 * Author: Bouhrour Stephane
 * Date: 2025-08-24
 */

#include <stdio.h>
#include <math.h>
#include "spmv.h"
#include "io.h"

/**
 * @brief Calculates comprehensive performance metrics for SpMV operations.
 *
 * @details
 * Computes key performance indicators:
 * - GFLOPS: Based on 2*nnz floating point operations (fused multiply-add per non-zero)
 * - Memory bandwidth: Accounts for matrix data, input vector, and output vector
 * - Sparsity ratio: Density of non-zero elements in the matrix
 * - Matrix characteristics: Dimensions and structural properties
 *
 * Memory traffic calculation:
 * - Matrix data: nnz * (sizeof(double) + sizeof(int)) for values and indices
 * - Input vector: cols * sizeof(double) 
 * - Output vector: rows * sizeof(double)
 *
 * @param execution_time_ms Measured GPU execution time in milliseconds
 * @param mat Matrix data structure with dimensions and sparsity information
 * @param operator_name Name of the SpMV implementation used
 * @param metrics Output structure for calculated performance metrics
 */
void calculate_spmv_metrics(double execution_time_ms, const MatrixData* mat, 
                           const char* operator_name, BenchmarkMetrics* metrics) {
    
    // Store basic matrix characteristics
    metrics->matrix_rows = mat->rows;
    metrics->matrix_cols = mat->cols;
    metrics->matrix_nnz = mat->nnz;
    metrics->execution_time_ms = execution_time_ms;
    metrics->operator_name = operator_name;
    
    // Calculate sparsity ratio (density of non-zero elements)
    double total_elements = (double)mat->rows * mat->cols;
    metrics->sparsity_ratio = (double)mat->nnz / total_elements;
    
    // Calculate GFLOPS based on SpMV operations
    // SpMV performs 2*nnz operations: one multiply + one add per non-zero element
    double total_flops = 2.0 * mat->nnz;
    double execution_time_s = execution_time_ms / 1000.0;
    metrics->gflops = (total_flops / execution_time_s) / 1e9;
    
    // Calculate format-specific memory bandwidth utilization using real structures
    // Memory traffic varies significantly between SpMV formats - use actual data
    extern CSRMatrix csr_mat;            // Global CSR structure
    extern ELLPACKMatrix ellpack_matrix; // Global ELLPACK structure
    
    double matrix_data_bytes = 0.0;
    double matrix_indices_bytes = 0.0;
    double input_vector_bytes = mat->cols * sizeof(double);        // Input vector
    double output_vector_bytes = mat->rows * sizeof(double);       // Output vector
    
    // Format-specific memory traffic calculation using real structures
    if (strcmp(operator_name, "csr") == 0) {
        // CSR Format - use actual CSR structure dimensions:
        // - Values: nnz * sizeof(double)
        // - Column indices: nnz * sizeof(int)  
        // - Row pointers: (rows + 1) * sizeof(int)
        matrix_data_bytes = csr_mat.nb_nonzeros * sizeof(double);
        matrix_indices_bytes = csr_mat.nb_nonzeros * sizeof(int) + 
                              (csr_mat.nb_rows + 1) * sizeof(int);
        
    } else if (strcmp(operator_name, "ellpack") == 0) {
        // ELLPACK Format - use actual ELLPACK structure dimensions:
        // - Values: rows * actual_ell_width * sizeof(double)
        // - Column indices: rows * actual_ell_width * sizeof(int)
        matrix_data_bytes = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
        matrix_indices_bytes = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int);
        
    } else if (strcmp(operator_name, "stencil5") == 0) {
        // STENCIL5 Format - uses ELLPACK storage with actual ell_width:
        // - Uses real ELLPACK structure built from CSR conversion
        // - Accounts for actual padding and boundary conditions
        matrix_data_bytes = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
        matrix_indices_bytes = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int);
        
    } else {
        // Fallback to generic calculation for unknown formats
        matrix_data_bytes = mat->nnz * sizeof(double);
        matrix_indices_bytes = mat->nnz * sizeof(int) * 2;
    }
    
    double total_bytes = matrix_data_bytes + matrix_indices_bytes + 
                        input_vector_bytes + output_vector_bytes;
    
    // Bandwidth in GB/s
    metrics->bandwidth_gb_s = (total_bytes / execution_time_s) / 1e9;
}

/**
 * @brief Prints comprehensive performance metrics in professional format.
 *
 * @details
 * Displays detailed performance analysis including:
 * - Matrix characteristics and sparsity information
 * - Execution timing with microsecond precision
 * - GFLOPS performance with efficiency context
 * - Memory bandwidth utilization with theoretical comparisons
 * - Performance classification (compute-bound vs memory-bound)
 *
 * Output format designed for:
 * - Professional HPC benchmarking reports
 * - Performance analysis and optimization
 * - Comparative studies between SpMV implementations
 *
 * @param metrics Complete benchmark metrics structure to display
 */
void print_benchmark_metrics(const BenchmarkMetrics* metrics) {
    printf("\n=== SpMV Performance Metrics ===\n");
    printf("Operator: %s\n", metrics->operator_name);
    printf("\n--- Matrix Characteristics ---\n");
    printf("Dimensions: %d x %d\n", metrics->matrix_rows, metrics->matrix_cols);
    printf("Non-zeros: %d\n", metrics->matrix_nnz);
    printf("Sparsity ratio: %.6f (%.4f%% non-zero)\n", 
           metrics->sparsity_ratio, metrics->sparsity_ratio * 100.0);
    
    printf("\n--- Performance Metrics ---\n");
    printf("Execution time: %.3f ms (%.1f μs)\n", 
           metrics->execution_time_ms, metrics->execution_time_ms * 1000.0);
    printf("GFLOPS: %.3f\n", metrics->gflops);
    printf("Memory bandwidth: %.3f GB/s\n", metrics->bandwidth_gb_s);
    
    // Performance analysis and context
    printf("\n--- Performance Analysis ---\n");
    
    // Arithmetic intensity calculation (FLOPS per byte)
    double total_flops = 2.0 * metrics->matrix_nnz;
    double matrix_data_bytes = metrics->matrix_nnz * sizeof(double);
    double matrix_indices_bytes = metrics->matrix_nnz * sizeof(int) * 2;
    double vector_bytes = (metrics->matrix_rows + metrics->matrix_cols) * sizeof(double);
    double total_bytes = matrix_data_bytes + matrix_indices_bytes + vector_bytes;
    double arithmetic_intensity = total_flops / total_bytes;
    
    printf("Arithmetic intensity: %.3f FLOP/byte\n", arithmetic_intensity);
    
    // Performance classification
    if (arithmetic_intensity < 0.5) {
        printf("Classification: Memory-bound (low arithmetic intensity)\n");
        printf("Optimization focus: Memory access patterns, data locality\n");
    } else if (arithmetic_intensity < 2.0) {
        printf("Classification: Balanced compute/memory\n"); 
        printf("Optimization focus: Both compute and memory optimization\n");
    } else {
        printf("Classification: Compute-bound (high arithmetic intensity)\n");
        printf("Optimization focus: Compute throughput, parallelization\n");
    }
    
    printf("=============================\n\n");
}