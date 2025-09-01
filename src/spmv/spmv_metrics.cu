/**
 * @file spmv_metrics.cu
 * @brief Performance metrics calculation and display for SpMV operations.
 *
 * @details
 * Performance analysis capabilities for SpMV operations:
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
 * @brief Calculates performance metrics for SpMV operations.
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
 * @brief Prints performance metrics in human-readable format.
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
 * - HPC benchmarking reports
 * - Performance analysis
 * - Comparative studies between SpMV implementations
 *
 * @param metrics Complete benchmark metrics structure to display
 */
void print_benchmark_metrics(const BenchmarkMetrics* metrics, FILE* output_file) {
    FILE* fp = (output_file != NULL) ? output_file : stdout;
    fprintf(fp, "\n=== SpMV Performance Metrics ===\n");
    fprintf(fp, "Operator: %s\n", metrics->operator_name);
    fprintf(fp, "\n--- Matrix Characteristics ---\n");
    fprintf(fp, "Dimensions: %d x %d\n", metrics->matrix_rows, metrics->matrix_cols);
    fprintf(fp, "Non-zeros: %d\n", metrics->matrix_nnz);
    fprintf(fp, "Sparsity ratio: %.6f (%.4f%% non-zero)\n", 
           metrics->sparsity_ratio, metrics->sparsity_ratio * 100.0);
    
    fprintf(fp, "\n--- Performance Metrics ---\n");
    fprintf(fp, "Execution time: %.3f ms (%.1f μs)\n", 
           metrics->execution_time_ms, metrics->execution_time_ms * 1000.0);
    fprintf(fp, "GFLOPS: %.3f\n", metrics->gflops);
    fprintf(fp, "Memory bandwidth: %.3f GB/s\n", metrics->bandwidth_gb_s);
    
    // Performance analysis and context
    fprintf(fp, "\n--- Performance Analysis ---\n");
    
    // Arithmetic intensity calculation (FLOPS per byte)
    double total_flops = 2.0 * metrics->matrix_nnz;
    double matrix_data_bytes = metrics->matrix_nnz * sizeof(double);
    double matrix_indices_bytes = metrics->matrix_nnz * sizeof(int) * 2;
    double vector_bytes = (metrics->matrix_rows + metrics->matrix_cols) * sizeof(double);
    double total_bytes = matrix_data_bytes + matrix_indices_bytes + vector_bytes;
    double arithmetic_intensity = total_flops / total_bytes;
    
    fprintf(fp, "Arithmetic intensity: %.3f FLOP/byte\n", arithmetic_intensity);
    
    // Performance classification
    if (arithmetic_intensity < 0.5) {
        fprintf(fp, "Classification: Memory-bound (low arithmetic intensity)\n");
        fprintf(fp, "Optimization focus: Memory access patterns, data locality\n");
    } else if (arithmetic_intensity < 2.0) {
        fprintf(fp, "Classification: Balanced compute/memory\n"); 
        fprintf(fp, "Optimization focus: Both compute and memory optimization\n");
    } else {
        fprintf(fp, "Classification: Compute-bound (high arithmetic intensity)\n");
        fprintf(fp, "Optimization focus: Compute throughput, parallelization\n");
    }
    
    fprintf(fp, "=============================\n\n");
}

/**
 * @brief Exports performance metrics in JSON format for automated processing.
 *
 * @details
 * Outputs structured JSON containing:
 * - Matrix characteristics (dimensions, sparsity, operator)
 * - Performance metrics (execution time, GFLOPS, bandwidth)
 * - Derived analysis (arithmetic intensity, performance classification)
 * - Metadata (timestamp, version info for reproducibility)
 *
 * JSON format designed for:
 * - Automated benchmarking pipelines
 * - Data visualization tools and scripts
 * - Comparative analysis and trend tracking
 * - Integration with monitoring systems
 *
 * @param metrics Complete benchmark metrics structure to export
 */
void print_metrics_json(const BenchmarkMetrics* metrics, FILE* output_file) {
    FILE* fp = (output_file != NULL) ? output_file : stdout;
    // Calculate additional derived metrics for JSON export
    double total_flops = 2.0 * metrics->matrix_nnz;
    double matrix_data_bytes, matrix_indices_bytes, vector_bytes;
    
    // Recalculate format-specific memory traffic for JSON details
    extern CSRMatrix csr_mat;
    extern ELLPACKMatrix ellpack_matrix;
    
    if (strcmp(metrics->operator_name, "csr") == 0) {
        matrix_data_bytes = csr_mat.nb_nonzeros * sizeof(double);
        matrix_indices_bytes = csr_mat.nb_nonzeros * sizeof(int) + (csr_mat.nb_rows + 1) * sizeof(int);
    } else if (strcmp(metrics->operator_name, "stencil5") == 0) {
        matrix_data_bytes = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(double);
        matrix_indices_bytes = ellpack_matrix.nb_rows * ellpack_matrix.ell_width * sizeof(int);
    } else {
        // Fallback calculation
        matrix_data_bytes = metrics->matrix_nnz * sizeof(double);
        matrix_indices_bytes = metrics->matrix_nnz * sizeof(int) * 2;
    }
    
    vector_bytes = (metrics->matrix_rows + metrics->matrix_cols) * sizeof(double);
    double total_bytes = matrix_data_bytes + matrix_indices_bytes + vector_bytes;
    double arithmetic_intensity = total_flops / total_bytes;
    
    // Output structured JSON
    fprintf(fp, "{\n");
    fprintf(fp, "  \"gpu\": {\n");
    fprintf(fp, "    \"name\": \"%s\",\n", metrics->gpu_info.name);
    fprintf(fp, "    \"memory_mb\": %d,\n", metrics->gpu_info.memory_mb);
    fprintf(fp, "    \"compute_capability\": \"%s\",\n", metrics->gpu_info.compute_capability);
    fprintf(fp, "    \"multiprocessor_count\": %d,\n", metrics->gpu_info.multiprocessor_count);
    fprintf(fp, "    \"memory_clock_khz\": %d,\n", metrics->gpu_info.memory_clock_khz);
    fprintf(fp, "    \"graphics_clock_mhz\": %d\n", metrics->gpu_info.graphics_clock_mhz);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"benchmark\": {\n");
    fprintf(fp, "    \"operator\": \"%s\",\n", metrics->operator_name);
    fprintf(fp, "    \"matrix\": {\n");
    fprintf(fp, "      \"rows\": %d,\n", metrics->matrix_rows);
    fprintf(fp, "      \"cols\": %d,\n", metrics->matrix_cols);
    fprintf(fp, "      \"nnz\": %d,\n", metrics->matrix_nnz);
    fprintf(fp, "      \"sparsity_ratio\": %.6f,\n", metrics->sparsity_ratio);
    fprintf(fp, "      \"sparsity_percent\": %.4f\n", metrics->sparsity_ratio * 100.0);
    fprintf(fp, "    },\n");
    fprintf(fp, "    \"performance\": {\n");
    fprintf(fp, "      \"execution_time_ms\": %.6f,\n", metrics->execution_time_ms);
    fprintf(fp, "      \"execution_time_us\": %.1f,\n", metrics->execution_time_ms * 1000.0);
    fprintf(fp, "      \"gflops\": %.6f,\n", metrics->gflops);
    fprintf(fp, "      \"bandwidth_gb_s\": %.6f\n", metrics->bandwidth_gb_s);
    fprintf(fp, "    },\n");
    fprintf(fp, "    \"analysis\": {\n");
    fprintf(fp, "      \"arithmetic_intensity\": %.6f,\n", arithmetic_intensity);
    fprintf(fp, "      \"total_flops\": %.0f,\n", total_flops);
    fprintf(fp, "      \"total_bytes\": %.0f,\n", total_bytes);
    fprintf(fp, "      \"matrix_data_bytes\": %.0f,\n", matrix_data_bytes);
    fprintf(fp, "      \"matrix_indices_bytes\": %.0f,\n", matrix_indices_bytes);
    fprintf(fp, "      \"vector_bytes\": %.0f,\n", vector_bytes);
    fprintf(fp, "      \"performance_bound\": \"%s\"\n", 
           (arithmetic_intensity < 0.5) ? "memory-bound" : 
           (arithmetic_intensity < 2.0) ? "balanced" : "compute-bound");
    fprintf(fp, "    }\n");
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");
}

/**
 * @brief Exports performance metrics in CSV format for spreadsheet analysis.
 *
 * @details
 * Outputs CSV with metrics suitable for:
 * - Spreadsheet analysis and visualization
 * - Statistical analysis and trend tracking
 * - Batch processing and automation scripts
 * - Performance comparison studies
 *
 * CSV includes all key metrics in a single row for easy aggregation
 * across multiple benchmarks and comparative analysis.
 *
 * @param metrics Complete benchmark metrics structure to export
 */
void print_metrics_csv(const BenchmarkMetrics* metrics, FILE* output_file) {
    FILE* fp = (output_file != NULL) ? output_file : stdout;
    // Calculate derived metrics for CSV
    double total_flops = 2.0 * metrics->matrix_nnz;
    double arithmetic_intensity = total_flops / ((metrics->matrix_nnz * 12.0) + 
                                                (metrics->matrix_rows + metrics->matrix_cols) * 8.0);
    
    // CSV Header (print only once - could be controlled by a flag)
    static int header_printed = 0;
    if (!header_printed) {
        fprintf(fp, "operator,matrix_rows,matrix_cols,matrix_nnz,sparsity_ratio,sparsity_percent,");
        fprintf(fp, "execution_time_ms,execution_time_us,gflops,bandwidth_gb_s,");
        fprintf(fp, "arithmetic_intensity,total_flops,performance_bound\n");
        header_printed = 1;
    }
    
    // CSV Data
    fprintf(fp, "%s,%d,%d,%d,%.6f,%.4f,",
           metrics->operator_name,
           metrics->matrix_rows,
           metrics->matrix_cols, 
           metrics->matrix_nnz,
           metrics->sparsity_ratio,
           metrics->sparsity_ratio * 100.0);
    
    fprintf(fp, "%.6f,%.1f,%.6f,%.6f,",
           metrics->execution_time_ms,
           metrics->execution_time_ms * 1000.0,
           metrics->gflops,
           metrics->bandwidth_gb_s);
    
    fprintf(fp, "%.6f,%.0f,%s\n",
           arithmetic_intensity,
           total_flops,
           (arithmetic_intensity < 0.5) ? "memory-bound" : 
           (arithmetic_intensity < 2.0) ? "balanced" : "compute-bound");
}