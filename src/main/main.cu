/**
 * @file main.cu
 * @brief Main entry point for benchmarking different SpMV implementations.
 *
 * @details
 * This file serves as the primary driver for the SpMV benchmark suite:
 * - Parses command-line arguments to determine benchmark mode and matrix file
 * - Loads sparse matrices from Matrix Market (.mtx) format files
 * - Selects and initializes the appropriate SpMV operator (CSR, ELLPACK, or STENCIL)
 * - Allocates and initializes input/output vectors on host memory
 * - Executes the SpMV benchmark using the selected GPU implementation
 * - Manages memory cleanup and resource deallocation
 * 
 * The program supports three SpMV implementations:
 * - CSR (Compressed Sparse Row) format
 * - ELLPACK format for regular sparsity patterns
 * - STENCIL format for structured grid operations
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "spmv.h"
#include "io.h"
#include "benchmark_stats.h"

/**
 * @brief Main function - Entry point for SpMV benchmark program.
 * @details Orchestrates the entire benchmark workflow including argument parsing,
 * matrix loading, operator selection, memory management, and benchmark execution.
 * The function expects command-line arguments specifying the matrix file and
 * the desired SpMV implementation mode.
 * 
 * Expected usage: ./program <matrix_file.mtx> --mode=<csr|ellpack|stencil>
 * 
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return EXIT_SUCCESS (0) on successful completion, EXIT_FAILURE (1) on error
 */
int main(int argc, char* argv[]) {
    // Check for correct number of command-line arguments
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx> --mode=<csr|ellpack|stencil> [--output-format=<human|json|csv>] [--output-file=<filename>]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* matrix_file = argv[1];        ///< Path to Matrix Market file containing sparse matrix
    const char* mode = NULL;                  ///< SpMV implementation mode (csr, ellpack, or stencil)
    const char* output_format = "human";      ///< Output format for metrics (default: human-readable)
    const char* output_file = NULL;           ///< Output file for metrics (default: stdout)

    // Parse command-line arguments to find mode, output format, and output file
    for (int i = 2; i < argc; ++i) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;  // Get mode value after "--mode="
        } else if (strncmp(argv[i], "--output-format=", 16) == 0) {
            output_format = argv[i] + 16;  // Get format value after "--output-format="
        } else if (strncmp(argv[i], "--output-file=", 14) == 0) {
            output_file = argv[i] + 14;  // Get filename after "--output-file="
        }
    }

    // Validate that mode was specified
    if (mode == NULL) {
        fprintf(stderr, "Error: mode not specified. Use --mode=<csr|ellpack|stencil>\n");
        return EXIT_FAILURE;
    }

    // Validate output format
    if (strcmp(output_format, "human") != 0 && strcmp(output_format, "json") != 0 && strcmp(output_format, "csv") != 0) {
        fprintf(stderr, "Error: Invalid output format '%s'. Use --output-format=<human|json|csv>\n", output_format);
        return EXIT_FAILURE;
    }

    // Select the corresponding SpMV operator (CSR, ELLPACK, STENCIL)
    SpmvOperator* op = get_operator(mode);
    if (op == NULL) {
        fprintf(stderr, "Error: Unknown mode '%s'. Available modes: csr, ellpack, stencil\n", mode);
        return EXIT_FAILURE;
    }

    // Load the matrix from Matrix Market file into a generic structure
    MatrixData mat; ///< Container for matrix data loaded from file
    if (load_matrix_market(matrix_file, &mat) != 0) {
        fprintf(stderr, "Failed to load matrix %s\n", matrix_file);
        return EXIT_FAILURE;
    }

    printf("Matrix loaded: %d rows, %d cols, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);

    // Initialize the selected SpMV operator (allocate GPU structures, transfer matrix)
    if (op->init(&mat) != 0) {
        fprintf(stderr, "Failed to initialize operator '%s'\n", op->name);
        return EXIT_FAILURE;
    }

    // Allocate and initialize input/output vectors on the host
    double* x = (double*)malloc(mat.cols * sizeof(double)); ///< Input vector for SpMV operation (x in y = A*x)
    double* y = (double*)malloc(mat.rows * sizeof(double)); ///< Output vector for SpMV operation (y in y = A*x)
    if (!x || !y) {
        fprintf(stderr, "Error allocating vectors\n");
        return EXIT_FAILURE;
    }

    // Initialize vectors with appropriate values
    for (int i = 0; i < mat.cols; i++) x[i] = 1.0; // Fill input vector with 1.0
    memset(y, 0, mat.rows * sizeof(double));       // Initialize output vector to 0

    // Statistical benchmark with outlier detection
    printf("Running statistical benchmark (10 iterations)...\n");
    BenchmarkStats bench_stats;
    if (benchmark_with_stats(op->run_timed, x, y, 10, &bench_stats) != 0) {
        fprintf(stderr, "Statistical benchmark failed for mode '%s'\n", op->name);
        return EXIT_FAILURE;
    }
    
    printf("Completed: %d valid runs, %d outliers removed\n", 
           bench_stats.valid_runs, bench_stats.outliers_removed);
    
    // Calculate performance metrics using median time
    BenchmarkMetrics metrics;
    calculate_spmv_metrics(bench_stats.median_ms, &mat, op->name, &metrics);
    
    // Add GPU specifications to metrics
    if (get_gpu_properties(&metrics) != 0) {
        fprintf(stderr, "Warning: Could not retrieve GPU properties\n");
    }
    
    // Open output file if specified
    FILE* output_fp = stdout;  // Default to stdout
    if (output_file != NULL) {
        output_fp = fopen(output_file, "w");
        if (output_fp == NULL) {
            fprintf(stderr, "Error: Could not open output file '%s' for writing\n", output_file);
            return EXIT_FAILURE;
        }
        printf("Writing metrics to file: %s\n", output_file);
    }
    
    // Output metrics in requested format
    if (strcmp(output_format, "json") == 0) {
        print_metrics_json(&metrics, output_fp);
    } else if (strcmp(output_format, "csv") == 0) {
        print_metrics_csv(&metrics, output_fp);
    } else {
        print_benchmark_metrics(&metrics, output_fp);  // Default human-readable format
    }
    
    // Close output file if it was opened
    if (output_fp != stdout) {
        fclose(output_fp);
        printf("Metrics successfully written to: %s\n", output_file);
    }
    
    printf("SpMV completed successfully using mode: %s\n", op->name);

    // Free GPU resources used by the selected operator
    if (op->free) {
        op->free();
    }

    // Free host memory for vectors and matrix data
    free(x);
    free(y);
    
    // Free matrix entries allocated in load_matrix_market
    if (mat.entries) {
        free(mat.entries);
    }

    return EXIT_SUCCESS;
}
