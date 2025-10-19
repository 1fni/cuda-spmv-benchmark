/**
 * @file cg_test.cu
 * @brief Simple test program for CG solver
 *
 * @details
 * Solves a simple Laplacian system using CG with stencil SpMV
 *
 * Author: Bouhrour Stephane
 * Date: 2025-10-14
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "io.h"
#include "spmv.h"
#include "solvers/cg_solver.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [--mode=<spmv_mode>] [--device]\n", argv[0]);
        printf("Example: %s matrix/stencil_512x512.mtx --mode=stencil5-csr-direct --device\n", argv[0]);
        printf("\nOptions:\n");
        printf("  --mode=<mode>  SpMV operator (default: stencil5-csr-direct)\n");
        printf("  --device       Use device-native CG (zero host transfers)\n");
        return 1;
    }

    const char* matrix_file = argv[1];
    const char* mode = "stencil5-csr-direct";  // Default
    bool use_device = false;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        } else if (strcmp(argv[i], "--device") == 0) {
            use_device = true;
        }
    }

    printf("========================================\n");
    printf("CG Solver Test\n");
    printf("========================================\n");
    printf("Matrix: %s\n", matrix_file);
    printf("SpMV mode: %s\n", mode);
    printf("Interface: %s\n", use_device ? "Device-native (GPU)" : "Host (MPI-compatible)");
    printf("\n");

    // Load matrix
    MatrixData mat;
    if (load_matrix_market(matrix_file, &mat) != 0) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }

    printf("Matrix loaded: %d x %d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);

    // Get SpMV operator
    SpmvOperator* spmv_op = get_operator(mode);
    if (!spmv_op) {
        fprintf(stderr, "Error: unknown SpMV mode '%s'\n", mode);
        return 1;
    }

    // Initialize SpMV operator
    spmv_op->init(&mat);

    // Create RHS: b = A * ones (so solution should be x = ones)
    double* ones = (double*)malloc(mat.cols * sizeof(double));
    double* b = (double*)malloc(mat.rows * sizeof(double));
    double* x = (double*)calloc(mat.rows, sizeof(double));  // Initial guess x0 = 0

    for (int i = 0; i < mat.cols; i++) {
        ones[i] = 1.0;
    }

    double dummy_time;
    spmv_op->run_timed(ones, b, &dummy_time);  // b = A * ones

    printf("RHS created (b = A*ones)\n");
    printf("Initial guess: x0 = 0\n");
    printf("\n");

    // CG configuration
    CGConfig config;
    config.max_iters = 1000;
    config.tolerance = 1e-6;
    config.verbose = 2;  // Per-iteration output

    // Solve
    CGStats stats;
    printf("Starting CG solver...\n");
    printf("========================================\n");

    if (use_device) {
        if (!spmv_op->run_device) {
            fprintf(stderr, "ERROR: Operator '%s' does not support device-native interface\n", mode);
            fprintf(stderr, "Supported operators: csr, stencil5-csr-direct\n");
            return 1;
        }
        cg_solve_device(spmv_op, &mat, b, x, config, &stats);
    } else {
        cg_solve(spmv_op, &mat, b, x, config, &stats);
    }

    printf("========================================\n");
    printf("\n");

    // Verify solution (should be close to ones)
    double error = 0.0;
    for (int i = 0; i < mat.rows; i++) {
        double diff = x[i] - 1.0;
        error += diff * diff;
    }
    error = sqrt(error / mat.rows);

    printf("Solution error (RMS vs exact=1): %e\n", error);
    printf("\n");

    printf("========================================\n");
    printf("Summary\n");
    printf("========================================\n");
    printf("Converged: %s\n", stats.converged ? "YES" : "NO");
    printf("Iterations: %d\n", stats.iterations);
    printf("Time-to-solution: %.3f ms\n", stats.time_total_ms);
    printf("\nBreakdown:\n");
    printf("  SpMV:       %.3f ms (%.1f%%)\n", stats.time_spmv_ms,
           100.0 * stats.time_spmv_ms / stats.time_total_ms);
    printf("  BLAS1:      %.3f ms (%.1f%%)\n", stats.time_blas1_ms,
           100.0 * stats.time_blas1_ms / stats.time_total_ms);
    printf("  Reductions: %.3f ms (%.1f%%)\n", stats.time_reductions_ms,
           100.0 * stats.time_reductions_ms / stats.time_total_ms);
    printf("========================================\n");

    // Cleanup
    spmv_op->free();
    free(ones);
    free(b);
    free(x);
    free(mat.entries);

    return 0;
}
