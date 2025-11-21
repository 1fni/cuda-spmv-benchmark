/**
 * @file cg_test.cu
 * @brief CG solver benchmark with multi-mode support
 *
 * @details
 * Solves a Laplacian system using CG with different SpMV operators.
 * Supports comma-separated modes for comparison without CSR rebuild.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-10-14
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "io.h"
#include "spmv.h"
#include "solvers/cg_solver.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [--mode=<modes>] [--host] [--tol=<tol>] [--maxiter=<n>]\n", argv[0]);
        printf("Example: %s matrix/stencil_512x512.mtx --mode=csr,stencil5-csr-direct\n", argv[0]);
        printf("\nOptions:\n");
        printf("  --mode=<modes>         SpMV operators, comma-separated (default: stencil5-csr-direct)\n");
        printf("  --host                 Use host interface (default: device-native for best GPU perf)\n");
        printf("  --tol=<tol>            Convergence tolerance (default: 1e-6)\n");
        printf("  --maxiter=<n>          Maximum iterations (default: 1000)\n");
        printf("  --no-detailed-timers   Disable per-category timing\n");
        return 1;
    }

    const char* matrix_file = argv[1];
    const char* modes_string = "stencil5-csr-direct";  // Default
    bool use_device = true;
    bool enable_detailed_timers = true;
    double tolerance = 1e-6;
    int max_iters = 1000;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            modes_string = argv[i] + 7;
        } else if (strcmp(argv[i], "--host") == 0) {
            use_device = false;
        } else if (strcmp(argv[i], "--device") == 0) {
            use_device = true;
        } else if (strcmp(argv[i], "--no-detailed-timers") == 0) {
            enable_detailed_timers = false;
        } else if (strncmp(argv[i], "--tol=", 6) == 0) {
            tolerance = atof(argv[i] + 6);
        } else if (strncmp(argv[i], "--maxiter=", 10) == 0) {
            max_iters = atoi(argv[i] + 10);
        }
    }

    // Parse modes (split by comma)
    char modes_buffer[256];
    strncpy(modes_buffer, modes_string, sizeof(modes_buffer) - 1);
    modes_buffer[sizeof(modes_buffer) - 1] = '\0';

    const char* mode_tokens[10];
    int num_modes = 0;

    char* token = strtok(modes_buffer, ",");
    while (token != NULL && num_modes < 10) {
        mode_tokens[num_modes++] = token;
        token = strtok(NULL, ",");
    }

    // Validate all modes before loading matrix
    printf("Validating %d mode(s): ", num_modes);
    for (int i = 0; i < num_modes; i++) {
        printf("%s%s", mode_tokens[i], (i < num_modes - 1) ? ", " : "\n");
        SpmvOperator* op = get_operator(mode_tokens[i]);
        if (!op) {
            fprintf(stderr, "Error: unknown SpMV mode '%s'\n", mode_tokens[i]);
            fprintf(stderr, "Available: csr, stencil5-csr-direct, stencil5-opt, ellpack, etc.\n");
            return 1;
        }
        if (use_device && !op->run_device) {
            fprintf(stderr, "Error: mode '%s' does not support device-native interface\n", mode_tokens[i]);
            return 1;
        }
    }

    // Load matrix
    printf("\nLoading matrix: %s\n", matrix_file);
    MatrixData mat;
    if (load_matrix_market(matrix_file, &mat) != 0) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }
    printf("Matrix loaded: %d x %d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);

    // Multi-mode warning
    if (num_modes > 1) {
        printf("\nNOTE: Multi-mode benchmark - performance may vary with order due to GPU state.\n");
        printf("      For accurate comparison, run each mode separately.\n");
    }

    // Create RHS and solution vectors (shared across modes)
    double* b = (double*)malloc(mat.rows * sizeof(double));
    double* x = (double*)calloc(mat.rows, sizeof(double));
    for (int i = 0; i < mat.rows; i++) {
        b[i] = 1.0;
    }

    // CG configuration
    CGConfig config;
    config.max_iters = max_iters;
    config.tolerance = tolerance;
    config.verbose = 2;
    config.enable_detailed_timers = enable_detailed_timers;

    // Loop through all modes
    for (int mode_idx = 0; mode_idx < num_modes; mode_idx++) {
        const char* current_mode = mode_tokens[mode_idx];

        printf("\n========================================\n");
        printf("CG Solver - Mode: %s\n", current_mode);
        printf("========================================\n");
        printf("Interface: %s\n", use_device ? "Device-native (GPU)" : "Host");
        printf("Tolerance: %.1e, Max iterations: %d\n", tolerance, max_iters);

        // Get and initialize operator
        SpmvOperator* spmv_op = get_operator(current_mode);
        spmv_op->init(&mat);

        // Reset solution vector
        memset(x, 0, mat.rows * sizeof(double));

        // Warmup run (1 CG iteration equivalent - just SpMV)
        printf("Warmup...\n");
        double* y_warmup = (double*)calloc(mat.rows, sizeof(double));
        double* x_warmup = (double*)malloc(mat.rows * sizeof(double));
        for (int i = 0; i < mat.rows; i++) x_warmup[i] = 1.0;
        double dummy_time;
        for (int w = 0; w < 3; w++) {
            spmv_op->run_timed(x_warmup, y_warmup, &dummy_time);
        }
        free(y_warmup);
        free(x_warmup);

        // Solve
        CGStats stats;
        printf("Starting CG solver...\n");

        if (use_device) {
            cg_solve_device(spmv_op, &mat, b, x, config, &stats);
        } else {
            cg_solve(spmv_op, &mat, b, x, config, &stats);
        }

        // Verify solution
        double error = 0.0;
        for (int i = 0; i < mat.rows; i++) {
            double diff = x[i] - 1.0;
            error += diff * diff;
        }
        error = sqrt(error / mat.rows);

        // Summary for this mode
        printf("\n--- Results for %s ---\n", current_mode);
        printf("Converged: %s in %d iterations\n", stats.converged ? "YES" : "NO", stats.iterations);
        printf("Time: %.3f ms (SpMV: %.1f%%, BLAS1: %.1f%%, Reductions: %.1f%%)\n",
               stats.time_total_ms,
               100.0 * stats.time_spmv_ms / stats.time_total_ms,
               100.0 * stats.time_blas1_ms / stats.time_total_ms,
               100.0 * stats.time_reductions_ms / stats.time_total_ms);
        printf("Solution error (RMS): %e\n", error);
        printf("GFLOPS (SpMV): %.3f\n",
               (2.0 * mat.nnz * stats.iterations) / (stats.time_spmv_ms * 1e6));

        // Cleanup operator (GPU memory only, CSR host preserved)
        spmv_op->free();
    }

    // Final cleanup
    printf("\n========================================\n");
    free(b);
    free(x);
    free(mat.entries);

    return 0;
}
