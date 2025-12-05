/**
 * @file test_mgpu_cg.cu
 * @brief Test program for multi-GPU CG solver with MPI+NCCL
 *
 * Launch: mpirun -np 2 ./bin/test_mgpu_cg matrix/stencil_512x512.mtx
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-06
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <mpi.h>
#include "io.h"
#include "solvers/cg_solver_mgpu.h"
#include "benchmark_stats_mgpu.h"
#include "spmv.h"

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Set GPU device early (before any CUDA operations including spmv_op->init)
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    int local_gpu = rank % num_gpus;
    cudaSetDevice(local_gpu);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <N> %s <matrix.mtx> [--mode=<modes>] [--tol=<tol>] [--maxiter=<n>]\n", argv[0]);
            printf("Example: mpirun -np 2 %s matrix/stencil_512x512.mtx --mode=csr,stencil5-csr-direct\n", argv[0]);
            printf("\nOptions:\n");
            printf("  --mode=<modes>         SpMV operators, comma-separated (default: csr)\n");
            printf("  --tol=<tol>            Convergence tolerance (default: 1e-6)\n");
            printf("  --maxiter=<n>          Maximum iterations (default: 100)\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char* matrix_file = argv[1];
    const char* modes_string = "csr";  // Default
    double tolerance = 1e-6;
    int max_iters = 1000;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            modes_string = argv[i] + 7;
        } else if (strncmp(argv[i], "--tol=", 6) == 0) {
            tolerance = atof(argv[i] + 6);
        } else if (strncmp(argv[i], "--maxiter=", 10) == 0) {
            max_iters = atoi(argv[i] + 10);
        }
    }

    // Parse modes (split by comma) BEFORE loading matrix
    char modes_buffer[256];
    strncpy(modes_buffer, modes_string, sizeof(modes_buffer) - 1);
    modes_buffer[sizeof(modes_buffer) - 1] = '\0';

    const char* mode_tokens[10];  // Support up to 10 modes
    int num_modes = 0;

    char* token = strtok(modes_buffer, ",");
    while (token != NULL && num_modes < 10) {
        mode_tokens[num_modes++] = token;
        token = strtok(NULL, ",");
    }

    // Validate all modes BEFORE loading matrix (rank 0 only)
    if (rank == 0) {
        printf("Validating %d mode(s): ", num_modes);
        for (int i = 0; i < num_modes; i++) {
            printf("%s%s", mode_tokens[i], (i < num_modes - 1) ? ", " : "\n");

            SpmvOperator* op = get_operator(mode_tokens[i]);
            if (op == NULL) {
                fprintf(stderr, "Error: Unknown mode '%s'\n", mode_tokens[i]);
                fprintf(stderr, "Available modes: csr, stencil5-csr-direct\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Rank 0 loads matrix and broadcasts dimensions
    MatrixData mat;
    int matrix_rows, matrix_cols, matrix_nnz, grid_size;

    if (rank == 0) {
        printf("\nLoading matrix: %s\n", matrix_file);
        if (load_matrix_market(matrix_file, &mat) != 0) {
            fprintf(stderr, "Error loading matrix: %s\n", matrix_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Matrix loaded: %d × %d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);

        if (num_modes > 1) {
            printf("NOTE: Multi-mode benchmark - performance may vary with order due to GPU state.\n");
            printf("      For accurate comparison, run each mode separately.\n");
        }

        matrix_rows = mat.rows;
        matrix_cols = mat.cols;
        matrix_nnz = mat.nnz;
        grid_size = mat.grid_size;
    }

    // Broadcast matrix dimensions to all ranks
    MPI_Bcast(&matrix_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&grid_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Non-zero ranks initialize mat structure and allocate entries
    if (rank != 0) {
        mat.rows = matrix_rows;
        mat.cols = matrix_cols;
        mat.nnz = matrix_nnz;
        mat.grid_size = grid_size;
        mat.entries = (Entry*)malloc(matrix_nnz * sizeof(Entry));
        if (!mat.entries) {
            fprintf(stderr, "[Rank %d] Failed to allocate matrix entries\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Create MPI datatype for Entry structure to avoid INT_MAX overflow with large matrices
    MPI_Datatype MPI_ENTRY;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};

    displacements[0] = offsetof(Entry, row);
    displacements[1] = offsetof(Entry, col);
    displacements[2] = offsetof(Entry, value);

    MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_ENTRY);
    MPI_Type_commit(&MPI_ENTRY);

    // Broadcast matrix entries to all ranks (AllGather requires full matrix)
    MPI_Bcast(mat.entries, matrix_nnz, MPI_ENTRY, 0, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_ENTRY);

    // Create deterministic RHS: b = ones (all ranks)
    double* b = (double*)malloc(matrix_rows * sizeof(double));
    double* x = (double*)calloc(matrix_rows, sizeof(double));

    for (int i = 0; i < matrix_rows; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

    // CG configuration
    CGConfigMultiGPU config;
    config.max_iters = max_iters;
    config.tolerance = tolerance;
    config.verbose = 2;
    config.enable_detailed_timers = 1;  // Enable detailed timers by default

    // Loop through all modes
    for (int mode_idx = 0; mode_idx < num_modes; mode_idx++) {
        const char* current_mode = mode_tokens[mode_idx];

        if (rank == 0) {
            printf("\n========================================\n");
            printf("CG Solver (Multi-GPU AllGather) - Mode: %s\n", current_mode);
            printf("========================================\n");
            printf("Tolerance: %.1e, Max iterations: %d\n", tolerance, max_iters);
        }

        // Get and initialize operator (all ranks)
        SpmvOperator* spmv_op = get_operator(current_mode);
        spmv_op->init(&mat);

        // Reset solution vector
        memset(x, 0, matrix_rows * sizeof(double));

        // Warmup: 5 full CG runs (HPC best practice for multi-GPU)
        if (rank == 0) printf("Warmup (5 runs)...\n");
        CGConfigMultiGPU warmup_config = config;
        warmup_config.verbose = 0;
        CGStatsMultiGPU warmup_stats;
        for (int w = 0; w < 5; w++) {
            memset(x, 0, matrix_rows * sizeof(double));
            cg_solve_mgpu(spmv_op, &mat, b, x, warmup_config, &warmup_stats);
        }

        // Benchmark: 30 runs with statistical analysis
        if (rank == 0) printf("Running benchmark (30 runs)...\n");
        BenchmarkStats bench_stats;
        CGStatsMultiGPU stats;
        cg_benchmark_with_stats_mgpu(spmv_op, &mat, b, x, config, 30, &bench_stats, &stats);

        if (rank == 0) {
            printf("Completed: %d valid runs, %d outliers removed\n",
                   bench_stats.valid_runs, bench_stats.outliers_removed);
        }

        // Display results for verification (rank 0 only)
        if (rank == 0) {
            printf("\n--- Results for %s ---\n", current_mode);
            printf("Converged: %s in %d iterations\n", stats.converged ? "YES" : "NO", stats.iterations);
            printf("Solution norm: %.15e\n", stats.residual_norm);

            // Timing summary with statistics
            printf("Time (median): %.3f ms (SpMV: %.1f%%, BLAS1: %.1f%%, Reductions: %.1f%%, AllReduce: %.1f%%, AllGather: %.1f%%)\n",
                   bench_stats.median_ms,
                   100.0 * stats.time_spmv_ms / stats.time_total_ms,
                   100.0 * stats.time_blas1_ms / stats.time_total_ms,
                   100.0 * stats.time_reductions_ms / stats.time_total_ms,
                   100.0 * stats.time_allreduce_ms / stats.time_total_ms,
                   100.0 * stats.time_allgather_ms / stats.time_total_ms);
            if (bench_stats.valid_runs > 1) {
                printf("Stats: min=%.3f ms, max=%.3f ms, std=%.3f ms\n",
                       bench_stats.min_ms, bench_stats.max_ms, bench_stats.std_dev_ms);
            }

            printf("GFLOPS (SpMV): %.3f\n",
                   (2.0 * matrix_nnz * stats.iterations) / (stats.time_spmv_ms * 1e6));
        }

        // Free GPU memory after each mode
        spmv_op->free();
    }

    if (rank == 0) {
        printf("\n========================================\n");
        printf("Multi-mode benchmark completed\n");
        printf("========================================\n");
    }

    // Cleanup
    free(b);
    free(x);
    if (mat.entries) {
        free(mat.entries);
    }

    MPI_Finalize();
    return 0;
}
