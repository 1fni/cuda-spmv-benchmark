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
#include <mpi.h>
#include "io.h"
#include "solvers/cg_solver_mgpu.h"
#include "spmv.h"

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <N> %s <matrix.mtx>\n", argv[0]);
            printf("Example: mpirun -np 2 %s matrix/stencil_512x512.mtx\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char* matrix_file = argv[1];

    // Rank 0 loads matrix and broadcasts dimensions
    MatrixData mat;
    int matrix_rows, matrix_cols, matrix_nnz, grid_size;

    if (rank == 0) {
        if (load_matrix_market(matrix_file, &mat) != 0) {
            fprintf(stderr, "Error loading matrix: %s\n", matrix_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Matrix loaded: %d × %d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);

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

    // Non-zero ranks initialize mat structure
    if (rank != 0) {
        mat.rows = matrix_rows;
        mat.cols = matrix_cols;
        mat.nnz = matrix_nnz;
        mat.grid_size = grid_size;
        mat.entries = NULL;
    }

    // Create deterministic RHS: b = ones (all ranks)
    double* b = (double*)malloc(matrix_rows * sizeof(double));
    double* x = (double*)calloc(matrix_rows, sizeof(double));

    for (int i = 0; i < matrix_rows; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

    // CG configuration
    CGConfigMultiGPU config;
    config.max_iters = 100;
    config.tolerance = 1e-6;
    config.verbose = 2;
    config.enable_detailed_timers = 1;  // Enable detailed timers by default

    // Solver statistics
    CGStatsMultiGPU stats;

    // Call multi-GPU CG solver with MPI distribution
    if (rank == 0) {
        printf("\nCalling multi-GPU CG solver...\n\n");
    }

    cg_solve_mgpu(NULL,  // SpMV handled internally
                  &mat,
                  b,
                  x,
                  config,
                  &stats);

    // Display results for verification (rank 0 only)
    if (rank == 0) {
        printf("\n========================================\n");
        printf("CG Solution (first 10 values):\n");
        printf("========================================\n");
        for (int i = 0; i < 10 && i < matrix_rows; i++) {
            printf("x[%d] = %.15e\n", i, x[i]);
        }
        printf("...\n");
        printf("x[%d] = %.15e (last)\n", matrix_rows - 1, x[matrix_rows - 1]);

        printf("\nSolution norm: %.15e\n", stats.residual_norm);
        printf("Converged: %s\n", stats.converged ? "YES" : "NO");
        printf("Iterations: %d\n", stats.iterations);

        // Timing breakdown
        if (config.enable_detailed_timers) {
            printf("\n========================================\n");
            printf("Timing Breakdown\n");
            printf("========================================\n");
            printf("Total time:     %.2f ms\n", stats.time_total_ms);
            printf("  SpMV:         %.2f ms (%.1f%%)\n",
                   stats.time_spmv_ms, 100.0 * stats.time_spmv_ms / stats.time_total_ms);
            printf("  BLAS1:        %.2f ms (%.1f%%)\n",
                   stats.time_blas1_ms, 100.0 * stats.time_blas1_ms / stats.time_total_ms);
            printf("  AllReduce:    %.2f ms (%.1f%%)\n",
                   stats.time_allreduce_ms, 100.0 * stats.time_allreduce_ms / stats.time_total_ms);
            printf("  AllGather:    %.2f ms (%.1f%%)\n",
                   stats.time_allgather_ms, 100.0 * stats.time_allgather_ms / stats.time_total_ms);
        }
        printf("========================================\n");
    }

    // Cleanup
    free(b);
    free(x);
    if (rank == 0 && mat.entries) {
        free(mat.entries);
    }

    MPI_Finalize();
    return 0;
}
