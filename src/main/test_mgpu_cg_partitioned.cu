/**
 * @file test_mgpu_cg_partitioned.cu
 * @brief Test program for partitioned multi-GPU CG solver
 *
 * Launch: mpirun -np 2 ./bin/test_mgpu_cg_partitioned matrix/stencil_512x512.mtx
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-11
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "io.h"
#include "spmv.h"
#include "solvers/cg_solver_mgpu_partitioned.h"

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <N> %s <matrix.mtx> [--timers]\n", argv[0]);
            printf("Example: mpirun -np 2 %s matrix/stencil_512x512.mtx\n", argv[0]);
            printf("Options:\n");
            printf("  --timers  Enable detailed timing breakdown (adds GPU sync overhead)\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char* matrix_file = argv[1];

    // Each rank loads matrix independently (avoids MPI_Bcast size limit)
    MatrixData mat;

    if (rank == 0) {
        printf("Loading matrix: %s\n", matrix_file);
    }

    if (load_matrix_market(matrix_file, &mat) != 0) {
        fprintf(stderr, "[Rank %d] Error loading matrix: %s\n", rank, matrix_file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Matrix loaded: %d × %d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);
        printf("\nCalling partitioned multi-GPU CG solver...\n");
    }

    // Create deterministic RHS: b = ones (all ranks)
    double* b = (double*)malloc(mat.rows * sizeof(double));
    double* x = (double*)calloc(mat.rows, sizeof(double));

    for (int i = 0; i < mat.rows; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

    // CG configuration
    CGConfigMultiGPU config;
    config.max_iters = 100;
    config.tolerance = 1e-6;
    config.verbose = 2;

    // Detailed timers: disabled by default (no sync overhead), enable with --timers flag
    config.enable_detailed_timers = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--timers") == 0) {
            config.enable_detailed_timers = 1;
            if (rank == 0) {
                printf("Detailed timers enabled (adds sync overhead)\n");
            }
        }
    }

    // Solver statistics
    CGStatsMultiGPU stats;

    // Call partitioned multi-GPU CG solver
    cg_solve_mgpu_partitioned(NULL, &mat, b, x, config, &stats);

    // Display results for verification (rank 0 only)
    if (rank == 0) {
        printf("\n========================================\n");
        printf("CG Solution (first 10 values):\n");
        printf("========================================\n");
        for (int i = 0; i < 10 && i < mat.rows; i++) {
            printf("x[%d] = %.15e\n", i, x[i]);
        }
        printf("...\n");
        printf("x[%d] = %.15e (last)\n", mat.rows - 1, x[mat.rows - 1]);

        printf("\nSolution norm: %.15e\n", stats.residual_norm);
        printf("Converged: %s\n", stats.converged ? "YES" : "NO");
        printf("Iterations: %d\n", stats.iterations);
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
