/**
 * @file cg_single_run_test.cu
 * @brief Single CG run for clean profiling (no warmup, no benchmark wrapper)
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "io.h"
#include "solvers/cg_solver_mgpu_partitioned.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <N> %s <matrix.mtx>\n", argv[0]);
            printf("Single CG run for profiling (no warmup, no wrapper)\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Load matrix (all ranks)
    MatrixData mat;
    if (load_matrix_market(argv[1], &mat) != 0) {
        fprintf(stderr, "[Rank %d] Error loading matrix\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Matrix: %d × %d, %d nnz\n", mat.rows, mat.cols, mat.nnz);
        printf("Running SINGLE CG solve (for profiling)...\n");
    }

    // Create RHS
    double* b = (double*)malloc(mat.rows * sizeof(double));
    double* x = (double*)calloc(mat.rows, sizeof(double));
    for (int i = 0; i < mat.rows; i++) {
        b[i] = 1.0;
    }

    // Config
    CGConfigMultiGPU config;
    config.max_iters = 1000;
    config.tolerance = 1e-6;
    config.verbose = 2;  // Print residual at each iteration
    config.enable_detailed_timers = 0;  // No sync overhead

    // === PROFILING TARGET: Single CG run ===
    CGStatsMultiGPU stats;
    cg_solve_mgpu_partitioned(NULL, &mat, b, x, config, &stats);

    if (rank == 0) {
        printf("\nConverged: %s in %d iterations\n",
               stats.converged ? "YES" : "NO", stats.iterations);
        printf("Total time: %.2f ms\n", stats.time_total_ms);
    }

    // Cleanup
    free(b);
    free(x);
    if (mat.entries) free(mat.entries);

    MPI_Finalize();
    return 0;
}
