/**
 * @file amgx_cg_solver.cpp
 * @brief CG solver using NVIDIA AMGX (official reference)
 *
 * Provides fair comparison with NVIDIA's optimized production solver.
 * Uses AMGX PCG solver without preconditioning.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define AMGX_CHECK(call) do { \
    AMGX_RC err = call; \
    if (err != AMGX_RC_OK) { \
        fprintf(stderr, "AMGX error at %s:%d: code %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct MatrixMarket {
    int rows, cols, nnz;
    int *row_ptr;
    int *col_idx;
    double *values;
};

MatrixMarket read_matrix_market(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Skip comments
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    MatrixMarket mat;
    sscanf(line, "%d %d %d", &mat.rows, &mat.cols, &mat.nnz);

    // Temporary storage for COO format
    int *coo_rows = (int*)malloc(mat.nnz * sizeof(int));
    int *coo_cols = (int*)malloc(mat.nnz * sizeof(int));
    double *coo_vals = (double*)malloc(mat.nnz * sizeof(double));

    for (int i = 0; i < mat.nnz; i++) {
        int ret = fscanf(f, "%d %d %lf", &coo_rows[i], &coo_cols[i], &coo_vals[i]);
        if (ret != 3) {
            fprintf(stderr, "Error reading matrix line %d\n", i);
            exit(EXIT_FAILURE);
        }
        coo_rows[i]--; // 1-based to 0-based
        coo_cols[i]--;
    }
    fclose(f);

    // Convert COO to CSR
    mat.row_ptr = (int*)calloc(mat.rows + 1, sizeof(int));
    mat.col_idx = (int*)malloc(mat.nnz * sizeof(int));
    mat.values = (double*)malloc(mat.nnz * sizeof(double));

    // Count non-zeros per row
    for (int i = 0; i < mat.nnz; i++) {
        mat.row_ptr[coo_rows[i] + 1]++;
    }

    // Prefix sum
    for (int i = 1; i <= mat.rows; i++) {
        mat.row_ptr[i] += mat.row_ptr[i - 1];
    }

    // Fill CSR arrays
    int *local_count = (int*)calloc(mat.rows, sizeof(int));
    for (int i = 0; i < mat.nnz; i++) {
        int r = coo_rows[i];
        int dst = mat.row_ptr[r] + local_count[r]++;
        mat.col_idx[dst] = coo_cols[i];
        mat.values[dst] = coo_vals[i];
    }

    free(coo_rows);
    free(coo_cols);
    free(coo_vals);
    free(local_count);

    return mat;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx> [--tol=1e-6] [--max-iters=1000]\n", argv[0]);
        fprintf(stderr, "Example: %s ../../matrix/stencil_512x512.mtx\n", argv[0]);
        return 1;
    }

    const char* matrix_file = argv[1];
    double tolerance = 1e-6;
    int max_iters = 1000;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--tol=", 6) == 0) {
            tolerance = atof(argv[i] + 6);
        } else if (strncmp(argv[i], "--max-iters=", 12) == 0) {
            max_iters = atoi(argv[i] + 12);
        }
    }

    printf("========================================\n");
    printf("AMGX CG Solver (NVIDIA Reference)\n");
    printf("========================================\n");
    printf("Matrix: %s\n", matrix_file);
    printf("Tolerance: %.0e\n", tolerance);
    printf("Max iterations: %d\n\n", max_iters);

    // Load matrix
    MatrixMarket mat = read_matrix_market(matrix_file);
    int grid_size = (int)sqrt(mat.rows);
    printf("Matrix loaded: %dx%d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);
    printf("Grid: %dx%d\n\n", grid_size, grid_size);

    // Initialize AMGX
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());

    // Create config for PCG solver (no preconditioning)
    char config_string[512];
    snprintf(config_string, sizeof(config_string),
             "config_version=2, "
             "solver=PCG, "
             "preconditioner=NOSOLVER, "
             "max_iters=%d, "
             "convergence=RELATIVE_INI_CORE, "
             "tolerance=%.15e, "
             "norm=L2, "
             "print_solve_stats=1, "
             "monitor_residual=1, "
             "obtain_timings=1",
             max_iters, tolerance);

    AMGX_config_handle cfg;
    AMGX_CHECK(AMGX_config_create(&cfg, config_string));

    // Create resources
    AMGX_resources_handle rsrc;
    AMGX_CHECK(AMGX_resources_create_simple(&rsrc, cfg));

    // Create matrix, vectors, and solver
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;

    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    // Upload matrix
    AMGX_CHECK(AMGX_matrix_upload_all(A, mat.rows, mat.nnz, 1, 1,
                                      mat.row_ptr, mat.col_idx, mat.values, NULL));

    // Create RHS: b = ones (for consistency across all solvers)
    printf("RHS created (b = ones)\n");
    printf("Initial guess: x0 = 0\n\n");

    double *h_b = (double*)malloc(mat.rows * sizeof(double));
    double *h_x = (double*)malloc(mat.rows * sizeof(double));

    for (int i = 0; i < mat.rows; i++) {
        h_b[i] = 1.0;
        h_x[i] = 0.0;
    }

    // Upload vectors
    double *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_b, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, mat.rows * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mat.rows * sizeof(double), cudaMemcpyHostToDevice));

    AMGX_CHECK(AMGX_vector_upload(b, mat.rows, 1, d_b));
    AMGX_CHECK(AMGX_vector_upload(x, mat.rows, 1, d_x));

    // Setup solver
    AMGX_CHECK(AMGX_solver_setup(solver, A));

    // Solve
    printf("Starting AMGX CG solver...\n");
    printf("========================================\n");

    auto start = std::chrono::high_resolution_clock::now();

    AMGX_CHECK(AMGX_solver_solve(solver, b, x));

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double, std::milli>(end - start).count();

    printf("========================================\n\n");

    // Get solver status
    AMGX_SOLVE_STATUS status;
    AMGX_CHECK(AMGX_solver_get_status(solver, &status));

    int iterations;
    AMGX_CHECK(AMGX_solver_get_iterations_number(solver, &iterations));

    // Download solution
    AMGX_CHECK(AMGX_vector_download(x, d_x));
    CUDA_CHECK(cudaMemcpy(h_x, d_x, mat.rows * sizeof(double), cudaMemcpyDeviceToHost));

    // Summary
    printf("========================================\n");
    printf("Summary\n");
    printf("========================================\n");
    printf("Converged: %s\n", (status == AMGX_SOLVE_SUCCESS) ? "YES" : "NO");
    printf("Iterations: %d\n", iterations);
    printf("Time-to-solution: %.3f ms\n", solve_time);
    printf("========================================\n");

    // Cleanup
    free(h_b);
    free(h_x);
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));

    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);

    AMGX_finalize_plugins();
    AMGX_finalize();

    free(mat.row_ptr);
    free(mat.col_idx);
    free(mat.values);

    return 0;
}
