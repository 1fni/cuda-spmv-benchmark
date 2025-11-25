/**
 * @file amgx_cg_solver_mgpu.cpp
 * @brief Multi-GPU CG solver using NVIDIA AmgX with MPI
 *
 * AmgX handles all internal communications (halos, overlap, etc).
 * We only provide: local CSR partitions + global_row_ids mapping.
 *
 * Launch: mpirun -np N ./amgx_cg_solver_mgpu matrix.mtx [options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include "amgx_benchmark.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define AMGX_CHECK(call) do { \
    AMGX_RC err = call; \
    if (err != AMGX_RC_OK) { \
        fprintf(stderr, "[Rank %d] AMGX error at %s:%d: code %d\n", rank, __FILE__, __LINE__, err); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

struct MatrixMarket {
    int rows, cols, nnz;
    int *row_ptr;
    int *col_idx;
    double *values;
};

MatrixMarket read_matrix_market(const char* filename, int rank) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "[Rank %d] Failed to open %s\n", rank, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
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
            fprintf(stderr, "[Rank %d] Error reading matrix line %d\n", rank, i);
            MPI_Abort(MPI_COMM_WORLD, 1);
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

struct RunResult {
    double time_ms;
    int iterations;
    bool converged;
};

RunResult run_amgx_solve_mgpu(AMGX_solver_handle solver,
                               AMGX_vector_handle b,
                               AMGX_vector_handle x,
                               double* d_x,
                               int n_local,
                               bool verbose,
                               int rank) {
    // Reset solution vector
    double *h_x_zero = (double*)calloc(n_local, sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_x, h_x_zero, n_local * sizeof(double), cudaMemcpyHostToDevice));
    AMGX_CHECK(AMGX_vector_upload(x, n_local, 1, d_x));
    free(h_x_zero);

    MPI_Barrier(MPI_COMM_WORLD);  // Sync all ranks before timing

    auto start = std::chrono::high_resolution_clock::now();
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));
    auto end = std::chrono::high_resolution_clock::now();

    RunResult result;
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    AMGX_SOLVE_STATUS status;
    AMGX_CHECK(AMGX_solver_get_status(solver, &status));
    result.converged = (status == AMGX_SOLVE_SUCCESS);

    AMGX_CHECK(AMGX_solver_get_iterations_number(solver, &result.iterations));

    return result;
}

double calculate_median(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    if (n % 2 == 0) {
        return (times[n/2 - 1] + times[n/2]) / 2.0;
    }
    return times[n/2];
}

double calculate_mean(const std::vector<double>& times) {
    double sum = 0.0;
    for (double t : times) sum += t;
    return sum / times.size();
}

double calculate_std_dev(const std::vector<double>& times, double mean) {
    double sum_sq = 0.0;
    for (double t : times) {
        double diff = t - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / times.size());
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <N> %s <matrix.mtx> [--tol=1e-6] [--max-iters=1000] [--runs=10] [--json=<file>] [--csv=<file>]\n", argv[0]);
            fprintf(stderr, "Example: mpirun -np 4 %s matrix/stencil_5000x5000.mtx --runs=10 --json=results/amgx_mgpu.json\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char* matrix_file = argv[1];
    double tolerance = 1e-6;
    int max_iters = 1000;
    int num_runs = 10;
    const char* json_file = nullptr;
    const char* csv_file = nullptr;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--tol=", 6) == 0) {
            tolerance = atof(argv[i] + 6);
        } else if (strncmp(argv[i], "--max-iters=", 12) == 0) {
            max_iters = atoi(argv[i] + 12);
        } else if (strncmp(argv[i], "--runs=", 7) == 0) {
            num_runs = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--json=", 7) == 0) {
            json_file = argv[i] + 7;
        } else if (strncmp(argv[i], "--csv=", 6) == 0) {
            csv_file = argv[i] + 6;
        }
    }

    if (rank == 0) {
        printf("========================================\n");
        printf("AmgX Multi-GPU CG Solver\n");
        printf("========================================\n");
        printf("Matrix: %s\n", matrix_file);
        printf("MPI ranks: %d\n", world_size);
        printf("Tolerance: %.0e\n", tolerance);
        printf("Max iterations: %d\n", max_iters);
        printf("Benchmark runs: %d\n\n", num_runs);
    }

    // Set GPU device (one GPU per MPI rank)
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));

    if (rank == 0) {
        printf("GPU assignment: %d GPUs, rank %d → GPU %d\n\n", num_devices, rank, device_id);
    }

    // Load full matrix (each rank independently)
    MatrixMarket mat = read_matrix_market(matrix_file, rank);
    int grid_size = (int)sqrt(mat.rows);

    if (rank == 0) {
        printf("Matrix loaded: %dx%d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);
        printf("Grid: %dx%d\n\n", grid_size, grid_size);
    }

    // Row-band partitioning
    int n_local = mat.rows / world_size;
    int row_offset = rank * n_local;
    if (rank == world_size - 1) {
        n_local = mat.rows - row_offset;  // Last rank takes remainder
    }

    // Create local partition (rows [row_offset : row_offset + n_local))
    int *local_row_ptr = (int*)malloc((n_local + 1) * sizeof(int));
    local_row_ptr[0] = 0;

    // Count nnz in local partition
    int local_nnz = 0;
    for (int i = 0; i < n_local; i++) {
        int global_row = row_offset + i;
        int row_nnz = mat.row_ptr[global_row + 1] - mat.row_ptr[global_row];
        local_nnz += row_nnz;
        local_row_ptr[i + 1] = local_nnz;
    }

    // Extract local CSR partition
    int *local_col_idx = (int*)malloc(local_nnz * sizeof(int));
    double *local_values = (double*)malloc(local_nnz * sizeof(double));

    for (int i = 0; i < n_local; i++) {
        int global_row = row_offset + i;
        int src_start = mat.row_ptr[global_row];
        int row_nnz = mat.row_ptr[global_row + 1] - src_start;
        int dst_start = local_row_ptr[i];

        memcpy(&local_col_idx[dst_start], &mat.col_idx[src_start], row_nnz * sizeof(int));
        memcpy(&local_values[dst_start], &mat.values[src_start], row_nnz * sizeof(double));
    }

    if (rank == 0) {
        printf("Partition created: rank %d has rows [%d:%d), %d nnz\n",
               rank, row_offset, row_offset + n_local, local_nnz);
    }

    // Gather partition sizes from all ranks for offsets array
    int *partition_sizes = nullptr;
    if (rank == 0) {
        partition_sizes = (int*)malloc((world_size + 1) * sizeof(int));
    }
    MPI_Gather(&n_local, 1, MPI_INT, partition_sizes + 1, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *partition_offsets = nullptr;
    if (rank == 0) {
        partition_offsets = (int*)malloc((world_size + 1) * sizeof(int));
        partition_offsets[0] = 0;
        for (int i = 1; i <= world_size; i++) {
            partition_offsets[i] = partition_offsets[i-1] + partition_sizes[i];
        }
    }

    // Initialize AmgX
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());

    // Create config for PCG solver (no preconditioning, fair comparison)
    char config_string[512];
    snprintf(config_string, sizeof(config_string),
             "config_version=2, "
             "solver=PCG, "
             "preconditioner=NOSOLVER, "
             "max_iters=%d, "
             "convergence=RELATIVE_INI, "
             "tolerance=%.15e, "
             "norm=L2, "
             "print_solve_stats=0, "
             "monitor_residual=1, "  // Required for convergence check
             "obtain_timings=0",
             max_iters, tolerance);

    AMGX_config_handle cfg;
    AMGX_CHECK(AMGX_config_create(&cfg, config_string));

    // Create resources with MPI communicator
    AMGX_resources_handle rsrc;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    void* mpi_comm_ptr = (void*)&mpi_comm;
    AMGX_CHECK(AMGX_resources_create(&rsrc, cfg, &mpi_comm_ptr, 1, &device_id));

    // Create distribution object for partitioned matrix
    AMGX_distribution_handle dist;
    AMGX_CHECK(AMGX_distribution_create(&dist, cfg));

    // Broadcast partition offsets to all ranks
    if (rank != 0) {
        partition_offsets = (int*)malloc((world_size + 1) * sizeof(int));
    }
    MPI_Bcast(partition_offsets, world_size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    AMGX_CHECK(AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partition_offsets));
    AMGX_CHECK(AMGX_distribution_set_32bit_colindices(dist, 1));  // Use 32-bit column indices

    // Create matrix, vectors, and solver
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;

    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    // Debug: print partition offsets
    if (rank == 0) {
        printf("Partition offsets: [");
        for (int i = 0; i <= world_size; i++) {
            printf("%d%s", partition_offsets[i], (i < world_size) ? ", " : "");
        }
        printf("]\n");
        printf("Uploading matrix: n_global=%d, n_local=%d, local_nnz=%d\n\n",
               mat.rows, n_local, local_nnz);
    }

    // Upload local matrix partition with distribution
    AMGX_CHECK(AMGX_matrix_upload_distributed(A,
                                               mat.rows,      // n_global
                                               n_local,       // n (local rows)
                                               local_nnz,     // nnz (local)
                                               1, 1,          // block_dimx, block_dimy
                                               local_row_ptr,
                                               local_col_idx, // Global column indices
                                               local_values,
                                               nullptr,       // diag_data
                                               dist));

    // Create RHS: b = ones
    if (rank == 0) {
        printf("RHS: b = ones, Initial guess: x0 = 0\n\n");
    }

    double *h_b = (double*)malloc(n_local * sizeof(double));
    for (int i = 0; i < n_local; i++) h_b[i] = 1.0;

    double *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n_local * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n_local * sizeof(double), cudaMemcpyHostToDevice));

    AMGX_CHECK(AMGX_vector_upload(b, n_local, 1, d_b));

    // Setup solver
    AMGX_CHECK(AMGX_solver_setup(solver, A));

    // Warmup
    if (rank == 0) printf("Warmup (3 runs)...\n");
    for (int i = 0; i < 3; i++) {
        run_amgx_solve_mgpu(solver, b, x, d_x, n_local, false, rank);
    }

    // Benchmark
    if (rank == 0) printf("Running benchmark (%d runs)...\n", num_runs);
    std::vector<RunResult> results;
    for (int i = 0; i < num_runs; i++) {
        results.push_back(run_amgx_solve_mgpu(solver, b, x, d_x, n_local, false, rank));
    }

    // Extract times (rank 0 only, all ranks have same timing)
    std::vector<double> times;
    for (const auto& r : results) {
        times.push_back(r.time_ms);
    }

    // Calculate statistics (rank 0)
    if (rank == 0) {
        double mean = calculate_mean(times);
        double std_dev = calculate_std_dev(times, mean);

        // Remove outliers (>2 std devs)
        std::vector<double> filtered_times;
        for (double t : times) {
            if (fabs(t - mean) <= 2.0 * std_dev) {
                filtered_times.push_back(t);
            }
        }

        double median = calculate_median(filtered_times);
        double final_mean = calculate_mean(filtered_times);
        double final_std = calculate_std_dev(filtered_times, final_mean);

        std::sort(filtered_times.begin(), filtered_times.end());
        double min_time = filtered_times.front();
        double max_time = filtered_times.back();

        int outliers_removed = times.size() - filtered_times.size();

        printf("Completed: %zu valid runs, %d outliers removed\n\n",
               filtered_times.size(), outliers_removed);

        // Print results
        printf("========================================\n");
        printf("Results\n");
        printf("========================================\n");
        printf("Converged: %s\n", results[0].converged ? "YES" : "NO");
        printf("Iterations: %d\n", results[0].iterations);
        printf("Time (median): %.3f ms\n", median);
        printf("Stats: min=%.3f ms, max=%.3f ms, std=%.3f ms\n",
               min_time, max_time, final_std);

        double gflops = (2.0 * mat.nnz * results[0].iterations) / (median * 1e6);
        printf("GFLOPS: %.3f\n", gflops);
        printf("========================================\n");

        // Export results
        if (json_file || csv_file) {
            char mode_str[64];
            snprintf(mode_str, sizeof(mode_str), "multi-gpu-%d", world_size);

            MatrixInfo mat_info = {mat.rows, mat.cols, mat.nnz, grid_size};
            BenchmarkResults bench_results = {
                results[0].converged,
                results[0].iterations,
                median,
                median,
                final_mean,
                min_time,
                max_time,
                final_std,
                (int)filtered_times.size(),
                outliers_removed
            };

            if (json_file) {
                export_amgx_json(json_file, mode_str, &mat_info, &bench_results);
            }
            if (csv_file) {
                export_amgx_csv(csv_file, mode_str, &mat_info, &bench_results, true);
            }
        }
    }

    // Cleanup
    free(h_b);
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));

    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_distribution_destroy(dist);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);

    AMGX_finalize_plugins();
    AMGX_finalize();

    free(mat.row_ptr);
    free(mat.col_idx);
    free(mat.values);
    free(local_row_ptr);
    free(local_col_idx);
    free(local_values);
    free(partition_offsets);
    if (rank == 0 && partition_sizes) {
        free(partition_sizes);
    }

    MPI_Finalize();
    return 0;
}
