// Standalone AmgX SpMV benchmark for stencil matrices
#include <stdio.h>
#include <stdlib.h>
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
        fprintf(stderr, "AmgX error at %s:%d: code %d\n", __FILE__, __LINE__, err); \
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
        fscanf(f, "%d %d %lf", &coo_rows[i], &coo_cols[i], &coo_vals[i]);
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
        fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    const char* matrix_file = argv[1];

    // Load matrix
    MatrixMarket mat = read_matrix_market(matrix_file);
    printf("Matrix: %dx%d, %d nnz\n", mat.rows, mat.cols, mat.nnz);

    // Initialize AmgX
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());

    // Create config for SpMV only
    const char* config_string = "config_version=2, determinism_flag=1";
    AMGX_config_handle cfg;
    AMGX_CHECK(AMGX_config_create(&cfg, config_string));

    // Create resources
    AMGX_resources_handle rsrc;
    AMGX_CHECK(AMGX_resources_create_simple(&rsrc, cfg));

    // Create matrix and vectors
    AMGX_matrix_handle A;
    AMGX_vector_handle x, y;
    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&y, rsrc, AMGX_mode_dDDI));

    // Upload matrix
    AMGX_CHECK(AMGX_matrix_upload_all(A, mat.rows, mat.nnz, 1, 1,
                                      mat.row_ptr, mat.col_idx, mat.values, NULL));

    // Allocate vectors
    double *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, mat.rows * sizeof(double)));

    // Initialize x = 1.0
    double *h_x = (double*)malloc(mat.rows * sizeof(double));
    for (int i = 0; i < mat.rows; i++) h_x[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mat.rows * sizeof(double), cudaMemcpyHostToDevice));

    AMGX_CHECK(AMGX_vector_upload(x, mat.rows, 1, d_x));
    AMGX_CHECK(AMGX_vector_upload(y, mat.rows, 1, d_y));

    // Warmup
    for (int i = 0; i < 10; i++) {
        AMGX_CHECK(AMGX_matrix_vector_multiply(A, x, y));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        AMGX_CHECK(AMGX_matrix_vector_multiply(A, x, y));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Download result and compute checksum
    AMGX_CHECK(AMGX_vector_download(y, d_y));
    double *h_y = (double*)malloc(mat.rows * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, mat.rows * sizeof(double), cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (int i = 0; i < mat.rows; i++) {
        checksum += h_y[i];
    }

    // Output
    printf("AmgX SpMV (stencil reference)\n");
    printf("Rows: %d\n", mat.rows);
    printf("NNZ: %d\n", mat.nnz);
    printf("Time: %.3f ms\n", time_ms);
    printf("Checksum: %.6e\n", checksum);

    // Cleanup
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    AMGX_vector_destroy(y);
    AMGX_vector_destroy(x);
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
