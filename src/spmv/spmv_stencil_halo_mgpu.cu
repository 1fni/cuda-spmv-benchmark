/**
 * @file spmv_stencil_halo_mgpu.cu
 * @brief Multi-GPU SpMV with halo zones (standalone benchmark)
 *
 * @details
 * Extracted from CG solver for direct SpMV benchmarking.
 * Uses shared kernel: stencil5_csr_partitioned_halo_kernel
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "spmv.h"
#include "io.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Shared kernel declaration
extern __global__ void stencil5_csr_partitioned_halo_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev,
    const double* __restrict__ x_halo_next,
    double* __restrict__ y,
    int n_local,
    int row_offset,
    int N,
    int grid_size
);

// Global context
static struct {
    int rank, world_size;
    int n_local, row_offset, grid_size;

    int *d_row_ptr, *d_col_idx;
    double *d_values;
    double *d_x_local, *d_y_local;
    double *d_x_halo_prev, *d_x_halo_next;

    double *h_send_prev, *h_send_next;
    double *h_recv_prev, *h_recv_next;

    cudaStream_t stream;
} ctx;

// MPI halo exchange
static void exchange_halo(double* d_x, int halo_size) {
    MPI_Request requests[4];
    int req_count = 0;

    // D2H staging
    if (ctx.rank > 0) {
        CUDA_CHECK(cudaMemcpyAsync(ctx.h_send_prev, d_x, halo_size * sizeof(double),
                                    cudaMemcpyDeviceToHost, ctx.stream));
    }
    if (ctx.rank < ctx.world_size - 1) {
        CUDA_CHECK(cudaMemcpyAsync(ctx.h_send_next, &d_x[ctx.n_local - halo_size],
                                    halo_size * sizeof(double), cudaMemcpyDeviceToHost, ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // MPI exchange
    if (ctx.rank > 0) {
        MPI_Isend(ctx.h_send_prev, halo_size, MPI_DOUBLE, ctx.rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(ctx.h_recv_prev, halo_size, MPI_DOUBLE, ctx.rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    if (ctx.rank < ctx.world_size - 1) {
        MPI_Isend(ctx.h_send_next, halo_size, MPI_DOUBLE, ctx.rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(ctx.h_recv_next, halo_size, MPI_DOUBLE, ctx.rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // H2D staging
    if (ctx.rank > 0) {
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_x_halo_prev, ctx.h_recv_prev, halo_size * sizeof(double),
                                    cudaMemcpyHostToDevice, ctx.stream));
    }
    if (ctx.rank < ctx.world_size - 1) {
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_x_halo_next, ctx.h_recv_next, halo_size * sizeof(double),
                                    cudaMemcpyHostToDevice, ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

static int spmv_halo_mgpu_init(MatrixData* mat) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.world_size);
    CUDA_CHECK(cudaSetDevice(ctx.rank));

    int n = mat->rows;
    ctx.grid_size = (int)sqrt((double)n);
    if (ctx.grid_size * ctx.grid_size != n) {
        if (ctx.rank == 0) fprintf(stderr, "Error: Matrix must be square stencil\n");
        return -1;
    }

    ctx.n_local = n / ctx.world_size;
    ctx.row_offset = ctx.rank * ctx.n_local;

    if (ctx.rank == 0) {
        printf("Multi-GPU SpMV with halo zones: %d GPUs, %d×%d matrix\n",
               ctx.world_size, ctx.grid_size, ctx.grid_size);
    }

    // Build local CSR
    int nnz_local = 0;
    for (int e = 0; e < mat->nnz; e++) {
        if (mat->entries[e].row >= ctx.row_offset &&
            mat->entries[e].row < ctx.row_offset + ctx.n_local) {
            nnz_local++;
        }
    }

    int *h_row_ptr = (int*)malloc((ctx.n_local + 1) * sizeof(int));
    int *h_col_idx = (int*)malloc(nnz_local * sizeof(int));
    double *h_values = (double*)malloc(nnz_local * sizeof(double));

    h_row_ptr[0] = 0;
    int idx = 0;
    for (int lr = 0; lr < ctx.n_local; lr++) {
        int gr = ctx.row_offset + lr;
        for (int e = 0; e < mat->nnz; e++) {
            if (mat->entries[e].row == gr) {
                h_col_idx[idx] = mat->entries[e].col;
                h_values[idx] = mat->entries[e].value;
                idx++;
            }
        }
        h_row_ptr[lr + 1] = idx;
    }

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&ctx.d_row_ptr, (ctx.n_local + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_col_idx, nnz_local * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_values, nnz_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_x_local, ctx.n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_y_local, ctx.n_local * sizeof(double)));

    if (ctx.rank > 0) {
        CUDA_CHECK(cudaMalloc(&ctx.d_x_halo_prev, ctx.grid_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&ctx.h_send_prev, ctx.grid_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&ctx.h_recv_prev, ctx.grid_size * sizeof(double)));
    }
    if (ctx.rank < ctx.world_size - 1) {
        CUDA_CHECK(cudaMalloc(&ctx.d_x_halo_next, ctx.grid_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&ctx.h_send_next, ctx.grid_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&ctx.h_recv_next, ctx.grid_size * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(ctx.d_row_ptr, h_row_ptr, (ctx.n_local + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_col_idx, h_col_idx, nnz_local * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_values, h_values, nnz_local * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamCreate(&ctx.stream));

    free(h_row_ptr);
    free(h_col_idx);
    free(h_values);

    return 0;
}

static int spmv_halo_mgpu_run_timed(const double* x, double* y, double* time_ms) {
    CUDA_CHECK(cudaMemcpy(ctx.d_x_local, &x[ctx.row_offset],
                          ctx.n_local * sizeof(double), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, ctx.stream));

    exchange_halo(ctx.d_x_local, ctx.grid_size);

    int threads = 256;
    int blocks = (ctx.n_local + threads - 1) / threads;
    stencil5_csr_partitioned_halo_kernel<<<blocks, threads, 0, ctx.stream>>>(
        ctx.d_row_ptr, ctx.d_col_idx, ctx.d_values,
        ctx.d_x_local, ctx.d_x_halo_prev, ctx.d_x_halo_next,
        ctx.d_y_local, ctx.n_local, ctx.row_offset,
        ctx.grid_size * ctx.grid_size, ctx.grid_size
    );

    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_ms_float;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms_float, start, stop));
    *time_ms = (double)time_ms_float;

    CUDA_CHECK(cudaMemcpy(&y[ctx.row_offset], ctx.d_y_local,
                          ctx.n_local * sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Gather(ctx.rank == 0 ? MPI_IN_PLACE : &y[ctx.row_offset], ctx.n_local, MPI_DOUBLE,
               y, ctx.n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}

static void spmv_halo_mgpu_free() {
    if (ctx.d_row_ptr) cudaFree(ctx.d_row_ptr);
    if (ctx.d_col_idx) cudaFree(ctx.d_col_idx);
    if (ctx.d_values) cudaFree(ctx.d_values);
    if (ctx.d_x_local) cudaFree(ctx.d_x_local);
    if (ctx.d_y_local) cudaFree(ctx.d_y_local);
    if (ctx.d_x_halo_prev) cudaFree(ctx.d_x_halo_prev);
    if (ctx.d_x_halo_next) cudaFree(ctx.d_x_halo_next);
    if (ctx.h_send_prev) cudaFreeHost(ctx.h_send_prev);
    if (ctx.h_send_next) cudaFreeHost(ctx.h_send_next);
    if (ctx.h_recv_prev) cudaFreeHost(ctx.h_recv_prev);
    if (ctx.h_recv_next) cudaFreeHost(ctx.h_recv_next);
    if (ctx.stream) cudaStreamDestroy(ctx.stream);
}

SpmvOperator SPMV_STENCIL_HALO_MGPU = {
    .name = "stencil5-halo-mgpu",
    .init = spmv_halo_mgpu_init,
    .run_timed = spmv_halo_mgpu_run_timed,
    .free = spmv_halo_mgpu_free
};
