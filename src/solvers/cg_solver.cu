/**
 * @file cg_solver.cu
 * @brief Conjugate Gradient solver implementation
 *
 * @details
 * Classic CG algorithm with detailed timing breakdown:
 * - SpMV operations
 * - BLAS1 operations (axpy, scale)
 * - Reductions (dot products, norms)
 *
 * Algorithm:
 *   r0 = b - A*x0
 *   p0 = r0
 *   for k=0,1,2,...
 *     alpha = (r,r) / (A*p, p)
 *     x = x + alpha*p
 *     r_new = r - alpha*A*p
 *     beta = (r_new,r_new) / (r,r)
 *     p = r_new + beta*p
 *     if ||r_new|| < tol: break
 *
 * Author: Bouhrour Stephane
 * Date: 2025-10-14
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "solvers/cg_solver.h"
#include "spmv.h"
#include "io.h"

// CUDA kernels for BLAS1 operations

/**
 * @brief AXPY: y = y + alpha*x
 */
__global__ void axpy_kernel(int n, double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = y[i] + alpha * x[i];
    }
}

/**
 * @brief AXPBY: z = alpha*x + beta*y
 */
__global__ void axpby_kernel(int n, double alpha, const double* x,
                              double beta, const double* y, double* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = alpha * x[i] + beta * y[i];
    }
}

/**
 * @brief Copy: y = x
 */
__global__ void copy_kernel(int n, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}

/**
 * @brief Dot product reduction - stage 1: per-block partial sums
 */
__global__ void dot_kernel(int n, const double* x, const double* y, double* block_results) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute partial dot product
    sdata[tid] = (i < n) ? x[i] * y[i] : 0.0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Dot product reduction - stage 2: sum block results on CPU
 */
double sum_block_results(const double* d_block_results, int num_blocks) {
    double* h_block_results = (double*)malloc(num_blocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_block_results, d_block_results, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        sum += h_block_results[i];
    }

    free(h_block_results);
    return sum;
}

/**
 * @brief CG solver implementation
 */
int cg_solve(SpmvOperator* spmv_op,
             MatrixData* mat,
             const double* b,
             double* x,
             CGConfig config,
             CGStats* stats) {

    int n = mat->rows;

    // Launch configuration
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Allocate device vectors
    double *d_x, *d_b, *d_r, *d_p, *d_Ap, *d_block_results;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_results, blocks * sizeof(double)));

    // Transfer initial data
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_spmv, stop_spmv;
    cudaEvent_t start_blas, stop_blas;
    cudaEvent_t start_reduce, stop_reduce;

    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_spmv));
    CUDA_CHECK(cudaEventCreate(&stop_spmv));
    CUDA_CHECK(cudaEventCreate(&start_blas));
    CUDA_CHECK(cudaEventCreate(&stop_blas));
    CUDA_CHECK(cudaEventCreate(&start_reduce));
    CUDA_CHECK(cudaEventCreate(&stop_reduce));

    double time_spmv = 0.0, time_blas = 0.0, time_reduce = 0.0;
    float ms;

    CUDA_CHECK(cudaEventRecord(start_total));

    // r0 = b - A*x0
    // Note: current SpMV operators expect host pointers, so we allocate temp host vectors
    double *h_temp_in = (double*)malloc(n * sizeof(double));
    double *h_temp_out = (double*)malloc(n * sizeof(double));

    double kernel_time_spmv;
    CUDA_CHECK(cudaEventRecord(start_spmv));
    CUDA_CHECK(cudaMemcpy(h_temp_in, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    spmv_op->run_timed(h_temp_in, h_temp_out, &kernel_time_spmv);  // h_temp_out = A*x0
    CUDA_CHECK(cudaMemcpy(d_Ap, h_temp_out, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_spmv));
    CUDA_CHECK(cudaEventSynchronize(stop_spmv));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_spmv, stop_spmv));
    time_spmv += ms;

    CUDA_CHECK(cudaEventRecord(start_blas));
    axpby_kernel<<<blocks, threads>>>(n, 1.0, d_b, -1.0, d_Ap, d_r);  // r = b - A*x0
    CUDA_CHECK(cudaEventRecord(stop_blas));
    CUDA_CHECK(cudaEventSynchronize(stop_blas));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_blas, stop_blas));
    time_blas += ms;

    // p0 = r0
    copy_kernel<<<blocks, threads>>>(n, d_r, d_p);

    // (r0, r0)
    CUDA_CHECK(cudaEventRecord(start_reduce));
    dot_kernel<<<blocks, threads, threads * sizeof(double)>>>(n, d_r, d_r, d_block_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    double rr_old = sum_block_results(d_block_results, blocks);
    CUDA_CHECK(cudaEventRecord(stop_reduce));
    CUDA_CHECK(cudaEventSynchronize(stop_reduce));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_reduce, stop_reduce));
    time_reduce += ms;

    double b_norm = sqrt(rr_old);  // Initial residual is b norm for x0=0

    if (config.verbose >= 1) {
        printf("[CG] Initial residual: %e\n", b_norm);
    }

    // CG iterations
    int iter;
    double residual_norm = b_norm;

    for (iter = 0; iter < config.max_iters; iter++) {
        // alpha = (r,r) / (A*p, p)
        CUDA_CHECK(cudaEventRecord(start_spmv));
        CUDA_CHECK(cudaMemcpy(h_temp_in, d_p, n * sizeof(double), cudaMemcpyDeviceToHost));
        spmv_op->run_timed(h_temp_in, h_temp_out, &kernel_time_spmv);  // h_temp_out = A*p
        CUDA_CHECK(cudaMemcpy(d_Ap, h_temp_out, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop_spmv));
        CUDA_CHECK(cudaEventSynchronize(stop_spmv));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_spmv, stop_spmv));
        time_spmv += ms;

        CUDA_CHECK(cudaEventRecord(start_reduce));
        dot_kernel<<<blocks, threads, threads * sizeof(double)>>>(n, d_Ap, d_p, d_block_results);
        CUDA_CHECK(cudaDeviceSynchronize());
        double pAp = sum_block_results(d_block_results, blocks);
        CUDA_CHECK(cudaEventRecord(stop_reduce));
        CUDA_CHECK(cudaEventSynchronize(stop_reduce));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_reduce, stop_reduce));
        time_reduce += ms;

        double alpha = rr_old / pAp;

        // x = x + alpha*p
        CUDA_CHECK(cudaEventRecord(start_blas));
        axpy_kernel<<<blocks, threads>>>(n, alpha, d_p, d_x);
        CUDA_CHECK(cudaEventRecord(stop_blas));
        CUDA_CHECK(cudaEventSynchronize(stop_blas));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_blas, stop_blas));
        time_blas += ms;

        // r = r - alpha*A*p
        CUDA_CHECK(cudaEventRecord(start_blas));
        axpy_kernel<<<blocks, threads>>>(n, -alpha, d_Ap, d_r);
        CUDA_CHECK(cudaEventRecord(stop_blas));
        CUDA_CHECK(cudaEventSynchronize(stop_blas));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_blas, stop_blas));
        time_blas += ms;

        // (r_new, r_new)
        CUDA_CHECK(cudaEventRecord(start_reduce));
        dot_kernel<<<blocks, threads, threads * sizeof(double)>>>(n, d_r, d_r, d_block_results);
        CUDA_CHECK(cudaDeviceSynchronize());
        double rr_new = sum_block_results(d_block_results, blocks);
        CUDA_CHECK(cudaEventRecord(stop_reduce));
        CUDA_CHECK(cudaEventSynchronize(stop_reduce));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_reduce, stop_reduce));
        time_reduce += ms;

        residual_norm = sqrt(rr_new);
        double rel_residual = residual_norm / b_norm;

        if (config.verbose >= 2) {
            printf("[CG] Iter %3d: residual = %e (rel = %e)\n", iter + 1, residual_norm, rel_residual);
        }

        // Check convergence
        if (rel_residual < config.tolerance) {
            iter++;  // Count this iteration
            break;
        }

        // beta = (r_new, r_new) / (r_old, r_old)
        double beta = rr_new / rr_old;

        // p = r + beta*p
        CUDA_CHECK(cudaEventRecord(start_blas));
        axpby_kernel<<<blocks, threads>>>(n, 1.0, d_r, beta, d_p, d_p);
        CUDA_CHECK(cudaEventRecord(stop_blas));
        CUDA_CHECK(cudaEventSynchronize(stop_blas));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_blas, stop_blas));
        time_blas += ms;

        rr_old = rr_new;
    }

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_total, stop_total));

    // Transfer solution back
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Fill statistics
    stats->iterations = iter;
    stats->residual_norm = residual_norm;
    stats->time_total_ms = total_ms;
    stats->time_spmv_ms = time_spmv;
    stats->time_blas1_ms = time_blas;
    stats->time_reductions_ms = time_reduce;
    stats->converged = (residual_norm / b_norm < config.tolerance) ? 1 : 0;

    if (config.verbose >= 1) {
        printf("[CG] Converged: %s\n", stats->converged ? "YES" : "NO");
        printf("[CG] Iterations: %d\n", stats->iterations);
        printf("[CG] Final residual: %e\n", stats->residual_norm);
        printf("[CG] Time breakdown:\n");
        printf("     Total:      %.3f ms\n", stats->time_total_ms);
        printf("     SpMV:       %.3f ms (%.1f%%)\n", stats->time_spmv_ms,
               100.0 * stats->time_spmv_ms / stats->time_total_ms);
        printf("     BLAS1:      %.3f ms (%.1f%%)\n", stats->time_blas1_ms,
               100.0 * stats->time_blas1_ms / stats->time_total_ms);
        printf("     Reductions: %.3f ms (%.1f%%)\n", stats->time_reductions_ms,
               100.0 * stats->time_reductions_ms / stats->time_total_ms);
    }

    // Cleanup
    free(h_temp_in);
    free(h_temp_out);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_block_results);

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_spmv);
    cudaEventDestroy(stop_spmv);
    cudaEventDestroy(start_blas);
    cudaEventDestroy(stop_blas);
    cudaEventDestroy(start_reduce);
    cudaEventDestroy(stop_reduce);

    return 0;
}

/**
 * @brief Device-native reduction: sums GPU array to single scalar on GPU
 */
__global__ void final_sum_kernel(const double* block_results, int num_blocks, double* result) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = threadIdx.x;

    // Load block results into shared memory
    sdata[tid] = (i < num_blocks) ? block_results[i] : 0.0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < num_blocks) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        *result = sdata[0];
    }
}

/**
 * @brief CG solver device-native implementation (zero host transfers)
 */
int cg_solve_device(SpmvOperator* spmv_op,
                    MatrixData* mat,
                    const double* b,
                    double* x,
                    CGConfig config,
                    CGStats* stats) {

    // Verify device-native support
    if (!spmv_op->run_device) {
        fprintf(stderr, "[ERROR] Operator '%s' does not support device-native interface\n", spmv_op->name);
        return 1;
    }

    int n = mat->rows;

    // Launch configuration
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Allocate device vectors
    double *d_x, *d_b, *d_r, *d_p, *d_Ap, *d_block_results;
    double *d_rr_old, *d_rr_new, *d_pAp;  // Device scalars

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_results, blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rr_old, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rr_new, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pAp, sizeof(double)));

    // Transfer initial data
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_spmv, stop_spmv;
    cudaEvent_t start_blas, stop_blas;
    cudaEvent_t start_reduce, stop_reduce;

    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_spmv));
    CUDA_CHECK(cudaEventCreate(&stop_spmv));
    CUDA_CHECK(cudaEventCreate(&start_blas));
    CUDA_CHECK(cudaEventCreate(&stop_blas));
    CUDA_CHECK(cudaEventCreate(&start_reduce));
    CUDA_CHECK(cudaEventCreate(&stop_reduce));

    double time_spmv = 0.0, time_blas = 0.0, time_reduce = 0.0;
    float ms;

    CUDA_CHECK(cudaEventRecord(start_total));

    // r0 = b - A*x0 (device-native SpMV)
    CUDA_CHECK(cudaEventRecord(start_spmv));
    spmv_op->run_device(d_x, d_Ap);  // d_Ap = A*x0
    CUDA_CHECK(cudaEventRecord(stop_spmv));
    CUDA_CHECK(cudaEventSynchronize(stop_spmv));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_spmv, stop_spmv));
    time_spmv += ms;

    CUDA_CHECK(cudaEventRecord(start_blas));
    axpby_kernel<<<blocks, threads>>>(n, 1.0, d_b, -1.0, d_Ap, d_r);  // r = b - A*x0
    CUDA_CHECK(cudaEventRecord(stop_blas));
    CUDA_CHECK(cudaEventSynchronize(stop_blas));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_blas, stop_blas));
    time_blas += ms;

    // p0 = r0
    copy_kernel<<<blocks, threads>>>(n, d_r, d_p);

    // (r0, r0) - device-native reduction
    CUDA_CHECK(cudaEventRecord(start_reduce));
    dot_kernel<<<blocks, threads, threads * sizeof(double)>>>(n, d_r, d_r, d_block_results);
    final_sum_kernel<<<1, 256, 256 * sizeof(double)>>>(d_block_results, blocks, d_rr_old);
    CUDA_CHECK(cudaEventRecord(stop_reduce));
    CUDA_CHECK(cudaEventSynchronize(stop_reduce));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_reduce, stop_reduce));
    time_reduce += ms;

    // Get initial residual for convergence check (one-time transfer)
    double h_rr_old;
    CUDA_CHECK(cudaMemcpy(&h_rr_old, d_rr_old, sizeof(double), cudaMemcpyDeviceToHost));
    double b_norm = sqrt(h_rr_old);

    if (config.verbose >= 1) {
        printf("[CG-DEVICE] Initial residual: %e\n", b_norm);
    }

    // CG iterations (fully on GPU)
    int iter;
    double residual_norm = b_norm;
    double h_rr_new;

    for (iter = 0; iter < config.max_iters; iter++) {
        // alpha = (r,r) / (A*p, p)
        CUDA_CHECK(cudaEventRecord(start_spmv));
        spmv_op->run_device(d_p, d_Ap);  // d_Ap = A*p (device-native)
        CUDA_CHECK(cudaEventRecord(stop_spmv));
        CUDA_CHECK(cudaEventSynchronize(stop_spmv));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_spmv, stop_spmv));
        time_spmv += ms;

        CUDA_CHECK(cudaEventRecord(start_reduce));
        dot_kernel<<<blocks, threads, threads * sizeof(double)>>>(n, d_Ap, d_p, d_block_results);
        final_sum_kernel<<<1, 256, 256 * sizeof(double)>>>(d_block_results, blocks, d_pAp);
        CUDA_CHECK(cudaEventRecord(stop_reduce));
        CUDA_CHECK(cudaEventSynchronize(stop_reduce));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_reduce, stop_reduce));
        time_reduce += ms;

        // Compute alpha on CPU (one small transfer)
        double h_pAp;
        CUDA_CHECK(cudaMemcpy(&h_pAp, d_pAp, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_rr_old, d_rr_old, sizeof(double), cudaMemcpyDeviceToHost));
        double alpha = h_rr_old / h_pAp;

        // x = x + alpha*p
        CUDA_CHECK(cudaEventRecord(start_blas));
        axpy_kernel<<<blocks, threads>>>(n, alpha, d_p, d_x);
        CUDA_CHECK(cudaEventRecord(stop_blas));
        CUDA_CHECK(cudaEventSynchronize(stop_blas));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_blas, stop_blas));
        time_blas += ms;

        // r = r - alpha*A*p
        CUDA_CHECK(cudaEventRecord(start_blas));
        axpy_kernel<<<blocks, threads>>>(n, -alpha, d_Ap, d_r);
        CUDA_CHECK(cudaEventRecord(stop_blas));
        CUDA_CHECK(cudaEventSynchronize(stop_blas));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_blas, stop_blas));
        time_blas += ms;

        // (r_new, r_new)
        CUDA_CHECK(cudaEventRecord(start_reduce));
        dot_kernel<<<blocks, threads, threads * sizeof(double)>>>(n, d_r, d_r, d_block_results);
        final_sum_kernel<<<1, 256, 256 * sizeof(double)>>>(d_block_results, blocks, d_rr_new);
        CUDA_CHECK(cudaEventRecord(stop_reduce));
        CUDA_CHECK(cudaEventSynchronize(stop_reduce));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_reduce, stop_reduce));
        time_reduce += ms;

        // Check convergence (one small transfer)
        CUDA_CHECK(cudaMemcpy(&h_rr_new, d_rr_new, sizeof(double), cudaMemcpyDeviceToHost));
        residual_norm = sqrt(h_rr_new);
        double rel_residual = residual_norm / b_norm;

        if (config.verbose >= 2) {
            printf("[CG-DEVICE] Iter %3d: residual = %e (rel = %e)\n", iter + 1, residual_norm, rel_residual);
        }

        // Check convergence
        if (rel_residual < config.tolerance) {
            iter++;
            break;
        }

        // beta = (r_new, r_new) / (r_old, r_old)
        double beta = h_rr_new / h_rr_old;

        // p = r + beta*p
        CUDA_CHECK(cudaEventRecord(start_blas));
        axpby_kernel<<<blocks, threads>>>(n, 1.0, d_r, beta, d_p, d_p);
        CUDA_CHECK(cudaEventRecord(stop_blas));
        CUDA_CHECK(cudaEventSynchronize(stop_blas));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_blas, stop_blas));
        time_blas += ms;

        // Update rr_old
        CUDA_CHECK(cudaMemcpy(d_rr_old, d_rr_new, sizeof(double), cudaMemcpyDeviceToDevice));
        h_rr_old = h_rr_new;
    }

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_total, stop_total));

    // Transfer solution back
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Fill statistics
    stats->iterations = iter;
    stats->residual_norm = residual_norm;
    stats->time_total_ms = total_ms;
    stats->time_spmv_ms = time_spmv;
    stats->time_blas1_ms = time_blas;
    stats->time_reductions_ms = time_reduce;
    stats->converged = (residual_norm / b_norm < config.tolerance) ? 1 : 0;

    if (config.verbose >= 1) {
        printf("[CG-DEVICE] Converged: %s\n", stats->converged ? "YES" : "NO");
        printf("[CG-DEVICE] Iterations: %d\n", stats->iterations);
        printf("[CG-DEVICE] Final residual: %e\n", stats->residual_norm);
        printf("[CG-DEVICE] Time breakdown:\n");
        printf("     Total:      %.3f ms\n", stats->time_total_ms);
        printf("     SpMV:       %.3f ms (%.1f%%)\n", stats->time_spmv_ms,
               100.0 * stats->time_spmv_ms / stats->time_total_ms);
        printf("     BLAS1:      %.3f ms (%.1f%%)\n", stats->time_blas1_ms,
               100.0 * stats->time_blas1_ms / stats->time_total_ms);
        printf("     Reductions: %.3f ms (%.1f%%)\n", stats->time_reductions_ms,
               100.0 * stats->time_reductions_ms / stats->time_total_ms);
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_block_results);
    cudaFree(d_rr_old);
    cudaFree(d_rr_new);
    cudaFree(d_pAp);

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_spmv);
    cudaEventDestroy(stop_spmv);
    cudaEventDestroy(start_blas);
    cudaEventDestroy(stop_blas);
    cudaEventDestroy(start_reduce);
    cudaEventDestroy(stop_reduce);

    return 0;
}
