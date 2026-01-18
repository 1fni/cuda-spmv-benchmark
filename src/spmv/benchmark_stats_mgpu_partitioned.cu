#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "benchmark_stats.h"
#include "solvers/cg_solver_mgpu_partitioned.h"

extern "C" {

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double calculate_median(double* times, int count) {
    qsort(times, count, sizeof(double), compare_doubles);
    if (count % 2 == 0) {
        return (times[count / 2 - 1] + times[count / 2]) / 2.0;
    }
    return times[count / 2];
}

static double calculate_mean(double* times, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += times[i];
    }
    return sum / count;
}

static double calculate_std_dev(double* times, int count, double mean) {
    double sum_sq_diff = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / count);
}

// CG multi-GPU partitioned benchmark with statistics
int cg_benchmark_with_stats_mgpu_partitioned(SpmvOperator* spmv_op, MatrixData* mat, double* b,
                                             double* x, CGConfigMultiGPU config, int num_runs,
                                             BenchmarkStats* bench_stats,
                                             CGStatsMultiGPU* final_stats) {

    double* times = (double*)malloc(num_runs * sizeof(double));
    if (!times)
        return -1;

    // Store all stats to pick median run later
    CGStatsMultiGPU* all_stats = (CGStatsMultiGPU*)malloc(num_runs * sizeof(CGStatsMultiGPU));
    if (!all_stats) {
        free(times);
        return -1;
    }

    // Backup for reset
    double* x_backup = (double*)malloc(mat->rows * sizeof(double));
    if (!x_backup) {
        free(times);
        free(all_stats);
        return -1;
    }
    memcpy(x_backup, x, mat->rows * sizeof(double));

    // Run benchmarks (silent for clean output)
    int valid_count = 0;
    int original_verbose = config.verbose;
    config.verbose = 0;  // Suppress iteration output during benchmark

    for (int i = 0; i < num_runs; i++) {
        // Reset solution vector
        memcpy(x, x_backup, mat->rows * sizeof(double));

        // Run CG solver
        if (cg_solve_mgpu_partitioned(spmv_op, mat, b, x, config, &all_stats[valid_count]) == 0) {
            times[valid_count] = all_stats[valid_count].time_total_ms;
            valid_count++;
        }
    }

    config.verbose = original_verbose;  // Restore original verbosity

    free(x_backup);

    if (valid_count < 3) {
        free(times);
        free(all_stats);
        return -1;
    }

    // Calculate stats
    double mean = calculate_mean(times, valid_count);
    double std_dev = calculate_std_dev(times, valid_count, mean);

    // Remove outliers and track indices
    double* filtered_times = (double*)malloc(valid_count * sizeof(double));
    int* filtered_indices = (int*)malloc(valid_count * sizeof(int));
    int filtered_count = 0;

    for (int i = 0; i < valid_count; i++) {
        if (fabs(times[i] - mean) <= 2.0 * std_dev) {
            filtered_times[filtered_count] = times[i];
            filtered_indices[filtered_count] = i;
            filtered_count++;
        }
    }

    bench_stats->mean_ms = calculate_mean(filtered_times, filtered_count);
    bench_stats->std_dev_ms =
        calculate_std_dev(filtered_times, filtered_count, bench_stats->mean_ms);
    bench_stats->median_ms = calculate_median(filtered_times, filtered_count);
    bench_stats->min_ms = filtered_times[0];
    bench_stats->max_ms = filtered_times[filtered_count - 1];
    bench_stats->valid_runs = filtered_count;
    bench_stats->outliers_removed = valid_count - filtered_count;

    // Find which run produced the median time
    int median_run_idx = filtered_indices[filtered_count / 2];
    *final_stats = all_stats[median_run_idx];

    free(times);
    free(all_stats);
    free(filtered_times);
    free(filtered_indices);
    return 0;
}

}  // extern "C"
