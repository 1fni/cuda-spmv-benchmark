#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "benchmark_stats.h"

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double calculate_median(double* times, int count) {
    qsort(times, count, sizeof(double), compare_doubles);
    if (count % 2 == 0) {
        return (times[count/2 - 1] + times[count/2]) / 2.0;
    }
    return times[count/2];
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

int benchmark_with_stats(int (*run_func)(const double*, double*, double*), 
                        const double* x, double* y, int num_runs,
                        BenchmarkStats* stats) {
    
    double* times = (double*)malloc(num_runs * sizeof(double));
    if (!times) return -1;
    
    // Run benchmarks
    int valid_count = 0;
    for (int i = 0; i < num_runs; i++) {
        double time_ms;
        if (run_func(x, y, &time_ms) == 0) {
            times[valid_count++] = time_ms;
        }
    }
    
    if (valid_count < 3) {
        free(times);
        return -1; // Need at least 3 valid runs
    }
    
    // Calculate initial stats
    double mean = calculate_mean(times, valid_count);
    double std_dev = calculate_std_dev(times, valid_count, mean);
    
    // Remove outliers (>2 std devs from mean)
    double* filtered_times = (double*)malloc(valid_count * sizeof(double));
    int filtered_count = 0;
    
    for (int i = 0; i < valid_count; i++) {
        if (fabs(times[i] - mean) <= 2.0 * std_dev) {
            filtered_times[filtered_count++] = times[i];
        }
    }
    
    // Recalculate stats without outliers
    stats->mean_ms = calculate_mean(filtered_times, filtered_count);
    stats->std_dev_ms = calculate_std_dev(filtered_times, filtered_count, stats->mean_ms);
    stats->median_ms = calculate_median(filtered_times, filtered_count);
    
    // Find min/max from filtered data
    stats->min_ms = filtered_times[0];  // Already sorted by median calculation
    stats->max_ms = filtered_times[filtered_count - 1];
    
    stats->valid_runs = filtered_count;
    stats->outliers_removed = valid_count - filtered_count;
    
    free(times);
    free(filtered_times);
    return 0;
}