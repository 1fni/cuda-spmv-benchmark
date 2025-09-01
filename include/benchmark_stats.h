#ifndef BENCHMARK_STATS_H
#define BENCHMARK_STATS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double median_ms;
    double mean_ms;
    double std_dev_ms;
    double min_ms;
    double max_ms;
    int valid_runs;
    int outliers_removed;
} BenchmarkStats;

// Run multiple iterations with outlier detection
int benchmark_with_stats(int (*run_func)(const double*, double*, double*), 
                        const double* x, double* y, int num_runs,
                        BenchmarkStats* stats);

#ifdef __cplusplus
}
#endif

#endif