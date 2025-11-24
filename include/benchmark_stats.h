#ifndef BENCHMARK_STATS_H
#define BENCHMARK_STATS_H

#include "spmv.h"
#include "io.h"
#include "solvers/cg_solver.h"

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

// SpMV benchmark: Run multiple iterations with outlier detection
int benchmark_with_stats(int (*run_func)(const double*, double*, double*),
                        const double* x, double* y, int num_runs,
                        BenchmarkStats* stats);

// CG single-GPU benchmark wrapper
int cg_benchmark_with_stats_device(
    SpmvOperator* spmv_op,
    MatrixData* mat,
    double* b,
    double* x,
    CGConfig config,
    int num_runs,
    BenchmarkStats* bench_stats,
    CGStats* final_stats);

#ifdef __cplusplus
}
#endif

#endif