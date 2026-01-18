#ifndef BENCHMARK_STATS_MGPU_H
#define BENCHMARK_STATS_MGPU_H

#include "benchmark_stats.h"
#include "solvers/cg_solver_mgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// CG multi-GPU partitioned benchmark wrapper (MPI halo exchange)
int cg_benchmark_with_stats_mgpu_partitioned(SpmvOperator* spmv_op, MatrixData* mat, double* b,
                                             double* x, CGConfigMultiGPU config, int num_runs,
                                             BenchmarkStats* bench_stats,
                                             CGStatsMultiGPU* final_stats);

#ifdef __cplusplus
}
#endif

#endif
