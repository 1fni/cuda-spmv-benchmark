/**
 * @file cg_solver.h
 * @brief Conjugate Gradient solver interface
 *
 * @details
 * Classic CG algorithm with timing breakdown for performance analysis.
 * Supports both preconditioned and unpreconditioned variants.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-10-14
 */

#ifndef CG_SOLVER_H
#define CG_SOLVER_H

#include "spmv.h"

/**
 * @brief CG solver configuration
 */
typedef struct {
    int max_iters;              ///< Maximum iterations
    double tolerance;           ///< Convergence tolerance (relative residual)
    int verbose;                ///< Print iteration info (0=silent, 1=summary, 2=per-iter)
    int enable_detailed_timers; ///< Enable per-category timing (adds ~50-100ms sync overhead)
} CGConfig;

/**
 * @brief CG solver statistics and timing breakdown
 */
typedef struct {
    int iterations;           ///< Actual iterations performed
    double residual_norm;     ///< Final residual norm
    double time_total_ms;     ///< Total solve time
    double time_spmv_ms;      ///< SpMV time (accumulated)
    double time_blas1_ms;     ///< BLAS1 ops time (axpy, etc.)
    double time_reductions_ms;///< Dot products and norms time
    int converged;            ///< 1 if converged, 0 otherwise
} CGStats;

/**
 * @brief Solve linear system Ax=b using Conjugate Gradient (host interface)
 *
 * Uses SpMV operators with host pointers (includes GPU↔CPU transfers per iteration).
 * MPI-compatible interface suitable for distributed computing.
 *
 * @param spmv_op SpMV operator for matrix A
 * @param mat Matrix data (for dimensions)
 * @param b Right-hand side vector
 * @param x Solution vector (input: initial guess, output: solution)
 * @param config CG configuration
 * @param stats Output statistics
 * @return 0 on success, non-zero on error
 */
int cg_solve(SpmvOperator* spmv_op,
             MatrixData* mat,
             const double* b,
             double* x,
             CGConfig config,
             CGStats* stats);

/**
 * @brief Solve linear system Ax=b using Conjugate Gradient (device-native interface)
 *
 * GPU-native implementation with zero host transfers during iterations.
 * Requires SpMV operator with run_device support. Optimal for single-GPU workloads.
 *
 * @param spmv_op SpMV operator for matrix A (must have run_device != NULL)
 * @param mat Matrix data (for dimensions)
 * @param b Right-hand side vector
 * @param x Solution vector (input: initial guess, output: solution)
 * @param config CG configuration
 * @param stats Output statistics
 * @return 0 on success, non-zero on error
 */
int cg_solve_device(SpmvOperator* spmv_op,
                    MatrixData* mat,
                    const double* b,
                    double* x,
                    CGConfig config,
                    CGStats* stats);

#endif // CG_SOLVER_H
