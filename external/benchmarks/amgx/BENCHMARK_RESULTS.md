# AmgX Multi-GPU CG Solver - Benchmark Results

NVIDIA AmgX reference implementation for Conjugate Gradient solver with multi-GPU scaling validation.

**Hardware**: 8× NVIDIA A100-SXM4-80GB (NVLink)
**Date**: January 14, 2026
**Solver**: PCG without preconditioner (equivalent to CG)
**Tolerance**: 1e-6
**Benchmark runs**: 10 (outlier removal, median reported)

**Code Version**:
- Git commit: `fd66760` (Switch AmgX to upload_all_global API with int64_t)
- Branch: `feature/amgx-distributed-api` (merged to main)
- API: `AMGX_matrix_upload_all_global` with `int64_t*` column indices, `nrings=2`
- Date: January 13-14, 2026

> **Note**: Results below generated on 8× A100-SXM4-80GB (NVLink) with this version. Uses distributed API with automatic halo detection.

## Performance Summary

### 10000×10000 Stencil (100M unknowns, 500M nnz)

| GPUs | Time (ms) | GFLOPS | Speedup | Efficiency |
|------|-----------|--------|---------|------------|
| 1    | 188.67    | 74.2   | 1.00×   | 100%       |
| 2    | 99.04     | 141.3  | 1.90×   | 95%        |
| 4    | 50.32     | 278.2  | 3.75×   | 94%        |
| 8    | 26.99     | 518.6  | 6.99×   | 87%        |

**Iterations**: 14 (converged)

### 15000×15000 Stencil (225M unknowns, 1.125B nnz)

| GPUs | Time (ms) | GFLOPS | Speedup | Efficiency |
|------|-----------|--------|---------|------------|
| 1    | 420.01    | 75.0   | 1.00×   | 100%       |
| 2    | 216.41    | 145.6  | 1.94×   | 97%        |
| 4    | 111.25    | 283.1  | 3.78×   | 94%        |
| 8    | 57.03     | 552.3  | 7.36×   | 92%        |

**Iterations**: 14 (converged)

### 20000×20000 Stencil (400M unknowns, 2B nnz)

| GPUs | Time (ms) | GFLOPS | Speedup | Efficiency |
|------|-----------|--------|---------|------------|
| 1    | 746.74    | 75.0   | 1.00×   | 100%       |
| 2    | 382.48    | 146.4  | 1.95×   | 98%        |
| 4    | 194.77    | 287.5  | 3.83×   | 96%        |
| 8    | 102.33    | 547.2  | 7.30×   | 91%        |

**Iterations**: 14 (converged)

## Key Observations

1. **Consistent Scaling**: 7.0-7.4× speedup across all problem sizes (87-92% efficiency on 8 GPUs)
2. **Size Independence**: Performance scales equally well from 100M to 400M unknowns
3. **Strong GFLOPS**: 518-552 GFLOPS on 8 GPUs for stencil operations
4. **Stable Convergence**: 14 iterations regardless of GPU count (deterministic algorithm)

## Implementation Details

**API Used**: `AMGX_matrix_upload_all()` with MPI auto-detection
**Matrix Format**: CSR with global column indices (int)
**Communication**: Automatic halo detection via MPI communicator
**Partitioning**: 1D row-band decomposition (equal distribution)

## Preconditioner Status

| Preconditioner | Single-GPU | Multi-GPU | Status |
|----------------|------------|-----------|--------|
| None (CG)      | ✅ Working | ✅ Working | Production |
| BLOCK_JACOBI   | ⚠️ Untested | ⚠️ Untested | Pending |
| AMG            | ✅ Working | ❌ Failed | Memory errors |

**AMG Issue**: Illegal memory access in `dense_lu_solver.cu` during coarse grid solve. Requires explicit communication maps (`AMGX_matrix_comm_from_maps_one_ring`) for multi-GPU AMG support.

## Comparison with Custom CG Solver

*Note: Custom CG multi-GPU results to be added for direct comparison*

Expected custom CG performance based on prior benchmarks:
- **Single-GPU**: ~190ms (similar to AmgX)
- **8 GPUs**: ~25ms (7.48× speedup, slightly faster than AmgX)

**Custom advantages**:
- Explicit halo exchange control (P2P or MPI staging)
- Lower communication overhead for regular stencil patterns
- Full control over reduction operations

**AmgX advantages**:
- Industry-validated reference implementation
- Extensive preconditioner support (single-GPU)
- Robust convergence and error handling

## Data Files

Raw results available in:
- `results_archive/amgx_result_20260114_cg.tar.gz`
- JSON format: per-run timing, statistics, GFLOPS
- CSV format: tabular export for analysis

## Command Reference

```bash
# 10000×10000 benchmark (4 GPUs)
mpirun -np 4 ./external/benchmarks/amgx/amgx_cg_solver_mgpu matrix/10000 \
    --precond=none --tol=1e-6 --max-iters=1000 --runs=10

# 15000×15000 benchmark (8 GPUs)
mpirun -np 8 ./external/benchmarks/amgx/amgx_cg_solver_mgpu matrix/15000 \
    --precond=none --tol=1e-6 --max-iters=1000 --runs=10

# 20000×20000 benchmark (8 GPUs)
mpirun -np 8 ./external/benchmarks/amgx/amgx_cg_solver_mgpu matrix/20000 \
    --precond=none --tol=1e-6 --max-iters=1000 --runs=10
```

## Conclusion

AmgX demonstrates excellent multi-GPU scaling (7.0-7.4× on 8 GPUs) with consistent performance across problem sizes from 100M to 400M unknowns. The implementation provides a production-grade reference for validating custom CG solver performance.

**Next Steps**:
- Add custom CG multi-GPU comparison data
- Resolve AMG multi-GPU communication maps issue
- Benchmark BLOCK_JACOBI preconditioner (single and multi-GPU)
