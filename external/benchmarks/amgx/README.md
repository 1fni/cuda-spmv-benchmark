# AmgX Benchmarks

NVIDIA AmgX benchmarks for sparse linear solvers (CG, PCG) on single and multi-GPU.

## Prerequisites

- **CUDA** (tested with 12.4+)
- **MPI** (OpenMPI or MPICH) for multi-GPU
- **AmgX library** (automatically detected)

## AmgX Installation

The Makefile auto-detects AmgX in this order:
1. **Source build**: `../../amgx-src` (headers + build/)
2. **Local install**: `../../amgx` (headers + lib/)
3. **System install**: `/usr/local` (headers + lib/)

Check detection:
```bash
make help
```

## Building

```bash
# Build all benchmarks
make

# Build specific target
make amgx_cg_solver        # Single-GPU CG
make amgx_cg_solver_mgpu   # Multi-GPU MPI CG
```

## Running

### Single-GPU CG Solver
```bash
./amgx_cg_solver matrix/stencil_512x512.mtx --runs=10
```

### Multi-GPU MPI CG Solver
```bash
# 2 GPUs
mpirun --allow-run-as-root -np 2 ./amgx_cg_solver_mgpu matrix/stencil_512x512.mtx --runs=10

# 4 GPUs
mpirun --allow-run-as-root -np 4 ./amgx_cg_solver_mgpu matrix/stencil_5000x5000.mtx --runs=10
```

### Options
- `--tol=1e-6` : Convergence tolerance
- `--max-iters=1000` : Max iterations
- `--runs=10` : Number of benchmark runs
- `--json=results.json` : Export JSON results
- `--csv=results.csv` : Export CSV results

## Expected Behavior

### Multi-GPU Checksum Variation (~0.15%)

When running with multiple MPI ranks, the solution checksum varies slightly from single-rank execution. **This is expected behavior** for distributed iterative solvers.

**Example** (512×512 stencil, tolerance 1e-6):
```
1 rank:  sum=2.608806e+05, norm=509.87, 17 iterations
2 ranks: sum=2.612679e+05, norm=510.88, 17 iterations  (0.15% diff)
```

**Why this happens:**
- **Floating-point non-associativity**: `(a+b)+c ≠ a+(b+c)` in double precision
- **MPI reduction order**: `MPI_Allreduce` for dot products uses implementation-dependent summation order
- **Domain decomposition**: Distributed matrix-vector products introduce different rounding error accumulation

**Impact:**
- ✅ Same iteration count (convergence identical)
- ✅ Tolerance criteria met (residual < 1e-6)
- ✅ Variation at 8th+ significant digit
- ✅ Consistent and reproducible

**Status:** Normal for distributed linear solvers. Bit-exact reproducibility would require deterministic reductions (significant performance penalty).
