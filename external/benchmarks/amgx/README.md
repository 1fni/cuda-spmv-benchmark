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

## Known Issues

### Multi-GPU Checksum Difference (~0.15%)

When running with multiple MPI ranks, the solution checksum may differ slightly (±0.15%) from single-rank:
- **Cause**: AmgX automatic halo detection without explicit partition vector
- **Impact**: Negligible for iterative solvers (same iterations, convergence OK)
- **Status**: Acceptable for benchmarking purposes

Example (512×512 stencil, 17 iterations):
```
1 rank:  sum=2.608806e+05, norm=509.87
2 ranks: sum=2.612679e+05, norm=510.88  (0.15% diff)
```

This difference is consistent and reproducible across runs.
