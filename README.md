# Multi-GPU Conjugate Gradient Solver

[![CI](https://github.com/1fni/cuda-spmv-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/1fni/cuda-spmv-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance multi-GPU Conjugate Gradient solver for large-scale sparse linear systems using CUDA and MPI. Optimized for structured stencil grids with excellent strong scaling efficiency.

## TL;DR — Key Numbers

| Metric | Result |
|--------|--------|
| **Stencil CG vs NVIDIA AmgX** | 1.40× faster (single-GPU), 1.44× faster (8 GPUs) |
| **Stencil SpMV vs cuSPARSE CSR** | 2.07× speedup on A100 80GB |
| **Strong scaling efficiency** | 87–94% from 1→8 GPUs |
| **Problem size tested** | Up to 400M unknowns (20k×20k stencil) |

**Hardware**: 8× NVIDIA A100-SXM4-80GB · CUDA 12.8 · Driver 575.57

---

## Performance Summary

Exploiting stencil structure enables consistent performance gains over generic sparse solvers, from single-GPU to multi-GPU.

| Configuration        | Custom Stencil CG | NVIDIA AmgX CG | Speedup   |
|----------------------|------------------:|---------------:|----------:|
| Single-GPU (20k×20k) |          531.4 ms |       746.7 ms | **1.40×** |
| 8 GPUs (20k×20k)     |           71.0 ms |       102.3 ms | **1.44×** |

<p align="center">
  <img src="docs/figures/performance_summary_horizontal.png" alt="Performance Summary: All Gains" width="100%">
</p>

- **SpMV kernel**: 2.07× faster than cuSPARSE CSR (single-GPU)
- **CG solver**: 1.40× faster than NVIDIA AmgX (single-GPU, same convergence)
- **Multi-GPU CG**: 1.44× faster than NVIDIA AmgX (8 GPUs, equivalent scaling)

**Key insight**: Generic solvers cannot exploit known stencil structure for memory access and communication minimization, leading to systematic overhead even when scaling efficiently.

---

## Performance Highlights

**Multi-GPU Strong Scaling** on 8× NVIDIA A100-SXM4-80GB

| Problem Size | 1 GPU | 8 GPUs | Speedup | Efficiency |
|--------------|-------|--------|---------|------------|
| **100M unknowns** (10k×10k stencil) | 133.9 ms | 19.3 ms | 6.94× | 86.8% |
| **225M unknowns** (15k×15k stencil) | 300.1 ms | 40.4 ms | 7.43× | 92.9% |
| **400M unknowns** (20k×20k stencil) | 531.4 ms | 71.0 ms | **7.48×** | **93.5%** |

**Key Results:**
- **7.48× speedup** on 400M unknowns with 8 GPUs (93.5% parallel efficiency)
- **Near-linear 2-GPU scaling**: 1.95-1.97× speedup (97-99% efficiency)
- **Deterministic convergence**: All configurations converge in exactly 14 iterations
- **Better scaling with larger problems**: Efficiency improves from 86.8% to 93.5%

### Strong Scaling Visualization

<p align="center">
  <img src="docs/figures/scaling_main_a100.png" alt="Multi-GPU Strong Scaling" width="100%">
</p>

<details>
<summary><b>📊 Detailed Analysis</b></summary>

<p align="center">
  <img src="docs/figures/scaling_detailed_a100.png" alt="Detailed Scaling Analysis" width="100%">
</p>

**Performance breakdown:**
- SpMV kernel time scales near-linearly with GPU count
- MPI communication overhead remains < 10% at 8 GPUs
- Reductions (cuBLAS dot products) maintain high efficiency
- Larger problems amortize communication cost more effectively

</details>

<details>
<summary><b>📈 Problem Size Scaling</b></summary>

<p align="center">
  <img src="docs/figures/problem_size_scaling_overview.png" alt="Problem Size Scaling" width="100%">
</p>

**Observations:**
- 20k×20k matrix achieves best parallel efficiency (93.5%)
- Communication overhead decreases relative to computation for larger problems
- All problem sizes maintain > 85% efficiency at 8 GPUs

See [detailed problem size analysis](docs/PROBLEM_SIZE_SCALING_RESULTS.md) for complete results.

</details>

### Single-GPU SpMV Performance

**Format Comparison** on NVIDIA A100 80GB PCIe

<p align="center">
  <img src="docs/figures/spmv_format_comparison_a100.png" alt="SpMV Format Comparison" width="100%">
</p>

| Matrix Size | CSR (cuSPARSE) | STENCIL5 (Custom) | Speedup | Bandwidth Improvement |
|-------------|----------------|-------------------|---------|----------------------|
| **10k×10k** (100M unknowns) | 6.77 ms | 3.25 ms | **2.08×** | 1.98× (1182 → 2339 GB/s) |
| **15k×15k** (225M unknowns) | 15.00 ms | 7.29 ms | **2.06×** | 1.96× (1200 → 2346 GB/s) |
| **20k×20k** (400M unknowns) | 26.77 ms | 12.86 ms | **2.08×** | 1.98× (1195 → 2364 GB/s) |

**Key Results:**
- **2.07× average speedup** over cuSPARSE CSR implementation
- **~310 GFLOPS sustained** across all problem sizes (STENCIL5)
- **1.97× bandwidth improvement** through optimized memory access patterns
- **Consistent performance scaling**: Speedup stable at 2.06-2.08× from 100M to 400M unknowns

<details>
<summary><b>📊 Detailed Format Analysis</b></summary>

**Optimization techniques:**
- **Grouped memory accesses**: W-C-E (stride-1) before N-S (stride grid_size) for cache efficiency
- **ELLPACK-based storage**: Exploit stencil structure to eliminate col_idx indirection
- **Interior point fast path**: Direct calculation for 95% of rows (no CSR traversal)
- **Boundary fallback**: Standard CSR traversal for edge cases

**Why STENCIL5 is faster:**
1. Predictable access pattern → better L1/L2 cache utilization
2. Reduced memory traffic (no column index lookups for interior points)
3. Coalesced memory accesses for contiguous elements

</details>

### Comparison with NVIDIA AmgX

AmgX is NVIDIA's production-grade multi-GPU solver library, used here as reference implementation.

**Hardware**: 8× NVIDIA A100-SXM4-80GB · CUDA 12.8 · Driver 575.57 (same configuration for both solvers)

<p align="center">
  <img src="docs/figures/custom_vs_amgx_overview.png" alt="Custom CG vs NVIDIA AmgX Comparison" width="100%">
</p>

| Matrix Size     | Implementation  |    1 GPU |   8 GPUs | Speedup | Efficiency |
|-----------------|-----------------|----------|----------|---------|------------|
| **10k×10k**     | Custom CG       | 133.9 ms |  19.3 ms |   6.94× |      86.8% |
| (100M unknowns) | NVIDIA AmgX     | 188.7 ms |  27.0 ms |   6.99× |      87.4% |
|                 |                 |          |          |         |            |
| **15k×15k**     | Custom CG       | 300.1 ms |  40.4 ms |   7.43× |      92.9% |
| (225M unknowns) | NVIDIA AmgX     | 420.0 ms |  57.0 ms |   7.36× |      92.0% |
|                 |                 |          |          |         |            |
| **20k×20k**     | Custom CG       | 531.4 ms |  71.0 ms |   7.48× |      93.5% |
| (400M unknowns) | NVIDIA AmgX     | 746.7 ms | 102.3 ms |   7.30× |      91.3% |

**Key Findings:**
- **~40% faster at every scale**: Custom CG outperforms AmgX on both single-GPU and 8 GPUs
- **Same convergence**: Both solvers converge in 14 iterations with identical tolerance
- **Similar scaling efficiency**: 87-94% for both implementations

**Why the performance difference?**

AmgX operates on generic CSR data structures with no knowledge of the underlying problem structure. The custom solver leverages compile-time knowledge of the 5-point stencil to:
- Reduce memory indirections (no col_idx lookups for interior points)
- Minimize halo exchange volume (160 KB vs 800 MB with AllGather)
- Optimize memory access patterns (grouped W-C-E then N-S)

This is not a limitation of AmgX—it correctly handles arbitrary sparse matrices. The performance gap reflects the benefit of specialization when problem structure is known.

See [`external/benchmarks/amgx/BENCHMARK_RESULTS.md`](external/benchmarks/amgx/BENCHMARK_RESULTS.md) for detailed AmgX methodology and results.

---

## Methodology

**How results were measured:**

| Parameter | Value |
|-----------|-------|
| Runs per configuration | 10 (median reported) |
| Warmup runs | 3 (discarded) |
| Timing scope | Solver only (excludes I/O, matrix setup) |
| Convergence criterion | Relative residual < 1e-6 |

**Compilation flags** (release build):
```
nvcc -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true -std=c++11
```

**Reproducibility:**
```bash
# Quick verification (< 2 min, small matrices)
./scripts/benchmarking/quick_verification.sh

# Full benchmark suite (single-GPU + multi-GPU + AmgX comparison)
./scripts/benchmarking/run_all_benchmarks.sh

# Individual benchmarks
./scripts/benchmarking/benchmark_problem_sizes.sh      # Multi-GPU scaling
./scripts/benchmarking/benchmark_single_gpu_formats.sh # SpMV comparison
./scripts/benchmarking/benchmark_amgx.sh               # AmgX comparison

# Results include JSON/CSV with environment info (CUDA version, driver, etc.)
```

---

## Technical Highlights

### Multi-GPU Architecture
- **MPI explicit staging**: D2H → MPI_Isend/Irecv → H2D for low-latency halo exchange
- **Row-band partitioning**: 1D decomposition with CSR format and halo zone exchange
- **Efficient reductions**: cuBLAS dot products instead of atomics (238× faster)
- **Optimized for A100**: Takes advantage of NVLink/PCIe Gen4 bandwidth

### Algorithm Features
- **Conjugate Gradient (CG)**: Iterative Krylov method for symmetric positive definite systems
- **5-point stencil**: Custom CUDA kernels for finite difference discretizations
- **Halo exchange**: Minimal communication (160 KB per exchange for 10k grid)
- **Convergence criterion**: Relative residual < 1e-6

### Performance Engineering
- **Compared NCCL vs MPI**: MPI staging 43% faster for small repeated messages
- **Profiling-driven**: Nsight Systems analysis to identify bottlenecks
- **Numerical stability**: Deterministic results across all GPU counts
- **Fair benchmarking**: Unified compilation flags (-O2) and consistent test methodology

---

## Quick Start

### Prerequisites
- **Hardware**: NVIDIA GPUs with Compute Capability ≥ 7.0
- **Software**: CUDA Toolkit ≥ 11.0, OpenMPI or MPICH, C++14 compiler

### Build and Run

```bash
# Clone repository
git clone https://github.com/1fni/cuda-spmv-benchmark.git
cd cuda-spmv-benchmark

# Build multi-GPU CG solver
make cg_solver_mgpu_stencil

# Generate 10k×10k 5-point stencil matrix
./bin/generate_matrix 10000 matrix/stencil_10k.mtx

# Run on 2 GPUs
mpirun -np 2 ./bin/cg_solver_mgpu_stencil matrix/stencil_10k.mtx

# Run on 8 GPUs with detailed output
mpirun -np 8 ./bin/cg_solver_mgpu_stencil matrix/stencil_10k.mtx --verbose=2
```

### Benchmark Suite

```bash
# Run problem size scaling benchmark (10k, 15k, 20k on 1/2/4/8 GPUs)
./scripts/benchmarking/benchmark_problem_sizes.sh

# Analyze results and generate plots
cd results_problem_size_scaling_*/
python3 analyze_scaling.py
```

---

## Architecture Overview

### Communication Pattern

```
Row-band partitioning (8 GPUs, 10k×10k grid):

GPU 0: rows [0, 12.5k)       ┐
GPU 1: rows [12.5k, 25k)     │
GPU 2: rows [25k, 37.5k)     │  Halo exchange:
GPU 3: rows [37.5k, 50k)     │  - 160 KB per GPU
GPU 4: rows [50k, 62.5k)     │  - MPI_Isend/Irecv
GPU 5: rows [62.5k, 75k)     │  - ~2 ms latency
GPU 6: rows [75k, 87.5k)     │
GPU 7: rows [87.5k, 100k)    ┘
```

### CG Algorithm Structure

```c
1. Initial setup:
   - Partition matrix rows across GPUs
   - Exchange halo zones for initial vectors (x, r)

2. CG iteration loop (14 iterations):
   a. SpMV: y = A×p (with halo exchange)
   b. Dot products: α = (r,r)/(p,y)  [MPI_Allreduce]
   c. AXPY updates: x += α×p, r -= α×y
   d. Dot products: β = (r_new,r_new)/(r_old,r_old)  [MPI_Allreduce]
   e. Vector update: p = r + β×p
   f. Convergence check: ||r||/||b|| < 1e-6

3. Gather final solution to all ranks
```

**Performance characteristics:**
- **SpMV dominates** (25-30% of total time)
- **BLAS1 operations** (AXPY, dot products): 30-35%
- **Reductions** (MPI_Allreduce): 10-12%
- **Halo exchange**: < 5% for large problems

---

## Repository Structure

```
├── README.md                       # This file
├── docs/                           # Detailed documentation
│   ├── figures/                    # Performance plots (PNG)
│   ├── SHOWCASE_SCALING_RESULTS.md # Strong scaling details
│   ├── PROBLEM_SIZE_SCALING_RESULTS.md # Multi-size analysis
│   └── scaling_summary.md          # Technical summary
├── src/                            # Source code
│   ├── main/                       # Entry points (main functions)
│   │   └── cg_solver_mgpu_stencil.cu   # Multi-GPU CG entry point
│   ├── solvers/                    # Solver implementations
│   │   ├── cg_solver_mgpu_partitioned.cu  # CG algorithm with halo exchange
│   │   └── benchmark_stats_mgpu_partitioned.cu  # Timing and metrics
│   ├── spmv/                       # SpMV kernels
│   │   └── spmv_stencil_csr_direct.cu  # Stencil-optimized SpMV
│   └── io/                         # Matrix I/O
├── include/                        # Header files
├── scripts/                        # Utility scripts
│   ├── benchmarking/               # Benchmark automation
│   ├── profiling/                  # Nsight Systems/Compute
│   └── analysis/                   # Result visualization
└── tests/                          # Unit tests (Google Test)
```

---

## Documentation

- **[Strong Scaling Results](docs/SHOWCASE_SCALING_RESULTS.md)**: Detailed analysis of 15k×15k scaling
- **[Problem Size Scaling](docs/PROBLEM_SIZE_SCALING_RESULTS.md)**: Multi-size benchmark results
- **[Performance Summary](docs/scaling_summary.md)**: Technical metrics and talking points
- **[API Documentation](docs/API.md)**: Code structure and operator interface

---

## Benchmarking

### Automated Benchmarking

```bash
# Problem size scaling (10k, 15k, 20k on 1/2/4/8 GPUs)
./scripts/benchmarking/benchmark_problem_sizes.sh

# Results saved to: results_problem_size_scaling_*/
# - JSON files per configuration
# - CSV files for spreadsheet analysis
# - analyze_scaling.py for visualization
```

### Custom Benchmarks

```bash
# Single configuration with JSON export
mpirun -np 4 ./bin/cg_solver_mgpu_stencil matrix/stencil_15k.mtx \
  --json=results.json --csv=results.csv

# Extract timing from JSON
jq '.timing.median_ms' results.json
```

### Profiling with Nsight Systems

```bash
# Profile 4-rank run
nsys profile --trace=cuda,mpi,nvtx \
  mpirun -np 4 ./bin/cg_solver_mgpu_stencil matrix/stencil_10k.mtx

# View timeline in GUI
nsys-ui report.nsys-rep
```

---

## Performance Comparison

### MPI vs NCCL for Halo Exchange

| Implementation | 8 GPUs (15k×15k) | Notes |
|----------------|------------------|-------|
| **MPI explicit staging** | 40.4 ms | ✅ Production (main branch) |
| NCCL P2P | 67.4 ms | ❌ 4.5ms launch latency per call |
| CUDA IPC direct | 70.1 ms | ❌ Similar overhead to NCCL |

**Conclusion**: MPI with explicit staging (D2H → MPI_Isend/Irecv → H2D) is 43% faster than NCCL for small repeated messages (160 KB halo zones).

See [profiling notes](.notes/OVERLAP_NUMERICAL_STABILITY_SIZE.md) for detailed analysis.

---

## Development

### Build System

**Dual build approach** for flexibility:
- **Makefile**: Primary build for CUDA/MPI binaries
- **CMake**: Testing framework with Google Test

```bash
# Release build (default)
make

# Debug build with GPU debugging (-g -G)
make BUILD_TYPE=debug

# Build specific targets
make cg_solver_mgpu_stencil
make generate_matrix

# Run tests
cd tests && mkdir build && cd build
cmake .. && make && ./test_runner
```

### Adding Features

1. **New SpMV kernel**: Implement in `src/spmv/`, register in `get_operator()`
2. **New solver**: Add to `src/solvers/`, create entry point in `src/main/`
3. **Performance metrics**: Extend `benchmark_stats_mgpu_partitioned.cu`

### Testing

```bash
# All tests
./test_runner

# Specific test suite
./test_runner --gtest_filter="PartitionedSolver*"
```

---

## Requirements

- **NVIDIA GPUs**: Compute Capability ≥ 7.0 (Volta, Turing, Ampere, Hopper)
- **CUDA Toolkit**: ≥ 11.0 with cuSPARSE and cuBLAS libraries
- **MPI Implementation**: OpenMPI ≥ 4.0 or MPICH ≥ 3.3
- **C++ Compiler**: Supporting C++14 (nvcc, g++, clang++)
- **Optional**: Nsight Systems/Compute for profiling

**Tested configurations:**
- NVIDIA A100-SXM4-80GB (8 GPUs) - Primary development
- NVIDIA RTX 3090 (2 GPUs) - Validation
- NVIDIA H100 NVL (single GPU) - Compatibility

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mgpu_cg_solver,
  author = {Bouhrour, Stephane},
  title = {Multi-GPU Conjugate Gradient Solver with MPI},
  year = {2026},
  url = {https://github.com/1fni/cuda-spmv-benchmark},
  note = {7.48× speedup on 400M unknowns with 8 A100 GPUs}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Contact

**Stephane Bouhrour**
Email: bouhrour.stephane@gmail.com
GitHub: [@1fni](https://github.com/1fni)

For questions, issues, or collaboration opportunities, please open an issue on GitHub.
