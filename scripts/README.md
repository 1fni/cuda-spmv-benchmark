# Benchmark Scripts

Automated benchmarking scripts for SpMV and CG solvers.

## Quick Start

```bash
# Local testing (single-GPU)
./scripts/quick_bench.sh matrix/stencil_512x512.mtx

# Full benchmark suite (8 GPUs A100)
./scripts/benchmark_suite.sh matrix/stencil_5000x5000.mtx results/a100_run
```

## Scripts Overview

### 1. `benchmark_suite.sh` - Complete benchmark suite

Full benchmarking pipeline for multi-GPU systems (designed for A100 8×GPU).

**Usage:**
```bash
./scripts/benchmark_suite.sh <matrix.mtx> [output_dir]
```

**What it runs:**
1. SpMV single-GPU - All modes (csr, stencil5-*)
2. CG single-GPU - Modes: csr, stencil5-csr-direct
3. CG multi-GPU AllGather - 2, 4, 8 GPUs
4. CG multi-GPU Halo P2P - 2, 4, 8 GPUs

### 2. `quick_bench.sh` - Fast local testing

Lightweight benchmark for local development (single-GPU only).

## Output Files

- `*.csv` - Tabular comparison (append mode for multi-run)
- `*.json` - Detailed run metadata (one per config)

## Example

```bash
# Generate matrix
./bin/generate_matrix 5000 matrix/stencil_5000x5000.mtx

# Run benchmark
./scripts/benchmark_suite.sh matrix/stencil_5000x5000.mtx results/a100

# Results in: results/a100/
```
