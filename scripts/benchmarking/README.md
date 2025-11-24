# Benchmarking Scripts

## Recommended Scripts (New - JSON/CSV exports)

### 1. `benchmark_suite.sh` ⭐ **[PRIMARY]**
**Full benchmark suite for A100 8×GPU systems**

```bash
./scripts/benchmarking/benchmark_suite.sh <matrix.mtx> [output_dir]
```

**What it runs:**
- SpMV single-GPU: all modes (csr, stencil5-*)
- CG single-GPU: csr, stencil5-csr-direct (with/without --timers)
- CG multi-GPU AllGather: 2, 4, 8 GPUs
- CG multi-GPU Halo P2P: 2, 4, 8 GPUs

**Output:** Clean JSON + CSV files for analysis

**Example:**
```bash
./bin/generate_matrix 5000 matrix/stencil_5000x5000.mtx
./scripts/benchmarking/benchmark_suite.sh matrix/stencil_5000x5000.mtx results/a100_run
```

---

### 2. `quick_bench.sh` ⭐ **[DEV/TEST]**
**Fast local testing (single-GPU only)**

```bash
./scripts/benchmarking/quick_bench.sh <matrix.mtx> [output_dir]
```

**What it runs:**
- SpMV: csr, stencil5-csr-direct
- CG: csr, stencil5-csr-direct

**Use for:** Quick validation before full benchmark suite

---

## Legacy Scripts (VastAI-specific)

These scripts are **VastAI-optimized** with auto-detection and setup. They remain useful for VastAI instances but are less general.

### `multimode_benchmark.sh` 🔧 [VASTAI]
- Auto-detects GPU and generates optimal matrix size
- Tests all SpMV modes with single matrix load
- VastAI-specific features (colors, emojis, auto-sizing)

**Use when:** Running on VastAI rental GPUs with auto-setup

---

### `benchmark_and_visualize.sh` 🔧 [OBSOLETE?]
- Benchmarking + visualization generation
- May be outdated (check dependencies)

**Status:** Consider replacing with `benchmark_suite.sh` + separate analysis script

---

### `benchmark_optimization.sh` 🔧 [SPECIFIC]
- Compares optimization against baseline
- Niche use case

**Status:** Keep for comparing specific optimizations

---

### `cg_comparison_setup.sh` 🔧 [KOKKOS]
- VastAI setup for CG + Kokkos comparison
- Clones repo, installs Kokkos, builds everything

**Status:** Keep for Kokkos benchmarking (Phase 3.5)

---

### `remote_benchmark.sh` 🔧 [VASTAI]
- VastAI automated benchmark with auto-detection
- Similar to `multimode_benchmark.sh`

**Status:** Redundant with `multimode_benchmark.sh`?

---

## Decision: Which Scripts to Keep?

### ✅ Keep (Active)
1. **benchmark_suite.sh** - Primary production benchmark
2. **quick_bench.sh** - Dev/test
3. **multimode_benchmark.sh** - VastAI auto-setup
4. **cg_comparison_setup.sh** - Kokkos comparison

### ⚠️ Review (Possibly Obsolete)
5. **benchmark_and_visualize.sh** - Check if still used
6. **remote_benchmark.sh** - Redundant with multimode_benchmark?
7. **benchmark_optimization.sh** - Niche use case

---

## Recommended Workflow

### Local Development
```bash
./scripts/benchmarking/quick_bench.sh matrix/test.mtx
```

### Full A100 Benchmark
```bash
./bin/generate_matrix 10000 matrix/stencil_10000x10000.mtx
./scripts/benchmarking/benchmark_suite.sh matrix/stencil_10000x10000.mtx results/a100
```

### VastAI Quick Test
```bash
./scripts/benchmarking/multimode_benchmark.sh  # Auto-detects and runs
```

---

## Output Structure

**benchmark_suite.sh** produces:
```
results/a100_run/
├── spmv_<matrix>_<mode>.csv/.json     # SpMV individual
├── cg_<matrix>_single.csv             # CG single-GPU comparison
├── cg_<matrix>_mgpu_allgather.csv     # Multi-GPU AllGather scaling
└── cg_<matrix>_mgpu_halo.csv          # Multi-GPU Halo P2P scaling
```

**CSV:** Tabular comparison (Excel/pandas-ready)
**JSON:** Full metadata (GPU specs, timing breakdown, convergence)
