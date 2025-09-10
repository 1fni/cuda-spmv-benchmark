# Vast.ai Quickstart Guide

## Setup Commands (copy-paste)

```bash
# 1. Setup instance
curl -s https://raw.githubusercontent.com/1fni/cuda-spmv-benchmark/main/scripts/vastai_setup.sh | bash

# 2. Navigate and update
cd cuda-spmv-benchmark
git pull && make

# 3. GPU detection and optimization
./scripts/detect_gpu_config.sh
source /tmp/gpu_config.env

# 4. Run benchmarks (auto-generates GPU memory-optimized matrix)
./scripts/benchmark_and_visualize.sh matrix/stencil_gpu_maxmem.mtx
```

## What Happens

1. **Setup** : Installs CUDA tools, clones repo, builds binaries
2. **Detection** : Detects GPU specs, calculates optimal matrix size
3. **Benchmark** : Auto-generates matrix, runs CSR vs STENCIL comparison
4. **Results** : JSON with full GPU specs + performance charts

## Expected Runtime

- **A100-40GB** : ~3-5 minutes total
- **H100-80GB** : ~2-4 minutes total  
- **RTX 4090** : ~5-8 minutes total

## Output Files

```
results/
├── vastai_YYYYMMDD_HHMM.csv           # Performance comparison
├── vastai_YYYYMMDD_HHMM_json/         # Detailed JSON results
├── vastai_YYYYMMDD_HHMM_performance.*  # Charts (PNG/SVG)
└── vastai_YYYYMMDD_HHMM_report.txt    # Analysis summary
```

## Quick Test (small matrix)

```bash
./scripts/benchmark_and_visualize.sh matrix/example100x100.mtx "test_small"
```

## Save Results to Git Branch

```bash
# 1. Create results branch with GPU architecture and timestamp
GPU_ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/NVIDIA //g' | tr ' -' '_' | tr '[:upper:]' '[:lower:]')
git checkout -b results-${GPU_ARCH}-$(date +%Y%m%d_%H%M)

# 2. Add results (force override .gitignore)
git add -f results/

# 3. Commit with GPU specs and performance summary
git commit -m "GPU benchmark results - $(nvidia-smi --query-gpu=name --format=csv,noheader)

Matrix: $(basename $MATRIX_FILE)
Performance summary from results CSV
Date: $(date)"

# 4. Push results branch to GitHub  
git push origin results-${GPU_ARCH}-$(date +%Y%m%d_%H%M)
```

## Retrieve Results Locally

```bash
# Fetch and checkout results branch
git fetch origin
git checkout results-GPU_ARCH-YYYYMMDD_HHMM

# Results now available in results/ directory
ls results/
```

**Note**: Do NOT merge the pull request - keep results branches separate from main codebase.

## Multimode Benchmark (NEW - Optimized)

For faster benchmarks that test all 6 SpMV implementations with a single matrix load:

```bash
# 1. Setup instance (same as above)
curl -s https://raw.githubusercontent.com/1fni/cuda-spmv-benchmark/main/scripts/vastai_setup.sh | bash

# 2. Navigate and build
cd cuda-spmv-benchmark
git pull && make

# 3. Run multimode benchmark (single matrix load, all 6 modes)
./scripts/vastai_multimode_benchmark.sh
```

### Multimode Advantages

- **⚡ Faster**: Single matrix load vs 6 separate loads
- **🔧 Efficient**: No I/O overhead between mode switches  
- **📊 Fair**: All modes use identical matrix in memory
- **🎯 Complete**: Tests all 6 implementations in one run

### Expected Runtime (Multimode)

- **A100-40GB**: ~2-3 minutes total (vs 5+ minutes separate)
- **H100-80GB**: ~1-2 minutes total (vs 4+ minutes separate)
- **RTX 4090**: ~3-5 minutes total (vs 8+ minutes separate)

### Multimode Output Files

```
results/
├── vastai_multimode_YYYYMMDD_HHMM_multimode_report.txt  # Full comparison
├── vastai_multimode_YYYYMMDD_HHMM_summary.csv          # CSV summary
├── vastai_multimode_YYYYMMDD_HHMM_json/                # Individual JSONs
├── vastai_multimode_YYYYMMDD_HHMM_performance.*        # Charts
```

### Standard vs Multimode

| Method | Matrix Loads | Runtime | Use Case |
|--------|--------------|---------|----------|
| **Standard** (`vastai_benchmark.sh`) | 1 per mode (6 total) | Longer | Single mode focus |
| **Multimode** (`vastai_multimode_benchmark.sh`) | 1 total | Faster | Complete comparison |

Choose **multimode** for comprehensive benchmarks, **standard** for individual mode analysis.