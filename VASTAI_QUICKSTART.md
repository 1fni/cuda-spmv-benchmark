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