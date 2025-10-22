# Scripts

Automation scripts for building, benchmarking, and profiling the CUDA SpMV project.

## Directory Structure

```
scripts/
├── setup/         # Installation and environment setup
├── benchmarking/  # Performance benchmarks (local + VastAI)
├── profiling/     # GPU profiling (nsys, ncu)
├── analysis/      # Results visualization and export
└── utils/         # Validation and helper scripts
```

## Quick Reference

### Setup
- `setup/full_setup.sh` - Complete setup (project + Kokkos + AMGX)
- `setup/remote_setup.sh` - Basic setup (project only)
- `setup/install_amgx.sh` - Install NVIDIA AMGX library
- `setup/detect_gpu_config.sh` - Detect GPU architecture and capabilities

### Benchmarking
- `benchmarking/multimode_benchmark.sh` - Compare all SpMV modes
- `benchmarking/cg_comparison_setup.sh` - CG solver benchmark setup
- `benchmarking/remote_benchmark.sh` - Single SpMV mode benchmark
- `benchmarking/benchmark_and_visualize.sh` - Benchmark with plots

### Profiling
- `profiling/profile_kernel.sh` - Profile kernels with nsys/ncu

### Analysis
- `analysis/generate_performance_chart.sh` - Create performance plots
- `analysis/save_results_to_git.sh` - Commit benchmark results

### Utils
- `utils/test_validation.sh` - Validate SpMV correctness
- `utils/push_remote_results.sh` - Transfer results from remote instance

## Usage Examples

**Complete setup (VastAI, RunPod, AWS, etc.):**
```bash
./scripts/setup/full_setup.sh
```

**Run multi-mode benchmark:**
```bash
./scripts/benchmarking/multimode_benchmark.sh
```

**Profile AMGX:**
```bash
cd external/benchmarks/amgx
nsys profile -o amgx_cg ./amgx_cg_solver ../../../matrix/stencil_5000x5000.mtx
```

**Profile our CG:**
```bash
nsys profile -o our_cg ./bin/release/cg_test matrix/stencil_5000x5000.mtx --mode=stencil5-csr-direct --device
```

**Compare results:**
```bash
nsys stats --report cuda_api_sum amgx_cg.nsys-rep > amgx_api.txt
nsys stats --report cuda_api_sum our_cg.nsys-rep > our_api.txt
```
