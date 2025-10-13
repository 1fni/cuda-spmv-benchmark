# CUDA SpMV Benchmark Suite

[![CI](https://github.com/1fni/cuda-spmv-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/1fni/cuda-spmv-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance Sparse Matrix-Vector Multiplication (SpMV) implementations with GPU acceleration, performance metrics, and automated benchmarking.

## Performance Results

**Enterprise-Scale Benchmark** on NVIDIA H100 NVL (94GB HBM3, Compute 9.0)
Matrix: 265M×265M with 1.33B non-zeros (0.002% sparsity)

| Format          | Time (ms) | GFLOPS  | Bandwidth (GB/s) | Speedup |
|-----------------|-----------|---------|------------------|---------|
| **CSR**         | 8.25      | 321.9   | 2575             | 1.00×   |
| **STENCIL5-OPT**| 5.53      | 480.7   | 3653             | **1.49×**|

*Kernel-level measurements via CUDA events*

## Technical Highlights

- **Optimized Stencil Kernels** - Custom CUDA implementation outperforms cuSPARSE CSR by 1.49× on H100 NVL
- **Multi-GPU Scaling** - Peer-to-peer communication with 1D row-band decomposition
- **Comprehensive Metrics** - GFLOPS, memory bandwidth, arithmetic intensity analysis
- **Fair Benchmarking** - Unified compilation flags (-O3) for credible industry comparisons (AmgX, Kokkos)
- **Export Formats** - JSON/CSV output for automated analysis and CI/CD integration

## Features

- **Multiple SpMV Formats**: CSR, ELLPACK, 5-point stencil with cuSPARSE and custom CUDA kernels
- **Performance Metrics**: GFLOPS, memory bandwidth, arithmetic intensity analysis  
- **Export Formats**: JSON, CSV, human-readable output with file export
- **Automated Benchmarking**: Comparison scripts and performance analysis
- **Memory Management**: Zero-leak implementation with proper cleanup
- **CI/CD Pipeline**: Automated build validation with GitHub Actions

## Quick Start

```bash
# Build the benchmark suite
make

# Run CSR benchmark  
./bin/spmv_bench matrix/example100x100.mtx --mode=csr

# Compare formats and export results
./benchmark_suite.sh

# Export metrics to JSON for analysis
./bin/spmv_bench matrix/example.mtx --mode=stencil5 --output-format=json --output-file=results.json
```

## Supported Formats

### CSR (Compressed Sparse Row)
- General sparse matrices using cuSPARSE
- Optimal for irregular sparsity patterns
- Industry-standard format with broad compatibility

### STENCIL5 (5-Point Stencil)  
- Custom CUDA kernels for structured grid operations
- Optimized memory access patterns with ELLPACK storage
- **Best performance** for finite difference applications

### ELLPACK
- Regular sparsity patterns using cuSPARSE
- Memory-aligned storage for consistent row widths

## Multi-GPU Architecture

Scalable implementations for stencil and CSR formats across multiple GPUs.

### Features

- **Peer-to-peer communication** - Direct GPU-to-GPU transfers without CPU bounce
- **1D row-band decomposition** - Each GPU handles contiguous row bands
- **Minimal overhead** - Halo exchange for stencil boundary conditions only
- **Automatic GPU detection** - Runtime device enumeration and validation

### Usage

```bash
# Multi-GPU stencil (2 GPUs)
./bin/spmv_bench matrix/stencil_large.mtx --mode=stencil5-mgpu

# Multi-GPU CSR
./bin/spmv_bench matrix/large_sparse.mtx --mode=csr-mgpu
```

### Architecture Details

**Row-band decomposition:**
```
GPU 0: rows [0, N/2)         ┐
GPU 1: rows [N/2, N)         ┘ Peer-to-peer halo exchange
```

Each GPU:
1. Computes local SpMV on its row band
2. Exchanges boundary data via P2P
3. Synchronizes results

**Performance considerations:**
- Best for large matrices where computation dominates communication
- Requires CUDA-capable peer access between GPUs
- Halo size: 1 row for 5-point stencil

## Architecture

```
src/
├── main/main.cu           # CLI interface and benchmark orchestration
├── spmv/
│   ├── spmv.cu           # Operator dispatch and selection
│   ├── spmv_cusparse_csr.cu      # CSR implementation with cuSPARSE
│   ├── spmv_stencil.cu   # Custom 5-point stencil CUDA kernels  
│   └── spmv_metrics.cu   # Performance metrics and analysis
├── io/io.cu              # Matrix Market I/O and format conversion
└── matrix/generate_matrix.cu     # Stencil matrix generator
```

## Build System

**Dual build approach** for maximum flexibility:
- **Makefile**: Primary build system optimized for CUDA/HPC workflows
- **CMake**: Testing framework with Google Test integration

```bash
# Release build (default)
make

# Debug build with GPU debugging
make BUILD_TYPE=debug  

# Run test suite
cd tests && mkdir build && cd build && cmake .. && make && ./test_runner
```

## Performance Analysis

The benchmark suite provides detailed analysis:

- **Execution timing** with CUDA events (kernel-only precision)
- **GFLOPS calculation** based on 2×nnz floating point operations  
- **Memory bandwidth** analysis with format-specific traffic calculation
- **Arithmetic intensity** classification (memory-bound vs compute-bound)
- **Performance recommendations** based on bottleneck analysis

Example output:
```
=== SpMV Performance Metrics ===
Operator: stencil5

--- Performance Metrics ---
Execution time: 15.0 μs
GFLOPS: 6.61
Memory bandwidth: 50.64 GB/s

--- Performance Analysis ---
Arithmetic intensity: 0.131 FLOP/byte
Classification: Memory-bound (low arithmetic intensity)
Optimization focus: Memory access patterns, data locality
```

## Usage Examples

### Basic Benchmarking
```bash
# Benchmark different formats
./bin/spmv_bench matrix/example100x100.mtx --mode=csr
./bin/spmv_bench matrix/stencil_100x100.mtx --mode=stencil5

# Generate 5-point stencil matrix
./bin/generate_matrix 100 matrix/my_stencil.mtx
```

### Advanced Analysis
```bash  
# Export detailed metrics to JSON
./bin/spmv_bench matrix/example.mtx --mode=csr \
    --output-format=json --output-file=csr_analysis.json

# Generate CSV for spreadsheet analysis  
./bin/spmv_bench matrix/example.mtx --mode=stencil5 \
    --output-format=csv --output-file=performance.csv

# Automated comparative benchmarking
./benchmark_suite.sh
```

### Integration Examples
```bash
# CI/CD performance regression testing
./bin/spmv_bench matrix/test.mtx --mode=stencil5 --output-format=json | \
  jq '.benchmark.performance.gflops' | awk '{if($1<5.0) exit 1}'

# Batch analysis across matrix sizes
for size in 50 100 200 500; do
  ./bin/generate_matrix $size matrix/test_${size}.mtx
  ./bin/spmv_bench matrix/test_${size}.mtx --mode=stencil5 --output-format=csv
done
```

## Matrix Formats

**Matrix Market (.mtx)** input format with automatic symmetry expansion:
```
%%MatrixMarket matrix coordinate real general
3 3 5
1 1 4.0
1 2 1.0  
2 2 6.0
2 3 2.0
3 3 3.0
```

**Generated stencil matrices** for structured grid applications:
```bash
./bin/generate_matrix 100 matrix/stencil_100x100.mtx
# Creates 100×100 5-point stencil: center=4, neighbors=-1
```

## Development

### Adding New SpMV Implementations
1. Create implementation file in `src/spmv/spmv_new_format.cu`
2. Define `SpmvOperator` structure with `init`, `run_timed`, `free` functions
3. Add extern declaration in `include/spmv.h`  
4. Register operator in `get_operator()` function

### Testing
```bash
# Local testing with GPU
cd tests && mkdir build && cd build
cmake .. && make
./test_runner

# Specific test categories
./test_runner --gtest_filter="WrapperTest*"
./test_runner --gtest_filter="*Performance*"
```

## Requirements

- **NVIDIA GPU** with Compute Capability ≥ 7.0
- **CUDA Toolkit** ≥ 11.0 with cuSPARSE and cuBLAS
- **C++ Compiler** supporting C++14
- **CMake** ≥ 3.18 (for testing framework)

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@software{cuda_spmv_benchmark,
  author = {Bouhrour, Stephane},
  title = {CUDA SpMV Benchmark Suite},
  year = {2025},
  url = {https://github.com/1fni/cuda-spmv-benchmark}
}
```