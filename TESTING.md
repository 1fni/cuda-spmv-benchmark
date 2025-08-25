# CUDA SpMV Local Testing Guide

This guide covers local testing with GPU hardware. The GitHub CI/CD performs build validation and smoke tests only - **full CUDA kernel testing requires GPU hardware**.

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with Compute Capability ≥ 7.0
  - RTX series (RTX 3060, 4070, etc.)  
  - Tesla/A100/V100 for HPC environments
  - GTX 1080 Ti or newer
- **Minimum 2GB GPU memory** (recommended 4GB+)

### Software Requirements
```bash
# CUDA Toolkit 12.0 or later
nvidia-smi  # Verify driver installation
nvcc --version  # Verify CUDA compiler

# Required libraries
sudo apt-get install libcusparse-dev libcublas-dev

# For testing framework
sudo apt-get install libgtest-dev cmake build-essential
```

### Verify GPU Setup
```bash
# Check GPU status
nvidia-smi

# Verify CUDA functionality
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

## Quick Start Testing

### 1. Build the Project
```bash
# Build release binaries
make

# Or build debug version for testing
make BUILD_TYPE=debug
```

### 2. Basic Functionality Test
```bash
# Test with the provided 3x3 matrix
./bin/release/spmv_bench matrix/example3x3.mtx --mode=csr
./bin/release/spmv_bench matrix/example3x3.mtx --mode=stencil5
./bin/release/spmv_bench matrix/example3x3.mtx --mode=ellpack
```

**Expected output:**
```
[CSR] SpMV completed in X.XX ms
[STENCIL5] SpMV completed in X.XX ms  
[ELLPACK] SpMV completed in X.XX ms
```

### 3. Generate Larger Test Matrices
```bash
# Generate different sizes for performance testing
./bin/release/generate_matrix 10 matrix/test_10x10.mtx
./bin/release/generate_matrix 50 matrix/test_50x50.mtx
./bin/release/generate_matrix 100 matrix/test_100x100.mtx

# Test with larger matrices
./bin/release/spmv_bench matrix/test_100x100.mtx --mode=csr
```

## Test Suite

### 1. Build and Run Full Test Framework
```bash
# Navigate to tests directory
cd tests

# Build test framework (first time only)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run complete test suite
./spmv_tests
```

### 2. Test Suite Components

#### Basic Wrapper Tests
```bash
# Test specific components
./spmv_tests --gtest_filter="SpMVWrapperTest*"
```

**What this tests:**
- Operator construction (CSR, ELLPACK, STENCIL5)
- Matrix loading and initialization
- CUDA memory management
- Error handling

#### Correctness Validation Tests
```bash
# Mathematical correctness tests
./spmv_tests --gtest_filter="*Correctness*"
```

**What this validates:**
- SpMV computation accuracy with known results
- Cross-validation between operators (CSR vs STENCIL5)
- Numerical precision (checksum validation)

#### Helper Utilities Tests
```bash
# Test CUDA utilities and matrix fixtures
./spmv_tests --gtest_filter="HelpersDemo*"
```

**What this covers:**
- CUDA test utilities (vector generation, comparison)
- Matrix fixtures (identity, diagonal, stencil patterns)
- Performance benchmarking tools
- GPU memory monitoring

### 3. Expected Test Results

#### Correctness Validation
For the 3x3 stencil matrix with input vector of all ones:
```
Expected checksum: -60.0
- CSR result: -60.0 ✅
- STENCIL5 result: -60.0 ✅
- Cross-validation: PASSED ✅
```

#### Performance Tests
```
Matrix: 3x3 (9 points, 33 non-zeros)
CSR Performance: ~X.XX GFLOPS
STENCIL5 Performance: ~X.XX GFLOPS
Memory Bandwidth: ~X.XX GB/s
```

## Performance Benchmarking

### 1. Comparative Performance Testing
```bash
# Generate matrices of different sizes
for size in 10 25 50 100; do
  ./bin/release/generate_matrix $size matrix/perf_${size}x${size}.mtx
done

# Run performance comparison
for matrix in matrix/perf_*.mtx; do
  echo "Testing $matrix:"
  ./bin/release/spmv_bench $matrix --mode=csr
  ./bin/release/spmv_bench $matrix --mode=stencil5
  ./bin/release/spmv_bench $matrix --mode=ellpack
  echo "---"
done
```

### 2. Automated Performance Suite
```bash
# Run performance tests
./spmv_tests --gtest_filter="*Performance*"
```

This generates:
- FLOPS measurements for each operator
- Memory bandwidth utilization
- Execution time comparisons
- Performance scaling analysis

## Troubleshooting

### Common Issues

#### 1. CUDA Device Not Found
```
Error: CUDA error: no CUDA-capable device is detected (2)
```

**Solutions:**
- Verify `nvidia-smi` works
- Check CUDA driver installation
- Ensure GPU is not being used by other processes

#### 2. Out of Memory Errors
```
Error: CUDA error: out of memory (2)
```

**Solutions:**
- Use smaller matrices for testing
- Check available GPU memory with `nvidia-smi`
- Close other GPU applications

#### 3. cuSPARSE Library Not Found
```
Error: error while loading shared libraries: libcusparse.so
```

**Solutions:**
```bash
# Install cuSPARSE development libraries
sudo apt-get install libcusparse-dev

# Or set library path manually
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 4. Test Compilation Issues
```bash
# Ensure all dependencies are installed
sudo apt-get update
sudo apt-get install libgtest-dev cmake build-essential

# Clean and rebuild tests
cd tests/build
make clean
cmake .. && make
```

### Debug Mode Testing
```bash
# Build debug version for detailed error information
make BUILD_TYPE=debug

# Run with debug binaries
./bin/debug/spmv_bench matrix/example3x3.mtx --mode=csr

# Debug test framework
cd tests/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make
./spmv_tests
```

## Advanced Testing

### 1. Memory Leak Detection
```bash
# Install Valgrind (if available for CUDA)
sudo apt-get install valgrind

# Note: Valgrind has limited CUDA support
# Use CUDA-MEMCHECK for CUDA-specific issues
cuda-memcheck ./bin/release/spmv_bench matrix/example3x3.mtx --mode=csr
```

### 2. Profile with Nsight
```bash
# Profile with NVIDIA Nsight Compute
ncu --set full ./bin/release/spmv_bench matrix/test_100x100.mtx --mode=csr

# Profile with NVIDIA Nsight Systems
nsys profile ./bin/release/spmv_bench matrix/test_100x100.mtx --mode=stencil5
```

### 3. Custom Matrix Testing
```bash
# Create custom matrices for specific testing scenarios
./bin/release/generate_matrix 200 matrix/custom_200x200.mtx

# Test with your custom matrices
./bin/release/spmv_bench matrix/custom_200x200.mtx --mode=csr
```

## Integration with Development Workflow

### 1. Pre-commit Testing
```bash
#!/bin/bash
# pre-commit-tests.sh
echo "Running CUDA SpMV pre-commit tests..."

# Build
make clean && make

# Quick validation
./bin/release/spmv_bench matrix/example3x3.mtx --mode=csr > /dev/null
if [ $? -eq 0 ]; then
  echo "✅ Basic functionality test passed"
else
  echo "❌ Basic functionality test failed"
  exit 1
fi

# Run critical tests
cd tests/build && ./spmv_tests --gtest_filter="*Correctness*" > /dev/null
if [ $? -eq 0 ]; then
  echo "✅ Correctness tests passed"
else
  echo "❌ Correctness tests failed"
  exit 1
fi

echo "✅ Pre-commit tests completed successfully"
```

### 2. Continuous Integration Workflow
```bash
# Local testing before push
make clean && make
cd tests/build && ./spmv_tests

# Push to trigger GitHub Actions CI/CD
git add . && git commit -m "feature: description"
git push origin feature-branch

# GitHub Actions runs build validation + smoke tests
# Local testing validates full CUDA functionality
```

---

## Summary

This testing approach provides:

- **🏗️ Build validation**: GitHub Actions CI/CD  
- **🧪 Full testing**: Local GPU-enabled environment
- **📊 Performance**: Benchmarking tools
- **🔧 Debug support**: Multiple debugging approaches
- **🚀 Integration**: Smooth development workflow

**Remember**: CI/CD validates build integrity, but **GPU hardware testing is essential** for validating CUDA kernel correctness and performance.