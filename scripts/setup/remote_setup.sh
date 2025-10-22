#!/bin/bash

# Vast.ai instance setup for CUDA SpMV benchmarking
# Installs dependencies and configures environment

set -e

echo "=== Vast.ai CUDA SpMV Setup ==="

# Update system
apt-get update -qq

# Install essential tools
apt-get install -y \
    build-essential \
    git \
    bc \
    gnuplot-nox \
    curl \
    wget

# Verify CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA not found. Use CUDA-enabled Vast.ai image."
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
echo "CUDA version: $CUDA_VERSION"

# Verify GPU is accessible
if ! nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi failed. GPU not accessible."
    exit 1
fi

# Display GPU info
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Clone/setup project if not exists
if [ ! -d "cuda-spmv-benchmark" ]; then
    echo "Cloning project..."
    git clone https://github.com/1fni/cuda-spmv-benchmark.git
    cd cuda-spmv-benchmark
else
    echo "Project exists, pulling latest..."
    cd cuda-spmv-benchmark
    git pull
fi

# Build project
echo "Building SpMV benchmark..."
make clean && make

# Test build
if [ -f "bin/spmv_bench" ] && [ -f "bin/generate_matrix" ]; then
    echo "Build successful!"
else
    echo "Build failed!"
    exit 1
fi

# Generate test matrix
echo "Generating test matrix..."
./bin/generate_matrix 128 matrix/test_128x128.mtx

# Quick verification run
echo "Running verification test..."
./bin/spmv_bench matrix/test_128x128.mtx --mode=csr --output-format=json > /tmp/test_result.json

if [ $? -eq 0 ]; then
    echo "Setup complete! Ready for benchmarking."
    echo
    echo "Next steps:"
    echo "1. ./scripts/detect_gpu_config.sh"
    echo "2. ./scripts/benchmark_and_visualize.sh matrix/stencil_gpu_maxmem.mtx"
else
    echo "Verification failed!"
    exit 1
fi