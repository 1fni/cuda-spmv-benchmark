#!/bin/bash
# Complete VastAI setup: CUDA project + Kokkos + AMGX
# Auto-detects GPU architecture

set -e

echo "=========================================="
echo "VastAI Full Setup (Project + Kokkos + AMGX)"
echo "=========================================="

# Detect GPU architecture
echo "Detecting GPU..."
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME (SM $GPU_ARCH)"

# Map compute capability to Kokkos arch
case $GPU_ARCH in
    70) KOKKOS_ARCH="VOLTA70" ;;
    75) KOKKOS_ARCH="TURING75" ;;
    80) KOKKOS_ARCH="AMPERE80" ;;
    86) KOKKOS_ARCH="AMPERE86" ;;
    89) KOKKOS_ARCH="ADA89" ;;
    90) KOKKOS_ARCH="HOPPER90" ;;
    *) echo "Warning: Unknown arch $GPU_ARCH, using AMPERE80"; KOKKOS_ARCH="AMPERE80" ;;
esac

echo "Kokkos architecture: $KOKKOS_ARCH"
echo ""

# 1. Install system dependencies
echo "Step 1: Installing system dependencies..."
apt-get update -qq
apt-get install -y build-essential git cmake wget nsight-systems-cli

# 2. Clone/update project
if [ ! -d "cuda-spmv-benchmark" ]; then
    echo "Step 2: Cloning project..."
    git clone https://github.com/1fni/cuda-spmv-benchmark.git
    cd cuda-spmv-benchmark
else
    echo "Step 2: Updating project..."
    cd cuda-spmv-benchmark
    git pull
fi

# 3. Build main project
echo ""
echo "Step 3: Building main project..."
make clean && make BUILD_TYPE=release

# 4. Install Kokkos + Kokkos-Kernels
echo ""
echo "Step 4: Setting up Kokkos..."

cd external

# Clone Kokkos
if [ ! -d "kokkos" ]; then
    git clone --depth 1 --branch 4.0.01 https://github.com/kokkos/kokkos.git
fi

# Clone Kokkos-Kernels
if [ ! -d "kokkos-kernels" ]; then
    git clone --depth 1 --branch 4.0.01 https://github.com/kokkos/kokkos-kernels.git
fi

# Build Kokkos
if [ ! -d "kokkos-install" ]; then
    echo "Building Kokkos..."
    rm -rf kokkos-build
    mkdir -p kokkos-build && cd kokkos-build
    cmake ../kokkos \
        -DCMAKE_INSTALL_PREFIX=../kokkos-install \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_${KOKKOS_ARCH}=ON \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc) && make install
    cd ..
    echo "✅ Kokkos installed"
fi

# Build Kokkos-Kernels
if [ ! -d "kokkos-kernels-install" ]; then
    echo "Building Kokkos-Kernels..."
    rm -rf kokkos-kernels-build
    mkdir -p kokkos-kernels-build && cd kokkos-kernels-build
    cmake ../kokkos-kernels \
        -DCMAKE_INSTALL_PREFIX=../kokkos-kernels-install \
        -DKokkos_DIR=../kokkos-install/lib/cmake/Kokkos \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc) && make install
    cd ..
    echo "✅ Kokkos-Kernels installed"
fi

# 5. Build Kokkos benchmarks
echo ""
echo "Step 5: Building Kokkos benchmarks..."
cd benchmarks/kokkos
make clean && make BUILD_TYPE=release
cd ../..

# 6. Install AMGX
echo ""
echo "Step 6: Installing AMGX..."
../../scripts/setup/install_amgx.sh

# 7. Build AMGX benchmarks
echo ""
echo "Step 7: Building AMGX benchmarks..."
cd benchmarks/amgx
make clean && make
cd ../..

cd ../..

# Verification
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo "GPU: $GPU_NAME (SM $GPU_ARCH)"
echo ""
echo "Built executables:"
ls -lh bin/release/cg_test \
       external/benchmarks/kokkos/kokkos_cg_baseline \
       external/benchmarks/amgx/amgx_cg_solver 2>/dev/null || echo "Some builds may have failed"
echo ""
echo "=========================================="
echo "Quick test commands:"
echo ""
echo "# Our CG solver:"
echo "./bin/release/cg_test matrix/stencil_5000x5000.mtx --mode=stencil5-csr-direct --device"
echo ""
echo "# Kokkos baseline:"
echo "./external/benchmarks/kokkos/kokkos_cg_baseline matrix/stencil_5000x5000_kokkos.mtx"
echo ""
echo "# AMGX solver:"
echo "./external/benchmarks/amgx/amgx_cg_solver matrix/stencil_5000x5000.mtx"
echo ""
echo "# Profile with nsys:"
echo "nsys profile -o results ./bin/release/cg_test matrix/stencil_5000x5000.mtx --mode=stencil5-csr-direct --device"
echo "=========================================="
