#!/bin/bash
# Complete setup: CUDA project + optional Kokkos + AMGX
# Auto-detects GPU architecture
#
# Usage:
#   ./full_setup.sh                    # Install main project only (default)
#   ./full_setup.sh --all              # Install everything (main + Kokkos + AMGX)
#   ./full_setup.sh --kokkos           # Install main + Kokkos
#   ./full_setup.sh --amgx             # Install main + AMGX
#   ./full_setup.sh --kokkos --amgx    # Install main + both externals

set -e

# Parse arguments - default: no externals
INSTALL_KOKKOS=false
INSTALL_AMGX=false

for arg in "$@"; do
    case $arg in
        --all)
            INSTALL_KOKKOS=true
            INSTALL_AMGX=true
            ;;
        --kokkos) INSTALL_KOKKOS=true ;;
        --amgx) INSTALL_AMGX=true ;;
        *) echo "Unknown option: $arg"; echo "Use: --all, --kokkos, or --amgx"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Setup Configuration"
echo "=========================================="
echo "Main project: YES"
echo "Kokkos:       $($INSTALL_KOKKOS && echo YES || echo NO)"
echo "AMGX:         $($INSTALL_AMGX && echo YES || echo NO)"
echo "=========================================="
echo ""

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

# Verify we're in project root
if [ ! -f "Makefile" ] || [ ! -d "src" ]; then
    echo "Error: Must be run from project root directory"
    echo "Usage: cd cuda-spmv-benchmark && ./scripts/setup/full_setup.sh"
    exit 1
fi

# 1. Install system dependencies
echo "Step 1: Installing system dependencies..."
apt-get update -qq
apt-get install -y build-essential git cmake wget nsight-systems-cli

# 2. Build main project
echo ""
echo "Step 2: Building main project..."
make clean && make BUILD_TYPE=release

# 3. Install Kokkos + Kokkos-Kernels (optional)
if [ "$INSTALL_KOKKOS" = true ]; then
    echo ""
    echo "Step 3: Setting up Kokkos..."

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

    # Build Kokkos benchmarks
    echo ""
    echo "Building Kokkos benchmarks..."
    cd benchmarks/kokkos
    make clean && make BUILD_TYPE=release
    cd ../..

    cd ../..
fi

# 4. Install AMGX (optional)
if [ "$INSTALL_AMGX" = true ]; then
    echo ""
    echo "Step 4: Installing AMGX..."
    ./scripts/setup/install_amgx.sh

    # Build AMGX benchmarks
    echo ""
    echo "Building AMGX benchmarks..."
    cd external/benchmarks/amgx
    make clean && make
    cd ../../..
fi

# Verification
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo "GPU: $GPU_NAME (SM $GPU_ARCH)"
echo ""
echo "Built executables:"
ls -lh bin/release/cg_test 2>/dev/null || echo "Main build may have failed"

if [ "$INSTALL_KOKKOS" = true ]; then
    ls -lh external/benchmarks/kokkos/kokkos_cg_baseline 2>/dev/null || echo "Kokkos build may have failed"
fi

if [ "$INSTALL_AMGX" = true ]; then
    ls -lh external/benchmarks/amgx/amgx_cg_solver 2>/dev/null || echo "AMGX build may have failed"
fi

echo ""
echo "=========================================="
echo "Quick test commands:"
echo ""
echo "# Our CG solver:"
echo "./bin/release/cg_test matrix/stencil_5000x5000.mtx --mode=stencil5-csr-direct --device"

if [ "$INSTALL_KOKKOS" = true ]; then
    echo ""
    echo "# Kokkos baseline:"
    echo "./external/benchmarks/kokkos/kokkos_cg_baseline matrix/stencil_5000x5000_kokkos.mtx"
fi

if [ "$INSTALL_AMGX" = true ]; then
    echo ""
    echo "# AMGX solver:"
    echo "./external/benchmarks/amgx/amgx_cg_solver matrix/stencil_5000x5000.mtx"
fi

echo ""
echo "# Profile with nsys:"
echo "nsys profile -o results ./bin/release/cg_test matrix/stencil_5000x5000.mtx --mode=stencil5-csr-direct --device"
echo "=========================================="
