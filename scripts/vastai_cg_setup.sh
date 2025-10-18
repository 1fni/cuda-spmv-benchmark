#!/bin/bash
# VastAI setup for CG solver + Kokkos comparison
# Clones repo, installs Kokkos, builds everything

set -e

echo "=========================================="
echo "VastAI CG + Kokkos Setup"
echo "=========================================="

# 1. Run base setup (git clone, build our project)
echo "Step 1: Base setup (git + build our project)"
bash scripts/vastai_setup.sh

# 2. Detect GPU and generate config
echo ""
echo "Step 2: Detect GPU capabilities"
bash scripts/detect_gpu_config.sh

# Load GPU config
source /tmp/gpu_config.env
echo "Max matrix size: ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE}"

# 3. Install Kokkos + Kokkos-Kernels
echo ""
echo "Step 3: Install Kokkos + Kokkos-Kernels"

cd external

if [ ! -d "kokkos" ]; then
    echo "Cloning Kokkos..."
    git clone https://github.com/kokkos/kokkos.git
    cd kokkos
    git checkout 4.0.01
    cd ..
else
    echo "Kokkos already exists"
fi

if [ ! -d "kokkos-kernels" ]; then
    echo "Cloning Kokkos-Kernels..."
    git clone https://github.com/kokkos/kokkos-kernels.git
    cd kokkos-kernels
    git checkout 4.0.01
    cd ..
else
    echo "Kokkos-Kernels already exists"
fi

# Build Kokkos
if [ ! -d "kokkos-install" ]; then
    echo "Building Kokkos..."
    mkdir -p kokkos-build
    cd kokkos-build
    cmake ../kokkos \
        -DCMAKE_INSTALL_PREFIX=../kokkos-install \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_AMPERE80=ON \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    make install
    cd ..
    echo "Kokkos installed"
else
    echo "Kokkos already installed"
fi

# Build Kokkos-Kernels
if [ ! -d "kokkos-kernels-install" ]; then
    echo "Building Kokkos-Kernels..."
    mkdir -p kokkos-kernels-build
    cd kokkos-kernels-build
    cmake ../kokkos-kernels \
        -DCMAKE_INSTALL_PREFIX=../kokkos-kernels-install \
        -DKokkos_DIR=../kokkos-install/lib/cmake/Kokkos \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    make install
    cd ..
    echo "Kokkos-Kernels installed"
else
    echo "Kokkos-Kernels already installed"
fi

cd ..

# 4. Build Kokkos comparison executables
echo ""
echo "Step 4: Build Kokkos comparison executables"

# Build Kokkos matrix generator
g++ -O3 external/generate_kokkos_matrix.cpp -o external/generate_kokkos_matrix

# Build Kokkos CG solver
cd external
make -f Makefile_kokkos kokkos_cg_struct
cd ..

# 5. Build our CG solver
echo ""
echo "Step 5: Build our CG solver"
make bin/cg_test

# Verification
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "GPU: $GPU_NAME"
echo "Max matrix size: ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE}"
echo ""
echo "Built executables:"
ls -lh bin/cg_test external/kokkos_cg_struct external/generate_kokkos_matrix
echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Generate max-size matrix:"
echo "   ./bin/generate_matrix ${MAX_MATRIX_SIZE} matrix/stencil_max.mtx"
echo ""
echo "2. Generate Kokkos matrix (manual):"
echo "   ./external/generate_kokkos_matrix ${MAX_MATRIX_SIZE} matrix/stencil_max_kokkos.mtx"
echo ""
echo "3. Run benchmarks (see commands in vastai_cg_commands.txt)"
echo "=========================================="
