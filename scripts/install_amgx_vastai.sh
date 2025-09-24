#!/bin/bash
# AmgX Installation Script for VastAI GPU Instances
# Handles various VastAI environments and permission constraints
# Author: Bouhrour Stephane
# Date: 2025-09-24

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AMGX_VERSION="v2.3.0"  # Stable version
INSTALL_PREFIX="$HOME/amgx_local"
TEMP_DIR="/tmp/amgx_build_$$"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on VastAI (common indicators)
detect_vastai() {
    if [[ -n "$VAST_CONTAINERNAME" ]] || [[ -n "$RUNPOD_POD_ID" ]] || [[ "$PWD" =~ "/workspace" ]] || [[ "$HOME" =~ "/root" ]]; then
        print_status "Detected cloud GPU environment (VastAI/RunPod/similar)"
        return 0
    fi
    return 1
}

# Check CUDA installation and version
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA toolkit not found. nvcc is required for AmgX compilation."
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    print_success "Found CUDA version: $CUDA_VERSION"
    
    # Check for cuSOLVER library
    if [[ -f "/usr/local/cuda/lib64/libcusolver.so" ]] || [[ -f "/usr/lib/x86_64-linux-gnu/libcusolver.so" ]]; then
        print_success "cuSOLVER library found"
    else
        print_warning "cuSOLVER library not found in standard locations"
        print_status "AmgX may fail to link. Continuing anyway..."
    fi
}

# Check system dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Essential build tools
    command -v cmake >/dev/null 2>&1 || missing_deps+=("cmake")
    command -v make >/dev/null 2>&1 || missing_deps+=("make")
    command -v g++ >/dev/null 2>&1 || missing_deps+=("g++")
    command -v git >/dev/null 2>&1 || missing_deps+=("git")
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_warning "Missing dependencies: ${missing_deps[*]}"
        print_status "Attempting to install dependencies..."
        
        # Try different package managers
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y cmake build-essential git
        elif command -v yum >/dev/null 2>&1; then
            sudo yum install -y cmake gcc-c++ make git
        else
            print_error "Cannot install dependencies. Please install manually: ${missing_deps[*]}"
            exit 1
        fi
    else
        print_success "All dependencies found"
    fi
}

# Check available disk space
check_disk_space() {
    print_status "Checking disk space..."
    
    local available=$(df "$HOME" | awk 'NR==2 {print $4}')
    local required=2097152  # 2GB in KB
    
    if [[ $available -lt $required ]]; then
        print_warning "Low disk space: $(($available/1024))MB available, 2GB recommended"
        print_status "Continuing with installation..."
    else
        print_success "Sufficient disk space: $(($available/1024))MB available"
    fi
}

# Test sudo access
check_sudo() {
    if sudo -n true 2>/dev/null; then
        print_success "Sudo access available"
        return 0
    else
        print_warning "No sudo access - using local installation prefix"
        return 1
    fi
}

# Clean up previous installations
cleanup_previous() {
    print_status "Cleaning up previous installations..."
    
    # Remove temporary directory if it exists
    [[ -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"
    
    # Clean previous local installation if requested
    if [[ -d "$INSTALL_PREFIX" ]]; then
        print_warning "Previous AmgX installation found at $INSTALL_PREFIX"
        read -p "Remove previous installation? [y/N]: " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_PREFIX"
            print_success "Previous installation removed"
        fi
    fi
}

# Download and build AmgX
build_amgx() {
    print_status "Starting AmgX build process..."
    
    # Create temporary build directory
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Clone AmgX repository
    print_status "Cloning AmgX repository (version $AMGX_VERSION)..."
    git clone --depth 1 --branch "$AMGX_VERSION" https://github.com/NVIDIA/AMGX.git
    cd AMGX
    
    # Initialize submodules (required for Thrust)
    print_status "Initializing git submodules (Thrust dependency)..."
    git submodule update --init --recursive
    
    # Create build directory
    mkdir build && cd build
    
    # Configure CMake
    print_status "Configuring CMake build..."
    local cmake_args=(
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_NO_MPI=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
    
    # Add CUDA architecture flags for better compatibility
    if detect_vastai; then
        # Common VastAI GPU architectures
        cmake_args+=(-DCUDA_ARCH="70;75;80;86;89;90")
        print_status "Using multi-architecture CUDA build for cloud GPU compatibility"
    fi
    
    cmake "${cmake_args[@]}" ..
    
    # Build AmgX
    print_status "Building AmgX (this may take 10-15 minutes)..."
    local nproc_count=$(nproc)
    make -j"$nproc_count"
    
    # Install AmgX
    print_status "Installing AmgX to $INSTALL_PREFIX..."
    make install
    
    print_success "AmgX build completed successfully"
}

# Update project Makefile
update_makefile() {
    local project_root
    project_root=$(cd "$(dirname "$0")/.." && pwd)
    local makefile="$project_root/Makefile"
    
    if [[ -f "$makefile" ]]; then
        print_status "Updating project Makefile..."
        
        # Create backup
        cp "$makefile" "$makefile.backup"
        
        # Update AMGX_DIR in Makefile
        sed -i "s|AMGX_DIR ?= /usr/local|AMGX_DIR ?= $INSTALL_PREFIX|g" "$makefile"
        
        print_success "Makefile updated (backup saved as Makefile.backup)"
    else
        print_warning "Project Makefile not found - you'll need to set AMGX_DIR manually"
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying AmgX installation..."
    
    # Check header files
    if [[ -f "$INSTALL_PREFIX/include/amgx_c.h" ]]; then
        print_success "AmgX headers found"
    else
        print_error "AmgX headers not found at $INSTALL_PREFIX/include/"
        return 1
    fi
    
    # Check library files
    if [[ -f "$INSTALL_PREFIX/lib/libamgx.a" ]] || [[ -f "$INSTALL_PREFIX/lib/libamgxsh.so" ]]; then
        print_success "AmgX libraries found"
    else
        print_error "AmgX libraries not found at $INSTALL_PREFIX/lib/"
        return 1
    fi
    
    print_success "AmgX installation verified"
}

# Test compilation
test_compilation() {
    print_status "Testing AmgX compilation with project..."
    
    local project_root
    project_root=$(cd "$(dirname "$0")/.." && pwd)
    
    cd "$project_root"
    
    # Try to compile
    if make clean && make; then
        print_success "Project compilation successful with AmgX"
        
        # Quick functionality test
        if [[ -f "matrix/test_stencil_32x32.mtx" ]]; then
            print_status "Running quick AmgX functionality test..."
            if timeout 30 ./bin/spmv_bench matrix/test_stencil_32x32.mtx --mode=amgx-stencil >/dev/null 2>&1; then
                print_success "AmgX functionality test passed"
            else
                print_warning "AmgX functionality test failed or timed out"
            fi
        fi
    else
        print_error "Project compilation failed with AmgX"
        return 1
    fi
}

# Cleanup temporary files
cleanup_temp() {
    print_status "Cleaning up temporary files..."
    [[ -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"
    print_success "Cleanup completed"
}

# Main installation process
main() {
    echo "=================================================="
    echo "  AmgX Installation Script for VastAI/Cloud GPU  "
    echo "=================================================="
    echo
    
    # Environment detection
    detect_vastai && print_status "Cloud GPU environment detected"
    
    # Pre-installation checks
    check_cuda
    check_dependencies
    check_disk_space
    check_sudo
    
    # Cleanup and build
    cleanup_previous
    build_amgx
    
    # Post-installation setup
    update_makefile
    verify_installation
    
    # Test the installation
    test_compilation
    
    # Final cleanup
    cleanup_temp
    
    echo
    echo "=================================================="
    print_success "AmgX installation completed successfully!"
    echo "=================================================="
    echo
    echo "Installation details:"
    echo "  - AmgX version: $AMGX_VERSION"
    echo "  - Install location: $INSTALL_PREFIX"
    echo "  - Headers: $INSTALL_PREFIX/include/"
    echo "  - Libraries: $INSTALL_PREFIX/lib/"
    echo
    echo "To use AmgX with your project:"
    echo "  export AMGX_DIR=$INSTALL_PREFIX"
    echo "  make clean && make"
    echo
    echo "Test AmgX integration:"
    echo "  ./bin/spmv_bench matrix/test.mtx --mode=amgx-stencil"
    echo
}

# Error handling
trap 'print_error "Installation failed at line $LINENO. Check the output above for details."' ERR
trap cleanup_temp EXIT

# Run main installation
main "$@"