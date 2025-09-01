#!/bin/bash

# GPU detection and benchmark parameter optimization
# Uses runtime GPU properties instead of hardcoded models

set -e

detect_gpu_capabilities() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found"
        exit 1
    fi
    
    # Get actual GPU properties
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    VRAM_GB=$((VRAM_MB / 1024))
    
    # Calculate max matrix size based on VRAM (safety factor 0.7)
    # Formula: size^2 * 5 (stencil points) * 8 (bytes) * 3 (x,y,matrix) < 0.7 * VRAM
    MAX_ELEMENTS=$(echo "$VRAM_GB * 1024^3 * 0.7 / (5 * 8 * 3)" | bc -l)
    MAX_SIZE=$(echo "sqrt($MAX_ELEMENTS)" | bc -l | cut -d. -f1)
    
    # Round down to nearest power of 2 for clean benchmarks
    MAX_SIZE_POW2=1024
    while [ $((MAX_SIZE_POW2 * 2)) -le "$MAX_SIZE" ]; do
        MAX_SIZE_POW2=$((MAX_SIZE_POW2 * 2))
    done
    
    echo "GPU: $GPU_NAME"
    echo "VRAM: ${VRAM_GB}GB"
    echo "Max matrix size: ${MAX_SIZE_POW2}x${MAX_SIZE_POW2}"
    
    # Generate test sizes (logarithmic scaling)
    SIZES=(256 512 1024 2048)
    current=4096
    while [ "$current" -le "$MAX_SIZE_POW2" ]; do
        SIZES+=($current)
        current=$((current * 2))
    done
    
    echo "Test sizes: ${SIZES[*]}"
    
    # Export for other scripts
    cat > /tmp/gpu_config.env << EOF
export GPU_NAME="$GPU_NAME"
export VRAM_GB=$VRAM_GB
export MAX_MATRIX_SIZE=$MAX_SIZE_POW2
export MATRIX_SIZES="${SIZES[*]}"
EOF
    
    echo "Config exported to /tmp/gpu_config.env"
}

detect_gpu_capabilities