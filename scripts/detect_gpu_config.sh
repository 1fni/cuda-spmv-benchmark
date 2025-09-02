#!/bin/bash

# GPU detection and benchmark parameter optimization
# Uses runtime GPU properties instead of hardcoded models

set -e

detect_gpu_capabilities() {
    # Memory usage safety factor (0.6=conservative, 0.8=aggressive, 0.9=risky)
    MEMORY_SAFETY_FACTOR=${MEMORY_SAFETY_FACTOR:-0.75}
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found"
        exit 1
    fi
    
    # Get actual GPU properties
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    VRAM_GB=$((VRAM_MB / 1024))
    
    # Calculate max matrix size based on VRAM with configurable safety factor
    # Formula: size^2 * 5 (stencil points) * 8 (bytes) + size * 16 (vectors) < 0.85 * VRAM
    # Simplified: size^2 * 40 + size * 16 < available_bytes
    VRAM_BYTES=$(echo "$VRAM_GB * 1024^3 * $MEMORY_SAFETY_FACTOR" | bc -l)
    
    # Solve quadratic: 40*size^2 + 16*size - VRAM_BYTES = 0
    # size = (-16 + sqrt(16^2 + 4*40*VRAM_BYTES)) / (2*40)
    DISCRIMINANT=$(echo "256 + 160 * $VRAM_BYTES" | bc -l)
    MAX_SIZE_EXACT=$(echo "(-16 + sqrt($DISCRIMINANT)) / 80" | bc -l)
    MAX_SIZE=$(echo "$MAX_SIZE_EXACT" | cut -d. -f1)
    
    # Check disk space constraint
    AVAILABLE_GB=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
    DISK_SAFETY_FACTOR=0.8
    DISK_AVAILABLE_BYTES=$(echo "$AVAILABLE_GB * 1024^3 * $DISK_SAFETY_FACTOR" | bc -l)
    
    # Max size based on disk: size^2 * 5 * 20 < available_disk_bytes  
    DISK_MAX_SIZE_EXACT=$(echo "sqrt($DISK_AVAILABLE_BYTES / 100)" | bc -l)
    DISK_MAX_SIZE=$(echo "$DISK_MAX_SIZE_EXACT" | cut -d. -f1)
    
    # Take the minimum of VRAM and disk constraints
    if (( $(echo "$DISK_MAX_SIZE < $MAX_SIZE" | bc -l) )); then
        MAX_SIZE=$DISK_MAX_SIZE
        MAX_SIZE_EXACT=$DISK_MAX_SIZE_EXACT
        LIMITING_FACTOR="disk space"
    else
        LIMITING_FACTOR="GPU memory"
    fi
    
    # Round to nearest 100 for clean numbers (9347 → 9300)
    MAX_SIZE_CLEAN=$(( (MAX_SIZE / 100) * 100 ))
    
    echo "GPU: $GPU_NAME"
    echo "VRAM: ${VRAM_GB}GB (safety factor: $MEMORY_SAFETY_FACTOR)"
    echo "Disk space: ${AVAILABLE_GB}GB (safety factor: $DISK_SAFETY_FACTOR)"
    echo "Max matrix size: ${MAX_SIZE_CLEAN}x${MAX_SIZE_CLEAN} (limited by $LIMITING_FACTOR)"
    
    # Generate test sizes (logarithmic scaling up to optimal size)
    SIZES=(256 512 1024 2048 4096)
    if [ "$MAX_SIZE_CLEAN" -gt 4096 ]; then
        SIZES+=(8192)
    fi
    if [ "$MAX_SIZE_CLEAN" -gt 8192 ]; then
        SIZES+=($MAX_SIZE_CLEAN)
    fi
    
    echo "Test sizes: ${SIZES[*]}"
    
    # Export for other scripts
    cat > /tmp/gpu_config.env << EOF
export GPU_NAME="$GPU_NAME"
export VRAM_GB=$VRAM_GB
export MAX_MATRIX_SIZE=$MAX_SIZE_CLEAN
export MATRIX_SIZES="${SIZES[*]}"
EOF
    
    echo "Config exported to /tmp/gpu_config.env"
}

detect_gpu_capabilities