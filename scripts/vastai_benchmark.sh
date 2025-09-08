#!/bin/bash

# VastAI Automated Benchmark Script
# Auto-detects GPU, generates optimal matrix size, runs complete benchmark suite

set -e

echo "🚀 VastAI GPU Benchmark Pipeline"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "bin/spmv_bench" ] || [ ! -f "scripts/benchmark_and_visualize.sh" ]; then
    echo "❌ Not in project root or binaries not built"
    echo "Run from project root after 'make'"
    exit 1
fi

# Step 1: GPU Detection and Configuration
echo -e "${BLUE}🔍 Step 1: GPU Detection & Sizing${NC}"
echo "Detecting GPU configuration..."
./scripts/detect_gpu_config.sh

if [ ! -f "/tmp/gpu_config.env" ]; then
    echo "❌ GPU detection failed"
    exit 1
fi

source /tmp/gpu_config.env
echo -e "${GREEN}✅ GPU detected: $GPU_NAME${NC}"
echo -e "${GREEN}✅ Optimal matrix size: ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE}${NC}"

# Step 2: Matrix Generation
MATRIX_FILE="matrix/vastai_${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE}.mtx"
echo ""
echo -e "${BLUE}🔧 Step 2: Matrix Generation${NC}"

if [ -f "$MATRIX_FILE" ]; then
    echo "Matrix file already exists: $MATRIX_FILE"
    MATRIX_SIZE_MB=$(du -m "$MATRIX_FILE" | cut -f1)
    echo "Size: ${MATRIX_SIZE_MB}MB"
else
    echo "Generating ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE} stencil matrix..."
    echo "This may take 1-3 minutes for large matrices..."
    ./bin/generate_matrix $MAX_MATRIX_SIZE "$MATRIX_FILE"
    
    if [ $? -eq 0 ]; then
        MATRIX_SIZE_MB=$(du -m "$MATRIX_FILE" | cut -f1)
        echo -e "${GREEN}✅ Matrix generated successfully: ${MATRIX_SIZE_MB}MB${NC}"
    else
        echo "❌ Matrix generation failed"
        exit 1
    fi
fi

# Step 3: Comprehensive Benchmark
echo ""
echo -e "${BLUE}🏁 Step 3: Comprehensive Benchmark${NC}"
echo "Running full SpMV benchmark suite with all 6 kernels..."

# Generate result prefix with GPU name and timestamp
RESULT_PREFIX="vastai_$(echo "$GPU_NAME" | tr ' -' '_' | tr '[:upper:]' '[:lower:]')_$(date +%Y%m%d_%H%M)"

echo "Result prefix: $RESULT_PREFIX"
echo "Matrix: $MATRIX_FILE"
echo ""

# Run the complete benchmark pipeline
./scripts/benchmark_and_visualize.sh "$MATRIX_FILE" "$RESULT_PREFIX"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 VastAI Benchmark Pipeline Completed Successfully!${NC}"
    echo ""
    echo -e "${YELLOW}📊 Results Summary:${NC}"
    echo "- Matrix: ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE} (${MATRIX_SIZE_MB}MB)"
    echo "- GPU: $GPU_NAME"
    echo "- Results: results/${RESULT_PREFIX}*"
    echo ""
    echo -e "${YELLOW}📁 Generated Files:${NC}"
    ls -la results/${RESULT_PREFIX}*
    echo ""
    echo -e "${YELLOW}⬇️ Download Instructions:${NC}"
    echo "Use 'scp' or VastAI file sync to download results/ directory"
    echo "Key files: CSV data, JSON metrics, PNG charts"
else
    echo "❌ Benchmark pipeline failed"
    exit 1
fi