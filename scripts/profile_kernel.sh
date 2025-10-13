#!/bin/bash
# Nsight Compute profiling script for SpMV kernels
# Usage: ./scripts/profile_kernel.sh <mode> <matrix_file> [output_dir]

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <mode> <matrix_file> [output_dir]"
    echo "Example: $0 stencil5-opt matrix/stencil_512x512.mtx"
    echo ""
    echo "Available modes: csr, stencil5, stencil5-opt, stencil5-shared, etc."
    exit 1
fi

MODE=$1
MATRIX_FILE=$2
OUTPUT_DIR=${3:-"profiling_results"}

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "Error: ncu (Nsight Compute) not found in PATH"
    echo "Install from: https://developer.nvidia.com/nsight-compute"
    exit 1
fi

# Check if spmv_bench exists
if [ ! -f "bin/spmv_bench" ]; then
    echo "Error: bin/spmv_bench not found. Run 'make' first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_BASE="${OUTPUT_DIR}/${MODE}_${TIMESTAMP}"

echo "=========================================="
echo "Nsight Compute Profiling"
echo "=========================================="
echo "Mode:         $MODE"
echo "Matrix:       $MATRIX_FILE"
echo "Output:       ${REPORT_BASE}"
echo ""

echo "Running Nsight Compute (this may take 30-60 seconds)..."
echo ""

# Run ncu with full metric set for comprehensive analysis
ncu \
    --set full \
    --export "${REPORT_BASE}" \
    --force-overwrite \
    --print-summary per-kernel \
    ./bin/spmv_bench "$MATRIX_FILE" --mode="$MODE" \
    2>&1 | tee "${REPORT_BASE}.log"

echo ""
echo "=========================================="
echo "Profiling Complete"
echo "=========================================="
echo "Report files:"
echo "  - ${REPORT_BASE}.ncu-rep (GUI: ncu-ui)"
echo "  - ${REPORT_BASE}.log (text summary)"
echo ""
echo "View with Nsight Compute UI:"
echo "  ncu-ui ${REPORT_BASE}.ncu-rep"
echo ""
echo "Key metrics to analyze:"
echo "  - SM Throughput: GPU compute utilization"
echo "  - DRAM Throughput: Memory bandwidth usage"
echo "  - Memory Efficiency: Cache hit rates"
echo "  - Occupancy: Warp/thread utilization"
echo "=========================================="
