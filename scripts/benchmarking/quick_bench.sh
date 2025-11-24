#!/bin/bash
# Quick benchmark for local testing (single-GPU only)
# Usage: ./scripts/benchmarking/quick_bench.sh <matrix.mtx> [output_dir]

set -e

MATRIX_FILE="$1"
OUTPUT_DIR="${2:-results/quick_$(date +%Y%m%d_%H%M%S)}"

if [ -z "$MATRIX_FILE" ]; then
    echo "Usage: $0 <matrix.mtx> [output_dir]"
    exit 1
fi

if [ ! -f "$MATRIX_FILE" ]; then
    echo "Error: Matrix file '$MATRIX_FILE' not found"
    exit 1
fi

MATRIX_NAME=$(basename "$MATRIX_FILE" .mtx)
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Quick Benchmark (Single-GPU)"
echo "=========================================="
echo "Matrix: $MATRIX_FILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# SpMV: Compare key modes
echo "1. SpMV Benchmark"
echo "---"
./bin/spmv_bench "$MATRIX_FILE" --mode=csr \
    --output-format=csv --output-file="$OUTPUT_DIR/spmv_${MATRIX_NAME}_csr.csv"

./bin/spmv_bench "$MATRIX_FILE" --mode=stencil5-csr-direct \
    --output-format=csv --output-file="$OUTPUT_DIR/spmv_${MATRIX_NAME}_stencil5.csv"

echo ""
echo "2. CG Solver Benchmark"
echo "---"
./bin/cg_solver "$MATRIX_FILE" --mode=csr,stencil5-csr-direct \
    --json="$OUTPUT_DIR/cg_${MATRIX_NAME}" \
    --csv="$OUTPUT_DIR/cg_${MATRIX_NAME}.csv"

echo ""
echo "=========================================="
echo "✓ Quick benchmark completed"
echo "Results: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
echo "=========================================="
