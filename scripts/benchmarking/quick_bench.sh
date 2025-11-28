#!/bin/bash
# Quick benchmark for local testing (single-GPU + multi-GPU)
# Usage: ./scripts/benchmarking/quick_bench.sh <matrix.mtx> [output_dir] [num_gpus]

set -e

MATRIX_FILE="$1"
OUTPUT_DIR="${2:-results/quick_$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${3:-1}"

if [ -z "$MATRIX_FILE" ]; then
    echo "Usage: $0 <matrix.mtx> [output_dir] [num_gpus]"
    exit 1
fi

if [ ! -f "$MATRIX_FILE" ]; then
    echo "Error: Matrix file '$MATRIX_FILE' not found"
    exit 1
fi

MATRIX_NAME=$(basename "$MATRIX_FILE" .mtx)
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Quick Benchmark ($NUM_GPUS GPU(s))"
echo "=========================================="
echo "Matrix: $MATRIX_FILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# SpMV: Compare key modes (single-GPU)
echo "1. SpMV Benchmark (Single-GPU)"
echo "---"
./bin/spmv_bench "$MATRIX_FILE" --mode=csr \
    --csv="$OUTPUT_DIR/spmv_${MATRIX_NAME}_csr.csv"

./bin/spmv_bench "$MATRIX_FILE" --mode=stencil5-csr-direct \
    --csv="$OUTPUT_DIR/spmv_${MATRIX_NAME}_stencil5.csv"

echo ""
echo "2. CG Solver Benchmark (Single-GPU)"
echo "---"
./bin/cg_solver "$MATRIX_FILE" --mode=csr,stencil5-csr-direct \
    --json="$OUTPUT_DIR/cg_${MATRIX_NAME}_single.json" \
    --csv="$OUTPUT_DIR/cg_${MATRIX_NAME}_single.csv"

# Multi-GPU if requested
if [ "$NUM_GPUS" -gt 1 ]; then
    echo ""
    echo "3. CG Solver Benchmark (Multi-GPU: $NUM_GPUS GPUs)"
    echo "---"
    mpirun --allow-run-as-root -np "$NUM_GPUS" \
        ./bin/cg_solver_mgpu_stencil "$MATRIX_FILE" \
        --json="$OUTPUT_DIR/cg_${MATRIX_NAME}_mgpu.json"
fi

echo ""
echo "=========================================="
echo "✓ Quick benchmark completed"
echo "Results: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
echo "=========================================="
