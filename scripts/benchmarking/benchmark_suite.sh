#!/bin/bash
# Comprehensive benchmark suite for SpMV and CG solvers
# Usage: ./scripts/benchmarking/benchmark_suite.sh <matrix.mtx> [output_dir]

set -e

MATRIX_FILE="$1"
OUTPUT_DIR="${2:-results/benchmark_$(date +%Y%m%d_%H%M%S)}"

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
echo "Benchmark Suite - A100 8×GPU"
echo "=========================================="
echo "Matrix: $MATRIX_FILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# 1. SpMV single-GPU
echo ""
echo "1. SpMV Benchmark (Single-GPU)"
for mode in csr stencil5-csr-direct; do
    echo "  - $mode"
    ./bin/spmv_bench "$MATRIX_FILE" --mode="$mode" \
        --output-format=json --output-file="$OUTPUT_DIR/spmv_${MATRIX_NAME}_${mode}.json"
    ./bin/spmv_bench "$MATRIX_FILE" --mode="$mode" \
        --output-format=csv --output-file="$OUTPUT_DIR/spmv_${MATRIX_NAME}_${mode}.csv"
done

# 2. CG single-GPU
echo ""
echo "2. CG Solver (Single-GPU)"
for mode in csr stencil5-csr-direct; do
    echo "  - $mode"
    ./bin/cg_solver "$MATRIX_FILE" --mode="$mode" \
        --json="$OUTPUT_DIR/cg_${MATRIX_NAME}_${mode}" \
        --csv="$OUTPUT_DIR/cg_${MATRIX_NAME}_single.csv"
done

echo "  - stencil5-csr-direct (detailed timers)"
./bin/cg_solver "$MATRIX_FILE" --mode=stencil5-csr-direct --timers \
    --json="$OUTPUT_DIR/cg_${MATRIX_NAME}_stencil5-csr-direct_timers"

# 3. CG multi-GPU AllGather
echo ""
echo "3. CG Multi-GPU (AllGather)"
if [ -f "./bin/cg_solver_mgpu" ]; then
    for ngpu in 2 4 8; do
        echo "  - $ngpu GPUs"
        mpirun -np "$ngpu" ./bin/cg_solver_mgpu "$MATRIX_FILE" \
            --json="$OUTPUT_DIR/cg_${MATRIX_NAME}_mgpu_allgather_${ngpu}gpu" \
            --csv="$OUTPUT_DIR/cg_${MATRIX_NAME}_mgpu_allgather.csv"
    done
fi

# 4. CG multi-GPU Halo P2P
echo ""
echo "4. CG Multi-GPU (Halo P2P)"
if [ -f "./bin/cg_solver_mgpu_stencil" ]; then
    for ngpu in 2 4 8; do
        echo "  - $ngpu GPUs"
        mpirun -np "$ngpu" ./bin/cg_solver_mgpu_stencil "$MATRIX_FILE" \
            --json="$OUTPUT_DIR/cg_${MATRIX_NAME}_mgpu_halo_${ngpu}gpu" \
            --csv="$OUTPUT_DIR/cg_${MATRIX_NAME}_mgpu_halo.csv"
    done

    echo "  - 8 GPUs (detailed timers)"
    mpirun -np 8 ./bin/cg_solver_mgpu_stencil "$MATRIX_FILE" --timers \
        --json="$OUTPUT_DIR/cg_${MATRIX_NAME}_mgpu_halo_8gpu_timers"
fi

echo ""
echo "=========================================="
echo "✓ Benchmark completed"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
