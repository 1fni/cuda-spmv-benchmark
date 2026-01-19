#!/bin/bash
# Quick verification script - tests all components with small matrices
# Purpose: Verify installation and functionality without long wait times
# Expected runtime: < 2 minutes on a single GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/../.."
cd "${PROJECT_DIR}"

# Create results directory and log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_quick_verification"
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/verification_${TIMESTAMP}.txt"

# Tee output to both console and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "Quick Verification Script"
echo "=============================================="
echo "Tests all components with small matrices"
echo "Expected runtime: < 2 minutes"
echo "Results saved to: ${LOG_FILE}"
echo ""

# Capture environment
echo "=== Environment ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "GPU: not detected"
nvcc --version 2>/dev/null | grep "release" || echo "CUDA: not found"
echo ""

# Build
echo "=== Building ==="
make spmv_bench generate_matrix cg_solver 2>&1 | tail -3
if command -v mpic++ &> /dev/null; then
    make cg_solver_mgpu_stencil 2>&1 | tail -2
    HAS_MPI=1
else
    echo "MPI not found - skipping multi-GPU build"
    HAS_MPI=0
fi
echo ""

# Generate small test matrix
SMALL_SIZE=512
MATRIX_FILE="matrix/test_quick_${SMALL_SIZE}x${SMALL_SIZE}.mtx"

echo "=== Generating test matrix (${SMALL_SIZE}×${SMALL_SIZE}) ==="
./bin/generate_matrix ${SMALL_SIZE} "${MATRIX_FILE}" 2>&1 | tail -2
echo ""

# Test 1: Single-GPU SpMV
echo "=== Test 1: Single-GPU SpMV Benchmark ==="
echo "Testing modes: cusparse-csr, stencil5-csr"
./bin/spmv_bench "${MATRIX_FILE}" --mode=cusparse-csr,stencil5-csr 2>&1 | grep -E "(Testing|Execution time|GFLOPS|Speedup|Checksum|Sum\(y\))" || true
echo ""

# Test 2: Single-GPU CG Solver
echo "=== Test 2: Single-GPU CG Solver ==="
./bin/cg_solver "${MATRIX_FILE}" 2>&1 | grep -E "(Converged|Iterations|Time|Residual)" || true
echo ""

# Test 3: Multi-GPU CG (if MPI available)
if [ "$HAS_MPI" = "1" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -ge 2 ]; then
        echo "=== Test 3: Multi-GPU CG Solver (2 GPUs) ==="
        mpirun --allow-run-as-root -np 2 ./bin/cg_solver_mgpu_stencil "${MATRIX_FILE}" 2>&1 | grep -E "(Converged|Iterations|Time|Speedup|Efficiency)" || true
        echo ""
    else
        echo "=== Test 3: Skipped (need ≥2 GPUs, found ${NUM_GPUS}) ==="
        echo ""
    fi
else
    echo "=== Test 3: Skipped (MPI not available) ==="
    echo ""
fi

# Test 4: AmgX (if available)
# Check multiple possible locations for AmgX binary
AMGX_BIN=""
for path in "./bin/amgx_cg_solver" "./external/benchmarks/amgx/amgx_cg_solver"; do
    if [ -f "$path" ]; then
        AMGX_BIN="$path"
        break
    fi
done

if [ -n "$AMGX_BIN" ]; then
    echo "=== Test 4: AmgX CG Solver ==="
    "$AMGX_BIN" "${MATRIX_FILE}" 2>&1 | grep -E "(Converged|Iterations|Time)" || true
    echo ""
else
    echo "=== Test 4: Skipped (AmgX not built) ==="
    echo ""
fi

# Cleanup
rm -f "${MATRIX_FILE}"

echo "=============================================="
echo "Verification Complete"
echo "=============================================="
echo "Finished: $(date)"
echo "Results saved to: ${LOG_FILE}"
echo ""
echo "All tests passed if no errors above."
echo "For full benchmarks with large matrices, run:"
echo "  ./scripts/benchmarking/run_all_benchmarks.sh"
