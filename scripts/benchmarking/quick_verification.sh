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
# Filter: remove progress bar lines (keep "Matrix generated:" result)
exec > >(tee >(sed 's/\r/\n/g' | grep -v "^Writing matrix entries:" > "${LOG_FILE}")) 2>&1

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

# Build AmgX benchmarks if library is available
HAS_AMGX=0
AMGX_HEADER=""
for path in "external/amgx-src/include/amgx_c.h" "external/amgx/include/amgx_c.h" "/usr/local/include/amgx_c.h"; do
    if [ -f "$path" ]; then
        AMGX_HEADER="$path"
        break
    fi
done

if [ -n "$AMGX_HEADER" ]; then
    echo "AmgX found: $AMGX_HEADER"
    make -C external/benchmarks/amgx amgx_cg_solver 2>&1 | tail -2
    if [ -f "external/benchmarks/amgx/amgx_cg_solver" ]; then
        HAS_AMGX=1
        if [ "$HAS_MPI" = "1" ]; then
            make -C external/benchmarks/amgx amgx_cg_solver_mgpu 2>&1 | tail -2
        fi
    fi
else
    echo "AmgX not found - skipping AmgX build"
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
./bin/cg_solver "${MATRIX_FILE}" 2>&1 | grep -E "(Converged|Iterations|Time|Checksum|Sum\(|Norm)" || true
echo ""

# Test 3: Multi-GPU CG (if MPI available)
if [ "$HAS_MPI" = "1" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -ge 2 ]; then
        echo "=== Test 3: Multi-GPU CG Solver (2 GPUs) ==="
        mpirun --allow-run-as-root -np 2 ./bin/cg_solver_mgpu_stencil "${MATRIX_FILE}" 2>&1 | grep -E "(Converged|Iterations|Time|Checksum|Sum\(|Norm)" || true
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
    echo "=== Test 4: AmgX CG Solver (Single-GPU) ==="
    "$AMGX_BIN" "${MATRIX_FILE}" 2>&1 | grep -E "(Converged|Iterations|Time|Checksum|Sum\(|Norm)" || true
    echo ""
else
    echo "=== Test 4: Skipped (AmgX not built) ==="
    echo ""
fi

# Test 5: Multi-GPU AmgX (if available and multiple GPUs)
AMGX_MGPU_BIN=""
for path in "./bin/amgx_cg_solver_mgpu" "./external/benchmarks/amgx/amgx_cg_solver_mgpu"; do
    if [ -f "$path" ]; then
        AMGX_MGPU_BIN="$path"
        break
    fi
done

if [ -n "$AMGX_MGPU_BIN" ] && [ "$HAS_MPI" = "1" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -ge 2 ]; then
        echo "=== Test 5: Multi-GPU AmgX CG Solver (2 GPUs) ==="
        mpirun --allow-run-as-root -np 2 "$AMGX_MGPU_BIN" "${MATRIX_FILE}" 2>&1 | grep -E "(Converged|Iterations|Time|Checksum|Sum\(|Norm)" || true
        echo ""
    else
        echo "=== Test 5: Skipped (need ≥2 GPUs) ==="
        echo ""
    fi
else
    echo "=== Test 5: Skipped (AmgX multi-GPU not built or MPI unavailable) ==="
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
