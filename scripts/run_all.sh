#!/bin/bash
# =============================================================================
# run_all.sh - Reproduce all benchmark results
# =============================================================================
# ONE COMMAND TO RULE THEM ALL
#
# Usage:
#   ./scripts/run_all.sh              # Full benchmarks (default: 5000x5000)
#   ./scripts/run_all.sh --quick      # Quick verification (512x512)
#   ./scripts/run_all.sh --size=10000 # Custom matrix size
#
# Output:
#   results/raw/         - Raw benchmark outputs (TXT)
#   results/json/        - Structured results (JSON)
#
# Requirements:
#   - CUDA toolkit (nvcc)
#   - MPI (optional, for multi-GPU)
#   - AmgX (optional, for comparison)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
cd "${PROJECT_DIR}"

# =============================================================================
# Configuration
# =============================================================================
MATRIX_SIZE=5000
QUICK_MODE=0
NUM_RUNS=10

# Parse arguments
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=1
            MATRIX_SIZE=512
            NUM_RUNS=3
            ;;
        --size=*)
            MATRIX_SIZE="${arg#*=}"
            ;;
        --help|-h)
            head -25 "$0" | tail -20
            exit 0
            ;;
    esac
done

# Paths
MATRIX_FILE="matrix/stencil_${MATRIX_SIZE}x${MATRIX_SIZE}.mtx"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_RAW="results/raw"
RESULTS_JSON="results/json"

mkdir -p "${RESULTS_RAW}" "${RESULTS_JSON}"

# =============================================================================
# Environment Detection
# =============================================================================
echo "=============================================="
echo "CUDA SpMV Benchmark Suite"
echo "=============================================="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Runs:        ${NUM_RUNS}"
echo ""

# GPU info
echo "=== GPU Configuration ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "Total GPUs:  ${NUM_GPUS}"
echo ""

# Check dependencies
HAS_MPI=0
HAS_AMGX=0

if command -v mpic++ &> /dev/null; then
    HAS_MPI=1
    echo "MPI:         $(mpirun --version 2>&1 | head -1)"
fi

for path in "external/amgx-src/include/amgx_c.h" "external/amgx/include/amgx_c.h"; do
    if [ -f "$path" ]; then
        HAS_AMGX=1
        echo "AmgX:        Found ($path)"
        break
    fi
done
echo ""

# =============================================================================
# Build
# =============================================================================
echo "=== Building ==="
make -j$(nproc) spmv_bench generate_matrix cg_solver 2>&1 | tail -3

if [ "$HAS_MPI" = "1" ]; then
    make -j$(nproc) cg_solver_mgpu_stencil 2>&1 | tail -2
fi

if [ "$HAS_AMGX" = "1" ]; then
    make -C external/benchmarks/amgx -j$(nproc) 2>&1 | tail -3
fi
echo ""

# =============================================================================
# Generate Matrix
# =============================================================================
echo "=== Generating Matrix ==="
if [ ! -f "${MATRIX_FILE}" ]; then
    ./bin/generate_matrix ${MATRIX_SIZE} "${MATRIX_FILE}" 2>&1 | grep -v "^Writing"
else
    echo "Matrix exists: ${MATRIX_FILE}"
    ls -lh "${MATRIX_FILE}"
fi
echo ""

# =============================================================================
# Benchmark 1: SpMV (Single-GPU)
# =============================================================================
SPMV_OUT="${RESULTS_RAW}/spmv_${MATRIX_SIZE}_${TIMESTAMP}.txt"
SPMV_JSON="${RESULTS_JSON}/spmv_${MATRIX_SIZE}.json"
echo "=== Benchmark 1: SpMV (Single-GPU) ==="
echo "Output: ${SPMV_OUT}"

./bin/spmv_bench "${MATRIX_FILE}" --mode=cusparse-csr,stencil5-csr --json="${SPMV_JSON}" 2>&1 | tee "${SPMV_OUT}"

echo ""
echo ""

# =============================================================================
# Benchmark 2: CG Solver (Single-GPU)
# =============================================================================
CG_OUT="${RESULTS_RAW}/cg_single_${MATRIX_SIZE}_${TIMESTAMP}.txt"
CG_JSON="${RESULTS_JSON}/cg_single_${MATRIX_SIZE}"
echo "=== Benchmark 2: CG Solver (Single-GPU) ==="
echo "Output: ${CG_OUT}"

./bin/cg_solver "${MATRIX_FILE}" --json="${CG_JSON}" 2>&1 | tee "${CG_OUT}"

echo ""
echo ""

# =============================================================================
# Benchmark 3: CG Solver (Multi-GPU)
# =============================================================================
if [ "$HAS_MPI" = "1" ] && [ "$NUM_GPUS" -ge 2 ]; then
    CG_MGPU_OUT="${RESULTS_RAW}/cg_mgpu_${MATRIX_SIZE}_${NUM_GPUS}gpu_${TIMESTAMP}.txt"
    CG_MGPU_JSON="${RESULTS_JSON}/cg_mgpu_${MATRIX_SIZE}_${NUM_GPUS}gpu.json"
    echo "=== Benchmark 3: CG Solver (Multi-GPU, ${NUM_GPUS} GPUs) ==="
    echo "Output: ${CG_MGPU_OUT}"

    mpirun --allow-run-as-root -np ${NUM_GPUS} ./bin/cg_solver_mgpu_stencil "${MATRIX_FILE}" \
        --json="${CG_MGPU_JSON}" 2>&1 | tee "${CG_MGPU_OUT}"
else
    echo "=== Benchmark 3: Skipped (need MPI + ≥2 GPUs) ==="
fi

echo ""
echo ""

# =============================================================================
# Benchmark 4: AmgX (Single-GPU) - Reference
# =============================================================================
if [ "$HAS_AMGX" = "1" ] && [ -f "external/benchmarks/amgx/amgx_cg_solver" ]; then
    AMGX_OUT="${RESULTS_RAW}/amgx_single_${MATRIX_SIZE}_${TIMESTAMP}.txt"
    AMGX_JSON="${RESULTS_JSON}/amgx_single_${MATRIX_SIZE}.json"
    echo "=== Benchmark 4: AmgX CG Solver (Single-GPU, Reference) ==="
    echo "Output: ${AMGX_OUT}"

    ./external/benchmarks/amgx/amgx_cg_solver "${MATRIX_FILE}" \
        --runs=${NUM_RUNS} --json="${AMGX_JSON}" 2>&1 | tee "${AMGX_OUT}"
else
    echo "=== Benchmark 4: Skipped (AmgX not available) ==="
fi

echo ""
echo ""

# =============================================================================
# Benchmark 5: AmgX (Multi-GPU) - Reference
# =============================================================================
if [ "$HAS_AMGX" = "1" ] && [ "$HAS_MPI" = "1" ] && [ "$NUM_GPUS" -ge 2 ] \
   && [ -f "external/benchmarks/amgx/amgx_cg_solver_mgpu" ]; then
    AMGX_MGPU_OUT="${RESULTS_RAW}/amgx_mgpu_${MATRIX_SIZE}_${NUM_GPUS}gpu_${TIMESTAMP}.txt"
    AMGX_MGPU_JSON="${RESULTS_JSON}/amgx_mgpu_${MATRIX_SIZE}_${NUM_GPUS}gpu.json"
    echo "=== Benchmark 5: AmgX CG Solver (Multi-GPU, ${NUM_GPUS} GPUs, Reference) ==="
    echo "Output: ${AMGX_MGPU_OUT}"

    mpirun --allow-run-as-root -np ${NUM_GPUS} ./external/benchmarks/amgx/amgx_cg_solver_mgpu \
        "${MATRIX_FILE}" --runs=${NUM_RUNS} --json="${AMGX_MGPU_JSON}" 2>&1 | tee "${AMGX_MGPU_OUT}"
else
    echo "=== Benchmark 5: Skipped (AmgX multi-GPU not available) ==="
fi

echo ""
echo ""

# =============================================================================
# Summary Table
# =============================================================================
echo "=============================================="
echo "PERFORMANCE SUMMARY"
echo "=============================================="
echo ""
printf "%-35s %12s %12s\n" "Benchmark" "Time (ms)" "Speedup"
printf "%-35s %12s %12s\n" "-----------------------------------" "------------" "------------"

# Extract SpMV results
CUSPARSE_TIME=""
STENCIL_TIME=""
if [ -f "${RESULTS_JSON}/spmv_${MATRIX_SIZE}_cusparse-csr.json" ]; then
    CUSPARSE_TIME=$(grep -o '"execution_time_ms": [0-9.]*' "${RESULTS_JSON}/spmv_${MATRIX_SIZE}_cusparse-csr.json" | head -1 | cut -d' ' -f2)
fi
if [ -f "${RESULTS_JSON}/spmv_${MATRIX_SIZE}_stencil5-csr.json" ]; then
    STENCIL_TIME=$(grep -o '"execution_time_ms": [0-9.]*' "${RESULTS_JSON}/spmv_${MATRIX_SIZE}_stencil5-csr.json" | head -1 | cut -d' ' -f2)
fi
if [ -n "$CUSPARSE_TIME" ]; then
    printf "%-35s %12.3f %12s\n" "SpMV cuSPARSE CSR" "$CUSPARSE_TIME" "(baseline)"
fi
if [ -n "$STENCIL_TIME" ] && [ -n "$CUSPARSE_TIME" ]; then
    SPMV_SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $CUSPARSE_TIME / $STENCIL_TIME}")
    printf "%-35s %12.3f %12s\n" "SpMV Stencil CSR" "$STENCIL_TIME" "${SPMV_SPEEDUP}x"
elif [ -n "$STENCIL_TIME" ]; then
    printf "%-35s %12.3f %12s\n" "SpMV Stencil CSR" "$STENCIL_TIME" "-"
fi

echo ""

# Extract CG results
CG_SINGLE_TIME=""
CG_MGPU_TIME=""
AMGX_SINGLE_TIME=""
AMGX_MGPU_TIME=""

for json in "${RESULTS_JSON}"/cg_single_${MATRIX_SIZE}*.json; do
    if [ -f "$json" ]; then
        CG_SINGLE_TIME=$(grep -o '"total_time_ms": [0-9.]*' "$json" | head -1 | cut -d' ' -f2)
        break
    fi
done
if [ -f "${RESULTS_JSON}/cg_mgpu_${MATRIX_SIZE}_${NUM_GPUS}gpu.json" ]; then
    CG_MGPU_TIME=$(grep -o '"total_time_ms": [0-9.]*' "${RESULTS_JSON}/cg_mgpu_${MATRIX_SIZE}_${NUM_GPUS}gpu.json" | head -1 | cut -d' ' -f2)
fi
if [ -f "${RESULTS_JSON}/amgx_single_${MATRIX_SIZE}.json" ]; then
    AMGX_SINGLE_TIME=$(grep -o '"total_time_ms": [0-9.]*' "${RESULTS_JSON}/amgx_single_${MATRIX_SIZE}.json" | head -1 | cut -d' ' -f2)
fi
if [ -f "${RESULTS_JSON}/amgx_mgpu_${MATRIX_SIZE}_${NUM_GPUS}gpu.json" ]; then
    AMGX_MGPU_TIME=$(grep -o '"total_time_ms": [0-9.]*' "${RESULTS_JSON}/amgx_mgpu_${MATRIX_SIZE}_${NUM_GPUS}gpu.json" | head -1 | cut -d' ' -f2)
fi

if [ -n "$CG_SINGLE_TIME" ]; then
    printf "%-35s %12.3f %12s\n" "CG Custom (1 GPU)" "$CG_SINGLE_TIME" "-"
fi
if [ -n "$AMGX_SINGLE_TIME" ] && [ -n "$CG_SINGLE_TIME" ]; then
    CG_VS_AMGX=$(awk "BEGIN {printf \"%.2f\", $AMGX_SINGLE_TIME / $CG_SINGLE_TIME}")
    printf "%-35s %12.3f %12s\n" "CG AmgX (1 GPU)" "$AMGX_SINGLE_TIME" "Custom ${CG_VS_AMGX}x faster"
elif [ -n "$AMGX_SINGLE_TIME" ]; then
    printf "%-35s %12.3f %12s\n" "CG AmgX (1 GPU)" "$AMGX_SINGLE_TIME" "-"
fi

echo ""

if [ -n "$CG_MGPU_TIME" ]; then
    if [ -n "$CG_SINGLE_TIME" ]; then
        MGPU_SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $CG_SINGLE_TIME / $CG_MGPU_TIME}")
        printf "%-35s %12.3f %12s\n" "CG Custom (${NUM_GPUS} GPUs)" "$CG_MGPU_TIME" "${MGPU_SPEEDUP}x vs 1 GPU"
    else
        printf "%-35s %12.3f %12s\n" "CG Custom (${NUM_GPUS} GPUs)" "$CG_MGPU_TIME" "-"
    fi
fi
if [ -n "$AMGX_MGPU_TIME" ] && [ -n "$CG_MGPU_TIME" ]; then
    MGPU_VS_AMGX=$(awk "BEGIN {printf \"%.2f\", $AMGX_MGPU_TIME / $CG_MGPU_TIME}")
    printf "%-35s %12.3f %12s\n" "CG AmgX (${NUM_GPUS} GPUs)" "$AMGX_MGPU_TIME" "Custom ${MGPU_VS_AMGX}x faster"
elif [ -n "$AMGX_MGPU_TIME" ]; then
    printf "%-35s %12.3f %12s\n" "CG AmgX (${NUM_GPUS} GPUs)" "$AMGX_MGPU_TIME" "-"
fi

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  Raw outputs: ${RESULTS_RAW}/"
echo "  JSON data:   ${RESULTS_JSON}/"
echo ""
