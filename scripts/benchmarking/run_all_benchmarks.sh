#!/bin/bash
# Master benchmark script - runs all benchmark suites and collects results
# Generates reproducible results with full environment information

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_full_benchmark_${TIMESTAMP}"

echo "=============================================="
echo "Full Benchmark Suite"
echo "Started: $(date)"
echo "Results: ${RESULTS_DIR}/"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"

# Capture environment info
echo ""
echo "=== Capturing Environment Info ==="
{
    echo "=== Environment Info ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo ""
    echo "=== CUDA ==="
    nvcc --version 2>/dev/null || echo "nvcc not found"
    echo ""
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "nvidia-smi not available"
    echo ""
    echo "=== MPI ==="
    mpirun --version 2>/dev/null || echo "mpirun not found"
    echo ""
    echo "=== Compilation Flags ==="
    grep "NVCCFLAGS :=" "${SCRIPT_DIR}/../../Makefile" | grep -v debug || echo "Could not read Makefile"
} > "${RESULTS_DIR}/environment.txt"
cat "${RESULTS_DIR}/environment.txt"

# Build if needed
echo ""
echo "=== Building Binaries ==="
make -C "${SCRIPT_DIR}/../.." cg_solver_mgpu_stencil generate_matrix spmv_bench 2>&1 | tail -5

# Run benchmarks
echo ""
echo "=== Running Single-GPU SpMV Benchmark ==="
if [ -f "${SCRIPT_DIR}/benchmark_single_gpu_formats.sh" ]; then
    bash "${SCRIPT_DIR}/benchmark_single_gpu_formats.sh" 2>&1 | tee "${RESULTS_DIR}/spmv_log.txt" || echo "SpMV benchmark skipped (no GPU)"
fi

echo ""
echo "=== Running Multi-GPU Scaling Benchmark ==="
if [ -f "${SCRIPT_DIR}/benchmark_problem_sizes.sh" ]; then
    bash "${SCRIPT_DIR}/benchmark_problem_sizes.sh" 2>&1 | tee "${RESULTS_DIR}/scaling_log.txt" || echo "Scaling benchmark skipped"
fi

echo ""
echo "=== Running AmgX Comparison ==="
if [ -f "${SCRIPT_DIR}/benchmark_amgx.sh" ]; then
    bash "${SCRIPT_DIR}/benchmark_amgx.sh" 2>&1 | tee "${RESULTS_DIR}/amgx_log.txt" || echo "AmgX benchmark skipped"
fi

# Collect results
echo ""
echo "=== Collecting Results ==="
for dir in results_*; do
    if [ -d "$dir" ] && [ "$dir" != "${RESULTS_DIR}" ]; then
        cp -r "$dir" "${RESULTS_DIR}/" 2>/dev/null || true
    fi
done

echo ""
echo "=============================================="
echo "Benchmark Complete"
echo "Results saved to: ${RESULTS_DIR}/"
echo "Finished: $(date)"
echo "=============================================="
