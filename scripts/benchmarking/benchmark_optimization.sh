#!/bin/bash
# Script to benchmark an optimization and compare with baseline
#
# Usage: ./scripts/benchmark_optimization.sh opt01_fused_reduction

set -e

OPT_NAME=$1
if [ -z "$OPT_NAME" ]; then
    echo "Usage: $0 <optimization_name>"
    echo "Example: $0 opt01_fused_reduction"
    exit 1
fi

MATRIX="matrix/stencil_5000x5000.mtx"
MODE="stencil5-csr-direct"
BINARY="./bin/release/cg_test"
RESULTS_DIR="profiling_results/${OPT_NAME}"

echo "=========================================="
echo "Benchmarking: ${OPT_NAME}"
echo "=========================================="

# Create results directory
mkdir -p "${RESULTS_DIR}"

# 1. Quick performance run (3 runs for averaging)
echo "[1/4] Running quick performance benchmark (3 runs)..."
for i in {1..3}; do
    echo "  Run $i/3..."
    ${BINARY} ${MATRIX} --mode=${MODE} --device 2>&1 | tee "${RESULTS_DIR}/run_${i}.log"
done

# Extract time-to-solution
echo ""
echo "Time-to-solution across runs:"
grep "Time-to-solution:" "${RESULTS_DIR}"/run_*.log | awk '{print $2, $3}'

# 2. Nsight Systems profiling
echo ""
echo "[2/4] Running Nsight Systems profiling..."
nsys profile \
    -o "${RESULTS_DIR}/nsys_${OPT_NAME}" \
    --stats=true \
    ${BINARY} ${MATRIX} --mode=${MODE} --device \
    2>&1 | tee "${RESULTS_DIR}/nsys_output.txt"

# 3. Nsight Compute profiling (detailed, all kernels, 1 sample each)
echo ""
echo "[3/4] Running Nsight Compute profiling (this will take time)..."
ncu --set detailed \
    --launch-count 1 \
    -o "${RESULTS_DIR}/ncu_${OPT_NAME}" \
    ${BINARY} ${MATRIX} --mode=${MODE} --device \
    2>&1 | tee "${RESULTS_DIR}/ncu_output.txt"

# 4. Generate comparison report
echo ""
echo "[4/4] Generating comparison report..."
cat > "${RESULTS_DIR}/comparison.md" <<EOF
# Optimization: ${OPT_NAME}

**Date**: $(date +%Y-%m-%d)
**Branch**: $(git branch --show-current)
**Commit**: $(git rev-parse --short HEAD)

## Performance Summary

### Time-to-solution
\`\`\`
$(grep "Time-to-solution:" "${RESULTS_DIR}"/run_*.log | awk '{print $2, $3}')
\`\`\`

**Average**: TBD (calculate manually)

### Comparison with Baseline
| Metric | Baseline | ${OPT_NAME} | Speedup |
|--------|----------|-------------|---------|
| Time-to-solution | 356 ms | TBD | TBD |
| SpMV | 105 ms | TBD | TBD |
| BLAS1 | 130 ms | TBD | TBD |
| Reductions | 118 ms | TBD | TBD |

## Profiling Files
- Nsight Systems: \`${RESULTS_DIR}/nsys_${OPT_NAME}.nsys-rep\`
- Nsight Compute: \`${RESULTS_DIR}/ncu_${OPT_NAME}.ncu-rep\`
- Logs: \`${RESULTS_DIR}/run_*.log\`

## Analysis

### Kernel Time Breakdown (from Nsight Systems)
\`\`\`
$(grep "Executing 'cuda_gpu_kern_sum'" "${RESULTS_DIR}/nsys_output.txt" -A 50 | head -30)
\`\`\`

## Next Steps
- [ ] Analyze detailed NCU metrics
- [ ] Compare with baseline in ncu-ui
- [ ] Update OPTIMIZATION_LOG.md
- [ ] Decision: merge, iterate, or abandon

EOF

echo ""
echo "=========================================="
echo "Benchmarking complete!"
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review results: cat ${RESULTS_DIR}/comparison.md"
echo "  2. Open NCU UI: ncu-ui ${RESULTS_DIR}/ncu_${OPT_NAME}.ncu-rep"
echo "  3. Update: profiling_results/OPTIMIZATION_LOG.md"
echo "=========================================="
