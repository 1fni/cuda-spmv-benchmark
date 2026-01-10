#!/bin/bash
# Multi-GPU SpMV Weak Scaling Benchmark
# Constant work per GPU: each GPU handles ~25M unknowns
# Extracts SpMV timing from CG solver (uses --timers flag)
# Outputs: JSON and CSV files per config

set -e

# ============================================================
# CONFIGURATION - EDIT THIS
# ============================================================
RUNS=10
BRANCH="main"

# Weak scaling: constant work per GPU (~25M unknowns per GPU)
# Format: "gpu_count:grid_size"
WEAK_SCALING_CONFIGS=(
    "1:5000"      # 25M unknowns (baseline)
    "2:7071"      # ~50M unknowns (2× baseline)
    "4:10000"     # 100M unknowns (4× baseline)
    "8:14142"     # ~200M unknowns (8× baseline)
)

# ============================================================
# Auto-detect configuration
# ============================================================
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | head -1 | tr -d ' ')
DATE=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_mgpu_spmv_weak_${GPU_NAME}_${DATE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Configuration:"
echo "  GPU: $GPU_NAME"
echo "  Branch: $BRANCH"
echo "  Runs per config: $RUNS"
echo "  Test: Multi-GPU SpMV Weak Scaling (constant ~25M/GPU)"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

# ============================================================
# Helper functions
# ============================================================
print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

generate_matrix() {
    local size=$1
    local matrix_file="matrix/${size}"

    if [ ! -f "$matrix_file" ]; then
        echo "Generating matrix ${size}×${size}..."
        ./bin/generate_matrix "$size" "$matrix_file"
        echo "✓ Matrix generated: $matrix_file"
    else
        echo "✓ Matrix exists: $matrix_file"
    fi
}

# ============================================================
# Verify/build tools
# ============================================================
print_header "Setup"

echo "[1/3] Checking out branch: $BRANCH"
git checkout "$BRANCH" 2>&1 | grep -E "(Switched|Already on)" || true

echo "[2/3] Building binaries..."
make clean > /dev/null 2>&1 || { echo "✗ make clean failed"; exit 1; }

if make cg_solver_mgpu_stencil > /dev/null 2>&1; then
    echo "✓ cg_solver_mgpu_stencil built"
else
    echo "✗ Build failed, showing errors:"
    make cg_solver_mgpu_stencil
    exit 1
fi

if make generate_matrix > /dev/null 2>&1; then
    echo "✓ generate_matrix built"
else
    echo "✗ Build failed, showing errors:"
    make generate_matrix
    exit 1
fi

mkdir -p matrix
echo "[3/3] Matrix directory ready"

# ============================================================
# Initialize summary file
# ============================================================
cat > "$SUMMARY_FILE" <<EOF
Multi-GPU SpMV - Weak Scaling Benchmark
========================================
GPU: $GPU_NAME
Branch: $BRANCH
Runs per config: $RUNS
Date: $(date)

Note: SpMV timings extracted from CG solver (--timers flag)
      Constant ~25M unknowns per GPU

Weak Scaling Configurations:
EOF

for config in "${WEAK_SCALING_CONFIGS[@]}"; do
    NP=$(echo "$config" | cut -d':' -f1)
    SIZE=$(echo "$config" | cut -d':' -f2)
    UNKNOWNS=$((SIZE * SIZE))
    UNKNOWNS_PER_GPU=$((UNKNOWNS / NP))
    echo "  ${NP} GPUs: ${SIZE}×${SIZE} (${UNKNOWNS} unknowns, ${UNKNOWNS_PER_GPU} per GPU)" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"

# ============================================================
# Main benchmark loop
# ============================================================
print_header "Multi-GPU SpMV Weak Scaling Benchmark"
echo "Start time: $(date)"

for config in "${WEAK_SCALING_CONFIGS[@]}"; do
    NP=$(echo "$config" | cut -d':' -f1)
    SIZE=$(echo "$config" | cut -d':' -f2)

    print_header "GPU COUNT: $NP | Matrix Size: ${SIZE}×${SIZE}"

    MATRIX_FILE="matrix/${SIZE}"
    generate_matrix "$SIZE"

    UNKNOWNS=$((SIZE * SIZE))
    UNKNOWNS_PER_GPU=$((UNKNOWNS / NP))

    BASE_NAME="${GPU_NAME}_spmv_weak_${SIZE}x${SIZE}_np${NP}"
    JSON_FILE="$RESULTS_DIR/${BASE_NAME}.json"
    CSV_FILE="$RESULTS_DIR/${BASE_NAME}.csv"

    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "GPUs: $NP | Matrix: ${SIZE}×${SIZE} (${UNKNOWNS} unknowns, ${UNKNOWNS_PER_GPU} per GPU)" >> "$SUMMARY_FILE"
    echo "Files: ${BASE_NAME}.{json,csv}" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"

    echo "Running: mpirun -np $NP ./bin/cg_solver_mgpu_stencil $MATRIX_FILE --timers --json=$JSON_FILE --csv=$CSV_FILE"

    if mpirun --allow-run-as-root -np "$NP" ./bin/cg_solver_mgpu_stencil "$MATRIX_FILE" \
        --timers --json="$JSON_FILE" --csv="$CSV_FILE" 2>&1 | tee -a "$SUMMARY_FILE"; then
        echo "✓ Test completed successfully"
        echo "  JSON: $JSON_FILE"
        echo "  CSV:  $CSV_FILE"

        if [ -f "$JSON_FILE" ]; then
            SPMV_TIME=$(jq -r '.timing.spmv_ms // "N/A"' "$JSON_FILE" 2>/dev/null)
            ITERS=$(jq -r '.convergence.iterations // 1' "$JSON_FILE" 2>/dev/null)
            if [ "$SPMV_TIME" != "N/A" ] && [ "$SPMV_TIME" != "0.000" ]; then
                SPMV_PER_ITER=$(echo "scale=3; $SPMV_TIME / $ITERS" | bc)
                echo "  SpMV time: ${SPMV_TIME} ms total (${SPMV_PER_ITER} ms/iter)" | tee -a "$SUMMARY_FILE"
            fi
        fi
    else
        echo "✗ Test failed (exit code: $?)"
        echo "FAILED: GPUs=$NP, Size=${SIZE}x${SIZE}" >> "$SUMMARY_FILE"
    fi

    sleep 2
done

# ============================================================
# Summary
# ============================================================
print_header "Benchmark Complete"

TOTAL_TESTS=${#WEAK_SCALING_CONFIGS[@]}

echo ""
echo "============================================================"
echo "Summary:"
echo "  GPU: $GPU_NAME"
echo "  Branch: $BRANCH"
echo "  Scaling type: WEAK (constant work per GPU)"
echo "  Total tests: $TOTAL_TESTS"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "  - Summary: $SUMMARY_FILE"
echo "  - JSON files: $TOTAL_TESTS files"
echo "  - CSV files:  $TOTAL_TESTS files"
echo "============================================================"
echo "End time: $(date)"

# ============================================================
# Generate analysis script
# ============================================================
cat > "$RESULTS_DIR/analyze_spmv_weak.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
"""
Analyze multi-GPU SpMV weak scaling results
"""
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Read all JSON files
results = []
for json_file in sorted(glob.glob("*.json")):
    with open(json_file) as f:
        content = f.read().replace(': inf', ': null')
        data = json.loads(content)

        grid_size = data['matrix']['grid_size']
        iterations = data['convergence']['iterations']
        spmv_total_ms = data['timing'].get('spmv_ms', 0)
        spmv_per_iter = spmv_total_ms / iterations if iterations > 0 else 0

        results.append({
            'gpus': data['num_gpus'],
            'size': grid_size,
            'unknowns': data['matrix']['rows'],
            'unknowns_per_gpu': data['matrix']['rows'] // data['num_gpus'],
            'iterations': iterations,
            'spmv_per_iter_ms': spmv_per_iter
        })

# Sort by GPU count
results = sorted(results, key=lambda x: x['gpus'])

# Create weak scaling plots (2-panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = '#2E86AB'
gpus_list = [r['gpus'] for r in results]
times = [r['spmv_per_iter_ms'] for r in results]

# Plot 1: SpMV time per iteration vs GPU count
baseline_time = results[0]['spmv_per_iter_ms']
ax1.plot(gpus_list, times, marker='o', linestyle='-',
         color=colors, linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white', label='Measured')
ax1.axhline(y=baseline_time, color='k', linestyle='--', linewidth=2,
            alpha=0.6, label=f'Ideal ({baseline_time:.2f} ms/iter)')

ax1.set_xlabel('Number of GPUs', fontsize=13, fontweight='bold')
ax1.set_ylabel('SpMV Time per Iteration (ms)', fontsize=13, fontweight='bold')
ax1.set_title('Weak Scaling: SpMV Time vs GPU Count', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(gpus_list)
ax1.legend(fontsize=11, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([0, max(times) * 1.2])

# Plot 2: Weak scaling efficiency
efficiencies = [100 * baseline_time / t for t in times]
ax2.bar(range(len(gpus_list)), efficiencies, color=colors, alpha=0.8,
        edgecolor='black', linewidth=1.5)
ax2.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.6, label='Ideal')

ax2.set_xticks(range(len(gpus_list)))
ax2.set_xticklabels([f'{g} GPU{"s" if g>1 else ""}' for g in gpus_list])
ax2.set_ylabel('Weak Scaling Efficiency (%)', fontsize=13, fontweight='bold')
ax2.set_title('SpMV Weak Scaling Efficiency', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_ylim([0, 110])

# Add value labels
for i, (eff, time) in enumerate(zip(efficiencies, times)):
    ax2.text(i, eff + 2, f'{eff:.1f}%\n({time:.2f}ms)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('mgpu_spmv_weak_scaling.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Generated: mgpu_spmv_weak_scaling.png")

# Print summary table
print("\n" + "="*100)
print("Multi-GPU SpMV Weak Scaling Results (Constant ~25M unknowns per GPU)")
print("="*100)
print(f"{'GPUs':<6} {'Size':<10} {'Total Unknowns':<15} {'Per GPU':<12} {'SpMV/iter (ms)':<15} {'Efficiency':<12}")
print("-"*100)

for r in results:
    efficiency = 100 * baseline_time / r['spmv_per_iter_ms']
    print(f"{r['gpus']:<6} {r['size']:<10} {r['unknowns']:<15,} {r['unknowns_per_gpu']:<12,} "
          f"{r['spmv_per_iter_ms']:<15.3f} {efficiency:<12.1f}%")

print("-"*100)
print(f"\nBaseline (1 GPU): {baseline_time:.3f} ms/iter")
print(f"Ideal weak scaling: constant time ({baseline_time:.3f} ms/iter) for all GPU counts")
print("="*100)
PYTHON_EOF

chmod +x "$RESULTS_DIR/analyze_spmv_weak.py"
echo ""
echo "Analysis script created: $RESULTS_DIR/analyze_spmv_weak.py"
echo "Run: cd $RESULTS_DIR && python3 analyze_spmv_weak.py"
