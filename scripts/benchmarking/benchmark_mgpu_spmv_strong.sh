#!/bin/bash
# Multi-GPU SpMV Strong Scaling Benchmark
# Tests same problem size on different GPU counts
# Extracts SpMV timing from CG solver (uses --timers flag)
# Outputs: JSON and CSV files per config

set -e

# ============================================================
# CONFIGURATION - EDIT THIS
# ============================================================
RUNS=10
BRANCH="main"

# Strong scaling: same problem sizes on different GPU counts
# Format: "gpu_count:matrix_size1,matrix_size2,..."
PROBLEM_CONFIGS=(
    "1:10000,15000,20000"
    "2:10000,15000,20000"
    "4:10000,15000,20000"
    "8:10000,15000,20000"
)

# ============================================================
# Auto-detect configuration
# ============================================================
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | head -1 | tr -d ' ')
DATE=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_mgpu_spmv_strong_${GPU_NAME}_${DATE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Configuration:"
echo "  GPU: $GPU_NAME"
echo "  Branch: $BRANCH"
echo "  Runs per config: $RUNS"
echo "  Test: Multi-GPU SpMV Strong Scaling"
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

print_section() {
    echo ""
    echo "------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------"
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

# Checkout target branch
echo "[1/3] Checking out branch: $BRANCH"
git checkout "$BRANCH" 2>&1 | grep -E "(Switched|Already on)" || true

# Build binaries
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

# Create matrix directory
mkdir -p matrix
echo "[3/3] Matrix directory ready"

# ============================================================
# Initialize summary file
# ============================================================
cat > "$SUMMARY_FILE" <<EOF
Multi-GPU SpMV - Strong Scaling Benchmark
==========================================
GPU: $GPU_NAME
Branch: $BRANCH
Runs per config: $RUNS
Date: $(date)

Note: SpMV timings extracted from CG solver (--timers flag)
      Average SpMV time per iteration reported

Problem Size Configurations:
EOF

for config in "${PROBLEM_CONFIGS[@]}"; do
    echo "  $config" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"

# ============================================================
# Main benchmark loop
# ============================================================
print_header "Multi-GPU SpMV Strong Scaling Benchmark"
echo "Start time: $(date)"

# Parse and run each configuration
for config in "${PROBLEM_CONFIGS[@]}"; do
    # Split config into GPU count and sizes
    NP=$(echo "$config" | cut -d':' -f1)
    SIZES=$(echo "$config" | cut -d':' -f2)

    print_header "GPU COUNT: $NP"

    # Split sizes by comma
    IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"

    for SIZE in "${SIZE_ARRAY[@]}"; do
        print_section "Matrix Size: ${SIZE}×${SIZE}"

        MATRIX_FILE="matrix/${SIZE}"

        # Generate matrix if needed
        generate_matrix "$SIZE"

        # Calculate matrix stats
        UNKNOWNS=$((SIZE * SIZE))

        # Generate output filenames
        BASE_NAME="${GPU_NAME}_spmv_${SIZE}x${SIZE}_np${NP}"
        JSON_FILE="$RESULTS_DIR/${BASE_NAME}.json"
        CSV_FILE="$RESULTS_DIR/${BASE_NAME}.csv"

        # Write header to summary
        echo "" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"
        echo "GPUs: $NP | Matrix: ${SIZE}×${SIZE} (${UNKNOWNS} unknowns)" >> "$SUMMARY_FILE"
        echo "Files: ${BASE_NAME}.{json,csv}" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"

        # Run benchmark with --timers to get SpMV breakdown
        echo "Running: mpirun -np $NP ./bin/cg_solver_mgpu_stencil $MATRIX_FILE --timers --json=$JSON_FILE --csv=$CSV_FILE"

        if mpirun --allow-run-as-root -np "$NP" ./bin/cg_solver_mgpu_stencil "$MATRIX_FILE" \
            --timers --json="$JSON_FILE" --csv="$CSV_FILE" 2>&1 | tee -a "$SUMMARY_FILE"; then
            echo "✓ Test completed successfully"
            echo "  JSON: $JSON_FILE"
            echo "  CSV:  $CSV_FILE"

            # Extract SpMV timing if available
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

        # Small delay between tests
        sleep 2
    done

    echo "✓ GPU count $NP complete"
done

# ============================================================
# Summary
# ============================================================
print_header "Benchmark Complete"

TOTAL_TESTS=0
for config in "${PROBLEM_CONFIGS[@]}"; do
    SIZES=$(echo "$config" | cut -d':' -f2)
    IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"
    TOTAL_TESTS=$((TOTAL_TESTS + ${#SIZE_ARRAY[@]}))
done

echo ""
echo "============================================================"
echo "Summary:"
echo "  GPU: $GPU_NAME"
echo "  Branch: $BRANCH"
echo "  GPU counts tested: 1, 2, 4, 8"
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
cat > "$RESULTS_DIR/analyze_spmv_scaling.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
"""
Analyze multi-GPU SpMV strong scaling results
Extracts SpMV timing from CG solver JSON output
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

        # Extract SpMV time (total across all iterations)
        spmv_total_ms = data['timing'].get('spmv_ms', 0)
        spmv_per_iter = spmv_total_ms / iterations if iterations > 0 else 0

        results.append({
            'gpus': data['num_gpus'],
            'size': grid_size,
            'unknowns': data['matrix']['rows'],
            'iterations': iterations,
            'spmv_total_ms': spmv_total_ms,
            'spmv_per_iter_ms': spmv_per_iter
        })

# Group by problem size (strong scaling)
by_size = {}
for r in results:
    if r['size'] not in by_size:
        by_size[r['size']] = []
    by_size[r['size']].append(r)

# Create strong scaling plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

colors = {10000: '#2E86AB', 15000: '#A23B72', 20000: '#F18F01'}
markers = {10000: 'o', 15000: 's', 20000: '^'}
labels = {10000: '10k×10k', 15000: '15k×15k', 20000: '20k×20k'}

# Plot 1: SpMV time per iteration vs GPU count
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    gpus = [d['gpus'] for d in data]
    times = [d['spmv_per_iter_ms'] for d in data]
    ax1.plot(gpus, times, marker=markers[size], linestyle='-',
             label=labels[size], color=colors[size],
             linewidth=2.5, markersize=10, markeredgewidth=1.5, markeredgecolor='white')

ax1.set_xlabel('Number of GPUs', fontsize=13, fontweight='bold')
ax1.set_ylabel('SpMV Time per Iteration (ms)', fontsize=13, fontweight='bold')
ax1.set_title('Multi-GPU SpMV Performance', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks([1, 2, 4, 8])
ax1.legend(fontsize=11, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)

# Plot 2: Speedup vs GPU count
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    gpus = [d['gpus'] for d in data]
    baseline = data[0]['spmv_per_iter_ms']
    speedups = [baseline / d['spmv_per_iter_ms'] for d in data]
    ax2.plot(gpus, speedups, marker=markers[size], linestyle='-',
             label=labels[size], color=colors[size],
             linewidth=2.5, markersize=10, markeredgewidth=1.5, markeredgecolor='white')

ax2.plot([1, 2, 4, 8], [1, 2, 4, 8], 'k--', linewidth=2, label='Ideal', alpha=0.6)
ax2.set_xlabel('Number of GPUs', fontsize=13, fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=13, fontweight='bold')
ax2.set_title('SpMV Speedup vs Single-GPU', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks([1, 2, 4, 8])
ax2.legend(fontsize=11, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0, 9])

# Plot 3: Parallel efficiency
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    gpus = [d['gpus'] for d in data]
    baseline = data[0]['spmv_per_iter_ms']
    efficiencies = [100 * baseline / (d['spmv_per_iter_ms'] * d['gpus']) for d in data]
    ax3.plot(gpus, efficiencies, marker=markers[size], linestyle='-',
             label=labels[size], color=colors[size],
             linewidth=2.5, markersize=10, markeredgewidth=1.5, markeredgecolor='white')

ax3.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.6, label='Ideal')
ax3.set_xlabel('Number of GPUs', fontsize=13, fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontsize=13, fontweight='bold')
ax3.set_title('SpMV Parallel Efficiency', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks([1, 2, 4, 8])
ax3.set_ylim([0, 110])
ax3.legend(fontsize=11, frameon=True, shadow=True, loc='lower left')
ax3.grid(True, alpha=0.3, linestyle='--')

# Plot 4: Total SpMV time (all iterations)
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    gpus = [d['gpus'] for d in data]
    totals = [d['spmv_total_ms'] for d in data]
    ax4.plot(gpus, totals, marker=markers[size], linestyle='-',
             label=labels[size], color=colors[size],
             linewidth=2.5, markersize=10, markeredgewidth=1.5, markeredgecolor='white')

ax4.set_xlabel('Number of GPUs', fontsize=13, fontweight='bold')
ax4.set_ylabel('Total SpMV Time (ms)', fontsize=13, fontweight='bold')
ax4.set_title('Total SpMV Time (All CG Iterations)', fontsize=14, fontweight='bold', pad=15)
ax4.set_xticks([1, 2, 4, 8])
ax4.legend(fontsize=11, frameon=True, shadow=True)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('mgpu_spmv_strong_scaling.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Generated: mgpu_spmv_strong_scaling.png")

# Print summary table
print("\n" + "="*100)
print("Multi-GPU SpMV Strong Scaling Results")
print("="*100)
print(f"{'Size':<10} {'GPUs':<6} {'Unknowns':<15} {'Iter':<6} {'SpMV/iter (ms)':<15} {'Speedup':<10} {'Efficiency':<12}")
print("-"*100)

for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    baseline = data[0]['spmv_per_iter_ms']
    print(f"\n{size}×{size} Matrix:")
    for r in data:
        speedup = baseline / r['spmv_per_iter_ms']
        efficiency = 100 * speedup / r['gpus']
        print(f"{'':10} {r['gpus']:<6} {r['unknowns']:<15,} {r['iterations']:<6} "
              f"{r['spmv_per_iter_ms']:<15.3f} {speedup:<10.2f} {efficiency:<12.1f}%")

print("="*100)
PYTHON_EOF

chmod +x "$RESULTS_DIR/analyze_spmv_scaling.py"
echo ""
echo "Analysis script created: $RESULTS_DIR/analyze_spmv_scaling.py"
echo "Run: cd $RESULTS_DIR && python3 analyze_spmv_scaling.py"
