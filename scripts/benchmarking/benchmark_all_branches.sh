#!/bin/bash
# Benchmark script for multi-GPU CG solver across all branches
# Tests: main, halo-cudamemcpypeer, overlap-streams, overlap-cudaipc
# Ranks: 1, 2, 4, 8
# Outputs: JSON and CSV files per config

set -e

# ============================================================
# CONFIGURATION - EDIT THIS
# ============================================================
MATRIX="matrix/15000"  # ← CHANGE THIS TO YOUR MATRIX
RUNS=10

# ============================================================
# Auto-detect configuration
# ============================================================
# Get GPU architecture (first GPU)
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | head -1 | tr -d ' ')

# Extract matrix size from filename (e.g., stencil_5000x5000.mtx → 5000x5000)
MATRIX_SIZE=$(basename "$MATRIX" .mtx | grep -oE '[0-9]+x[0-9]+' || echo "unknown")

# Date for filename
DATE=$(date +%Y%m%d_%H%M%S)

# Results directory
RESULTS_DIR="results_${GPU_NAME}_${MATRIX_SIZE}_${DATE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Configuration:"
echo "  GPU: $GPU_NAME"
echo "  Matrix: $MATRIX (size: $MATRIX_SIZE)"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"

# ============================================================
# Branch list
# ============================================================
BRANCHES=(
    "fix/p2p-cache-coherency"
    "main"
    #"feature/overlap-streams"
    #"feature/overlap-cudaipc"
    #"feature/halo-cudamemcpypeer"
)

# Rank configurations to test
RANKS=(1 2 4 8)

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

# ============================================================
# Main benchmark loop
# ============================================================
print_header "Multi-GPU CG Solver Benchmark"
echo "Start time: $(date)"

# Initialize summary file
cat > "$SUMMARY_FILE" <<EOF
Multi-GPU CG Solver Benchmark Results
======================================
GPU: $GPU_NAME
Matrix: $MATRIX (size: $MATRIX_SIZE)
Runs per config: $RUNS
Date: $(date)

EOF

# Save current branch to restore later
ORIGINAL_BRANCH=$(git branch --show-current)
echo "Original branch: $ORIGINAL_BRANCH"

# Loop over branches
for BRANCH in "${BRANCHES[@]}"; do
    print_header "BRANCH: $BRANCH"

    # Sanitize branch name for filename (replace / with -)
    BRANCH_CLEAN=$(echo "$BRANCH" | sed 's|/|-|g')

    # Checkout branch
    echo "[1/3] Checking out branch: $BRANCH"
    git checkout "$BRANCH" 2>&1 | grep -E "(Switched|Already on)" || true

    # Clean and compile
    echo "[2/3] Compiling..."
    make clean > /dev/null 2>&1
    make cg_solver_mgpu_stencil > /dev/null 2>&1
    echo "✓ Compilation successful"

    # Run benchmarks for each rank count
    echo "[3/3] Running benchmarks..."

    for NP in "${RANKS[@]}"; do
        print_section "Testing with $NP rank(s)"

        # Generate output filenames
        BASE_NAME="${GPU_NAME}_${MATRIX_SIZE}_${BRANCH_CLEAN}_np${NP}"
        JSON_FILE="$RESULTS_DIR/${BASE_NAME}.json"
        CSV_FILE="$RESULTS_DIR/${BASE_NAME}.csv"

        # Write header to summary
        echo "" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"
        echo "Branch: $BRANCH | Ranks: $NP" >> "$SUMMARY_FILE"
        echo "Files: ${BASE_NAME}.{json,csv}" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"

        # Run benchmark with JSON/CSV output
        echo "Running: mpirun -np $NP ./bin/cg_solver_mgpu_stencil $MATRIX --json=$JSON_FILE --csv=$CSV_FILE"

        if mpirun --allow-run-as-root -np "$NP" ./bin/cg_solver_mgpu_stencil "$MATRIX" \
            --json="$JSON_FILE" --csv="$CSV_FILE" 2>&1 | tee -a "$SUMMARY_FILE"; then
            echo "✓ Test completed successfully"
            echo "  JSON: $JSON_FILE"
            echo "  CSV:  $CSV_FILE"
        else
            echo "✗ Test failed (exit code: $?)"
            echo "FAILED: Branch=$BRANCH, Ranks=$NP" >> "$SUMMARY_FILE"
        fi

        # Small delay between tests
        sleep 2
    done

    echo "✓ Branch $BRANCH complete"
done

# Restore original branch
print_header "Benchmark Complete"
echo "Restoring original branch: $ORIGINAL_BRANCH"
git checkout "$ORIGINAL_BRANCH" 2>&1 | grep -E "(Switched|Already on)" || true

echo ""
echo "============================================================"
echo "Summary:"
echo "  GPU: $GPU_NAME"
echo "  Matrix: $MATRIX_SIZE"
echo "  Branches tested: ${#BRANCHES[@]}"
echo "  Rank configs: ${RANKS[*]}"
echo "  Total tests: $((${#BRANCHES[@]} * ${#RANKS[@]}))"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "  - Summary: $SUMMARY_FILE"
echo "  - JSON files: $((${#BRANCHES[@]} * ${#RANKS[@]})) files"
echo "  - CSV files:  $((${#BRANCHES[@]} * ${#RANKS[@]})) files"
echo "============================================================"
echo "End time: $(date)"
