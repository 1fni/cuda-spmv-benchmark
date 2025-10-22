#!/bin/bash

# Save benchmark results to dedicated Git branch
# Handles all Git configuration and branch creation automatically

set -e

echo "=== Saving Results to Git Branch ==="

# Setup Git user if not configured
if ! git config user.email &> /dev/null; then
    echo "Configuring Git user..."
    git config --global user.email "bouhrour.stephane@gmail.com"
    git config --global user.name "1fni"
fi

# Get GPU architecture for branch naming
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/NVIDIA //g' | tr ' -' '_' | tr '[:upper:]' '[:lower:]')
    echo "Detected GPU: $GPU_ARCH"
else
    GPU_ARCH="unknown_gpu"
    echo "Warning: Could not detect GPU, using default name"
fi

# Create branch name with GPU and timestamp
BRANCH_NAME="results-${GPU_ARCH}-$(date +%Y%m%d_%H%M)"
echo "Creating branch: $BRANCH_NAME"

# Check if results directory exists
if [ ! -d "results" ]; then
    echo "Error: No results directory found. Run benchmark first."
    exit 1
fi

# Create and switch to results branch
git checkout -b "$BRANCH_NAME"

# Add results (force override .gitignore)
echo "Adding results to Git..."
git add -f results/

# Get matrix info for commit message
MATRIX_FILES=$(find results/ -name "*results.json" | head -1)
if [ -f "$MATRIX_FILES" ]; then
    MATRIX_ROWS=$(grep -o '"rows":[0-9]*' "$MATRIX_FILES" | sed 's/"rows"://' | head -1)
    MATRIX_COLS=$(grep -o '"cols":[0-9]*' "$MATRIX_FILES" | sed 's/"cols"://' | head -1)
    MATRIX_NNZ=$(grep -o '"nnz":[0-9]*' "$MATRIX_FILES" | sed 's/"nnz"://' | head -1)
    MATRIX_INFO="${MATRIX_ROWS}x${MATRIX_COLS} (${MATRIX_NNZ} nnz)"
else
    MATRIX_INFO="Unknown matrix"
fi

# Extract CSR vs STENCIL performance from CSV
CSV_FILE=$(find results/ -name "*.csv" | head -1)
CSR_PERF=""
STENCIL_PERF=""
if [ -f "$CSV_FILE" ]; then
    while IFS=',' read -r op time_ms gflops bandwidth; do
        if [ "$op" = "csr" ]; then
            CSR_PERF="CSR: ${gflops} GFLOPS (${time_ms}ms)"
        elif [ "$op" = "stencil5" ]; then
            STENCIL_PERF="STENCIL: ${gflops} GFLOPS (${time_ms}ms)"
        fi
    done < "$CSV_FILE"
fi

PERF_COMPARISON="$CSR_PERF vs $STENCIL_PERF"

# Commit with detailed information
git commit -m "GPU benchmark results - $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo $GPU_ARCH)

Matrix: $MATRIX_INFO
Performance: $PERF_COMPARISON
Date: $(date)"

# Push to origin
echo "Pushing to GitHub..."
git push origin "$BRANCH_NAME"

if [ $? -eq 0 ]; then
    echo
    echo "✅ Results saved successfully!"
    echo "Branch: $BRANCH_NAME"
    echo "GitHub URL: https://github.com/1fni/cuda-spmv-benchmark/tree/$BRANCH_NAME"
    echo
    echo "To retrieve locally:"
    echo "  git fetch origin"
    echo "  git checkout $BRANCH_NAME"
    echo
    echo "Note: Ignore the pull request suggestion - keep results branches separate."
else
    echo "❌ Push failed. Check your GitHub credentials."
    exit 1
fi