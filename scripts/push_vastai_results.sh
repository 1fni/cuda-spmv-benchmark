#!/bin/bash

# Push VastAI Results Script
# Creates a new branch with benchmark results for easy download from GitHub

set -e

echo "📤 VastAI Results Push"
echo "====================="

# Check if results directory exists
if [ ! -d "results" ]; then
    echo "❌ No results directory found"
    exit 1
fi

# Detect GPU architecture for branch naming
GPU_ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 | sed 's/ /_/g' | tr '[:upper:]' '[:lower:]')
TIMESTAMP=$(date +%Y%m%d_%H%M)
BRANCH_NAME="vastai-results-${GPU_ARCH}-${TIMESTAMP}"

echo "🌟 Creating results branch: $BRANCH_NAME"

# Create and switch to new branch
git checkout -b "$BRANCH_NAME"

# Configure git user for this commit
git config user.email "bouhrour.stephane@gmail.com"
git config user.name "1fni"

# Add results directory with force flag (ignore gitignore)
git add -f results/
git commit -m "Add VastAI benchmark results - $GPU_ARCH - $TIMESTAMP"

# Push to remote with force flag
git push origin "$BRANCH_NAME" -f

echo ""
echo "🎉 Results pushed to GitHub!"
echo "Branch: $BRANCH_NAME"
echo "Download: https://github.com/1fni/cuda-spmv-benchmark/archive/refs/heads/$BRANCH_NAME.zip"

# Return to original branch
git checkout -