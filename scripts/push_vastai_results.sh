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

# Generate branch name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M)
BRANCH_NAME="vastai-results-${TIMESTAMP}"

echo "🌟 Creating results branch: $BRANCH_NAME"

# Create and switch to new branch
git checkout -b "$BRANCH_NAME"

# Add results directory
git add results/
git commit -m "Add VastAI benchmark results - $TIMESTAMP"

# Push to remote
git push origin "$BRANCH_NAME"

echo ""
echo "🎉 Results pushed to GitHub!"
echo "Branch: $BRANCH_NAME"
echo "Download: https://github.com/1fni/cuda-spmv-benchmark/archive/refs/heads/$BRANCH_NAME.zip"

# Return to original branch
git checkout -