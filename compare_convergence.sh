#!/bin/bash
# Compare iteration-by-iteration convergence between main and overlap
# Run on remote to see where they diverge

set -e

MATRIX="matrix/10000"
RANKS=2

echo "========================================="
echo "Convergence Comparison: Main vs Overlap"
echo "Matrix: $MATRIX"
echo "Ranks: $RANKS"
echo "========================================="
echo ""

# Test Main with verbose output
echo "[1/2] Running MAIN with iteration details..."
git checkout investigate/convergence-difference > /dev/null 2>&1
make cg_single_run_test > /dev/null 2>&1

mpirun --allow-run-as-root -np $RANKS ./bin/cg_single_run_test $MATRIX 2>&1 | tee /tmp/main_iterations.log

echo ""
echo "[2/2] Running OVERLAP with iteration details..."
git checkout feature/overlap-streams > /dev/null 2>&1
make cg_single_run_test > /dev/null 2>&1

mpirun --allow-run-as-root -np $RANKS ./bin/cg_single_run_test $MATRIX 2>&1 | tee /tmp/overlap_iterations.log

echo ""
echo "========================================="
echo "Analyzing convergence difference..."
echo ""

# Extract iteration residuals
echo "MAIN convergence (first 20 iterations):"
grep "Iter" /tmp/main_iterations.log | head -20

echo ""
echo "OVERLAP convergence (first 20 iterations):"
grep "Iter" /tmp/overlap_iterations.log | head -20

echo ""
echo "========================================="
echo "Convergence summary:"
echo ""
echo "MAIN:"
grep "Converged" /tmp/main_iterations.log

echo ""
echo "OVERLAP:"
grep "Converged" /tmp/overlap_iterations.log

echo ""
echo "To see full logs:"
echo "  cat /tmp/main_iterations.log"
echo "  cat /tmp/overlap_iterations.log"
echo "========================================="
