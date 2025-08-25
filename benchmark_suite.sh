#!/bin/bash
# SpMV Benchmark Suite with File Output

echo "=== SpMV Benchmark Suite ==="
echo "Performance analysis with JSON/CSV export"
echo ""

# Create results directory
mkdir -p results

# Generate test matrices
echo "Generating test matrices..."
./bin/generate_matrix 100 matrix/stencil_100x100.mtx

echo ""
echo "=== Running Benchmarks ==="

# Benchmark CSR format
echo "Benchmarking CSR format..."
./bin/spmv_bench matrix/example100x100.mtx --mode=csr \
    --output-format=json --output-file=results/csr_performance.json

./bin/spmv_bench matrix/example100x100.mtx --mode=csr \
    --output-format=csv --output-file=results/csr_performance.csv

# Benchmark STENCIL5 format  
echo ""
echo "Benchmarking STENCIL5 format..."
./bin/spmv_bench matrix/stencil_100x100.mtx --mode=stencil5 \
    --output-format=json --output-file=results/stencil5_performance.json

./bin/spmv_bench matrix/stencil_100x100.mtx --mode=stencil5 \
    --output-format=csv --output-file=results/stencil5_performance.csv

echo ""
echo "=== Results Files ==="
ls -la results/

echo ""
echo "=== Performance Summary ==="
echo "CSR Performance:"
grep "gflops\|bandwidth_gb_s\|execution_time_ms" results/csr_performance.json

echo ""
echo "STENCIL5 Performance:"
grep "gflops\|bandwidth_gb_s\|execution_time_ms" results/stencil5_performance.json

echo ""
echo "=== Combined CSV Data ==="
{
    head -1 results/csr_performance.csv  # Header
    tail -n +2 results/csr_performance.csv  # CSR data
    tail -n +2 results/stencil5_performance.csv  # STENCIL5 data  
}

echo ""
echo "Benchmark suite completed."