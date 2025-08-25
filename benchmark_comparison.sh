#!/bin/bash
# SpMV Performance Comparison Script
# Automated benchmarking with structured output formats

echo "=== SpMV Performance Comparison Benchmark ==="
echo "Testing CSR vs STENCIL5 formats with structured data export"
echo ""

# Generate fresh stencil matrix
echo "Generating 100x100 5-point stencil matrix..."
./bin/generate_matrix 100 matrix/stencil_100x100.mtx

echo ""
echo "=== JSON Comparison Results ==="

echo "CSR Performance:"
./bin/spmv_bench matrix/example100x100.mtx --mode=csr --output-format=json 2>/dev/null | grep -v "Loading\|Matrix loaded\|Performing\|Executing\|SpMV completed\|Cleaning\|checksum\|Kernel time"

echo ""
echo "STENCIL5 Performance:"  
./bin/spmv_bench matrix/stencil_100x100.mtx --mode=stencil5 --output-format=json 2>/dev/null | grep -v "Loading\|Matrix loaded\|ELL WIDTH\|Performing\|Executing\|SpMV completed\|Cleaning\|checksum\|Kernel time\|Matrix rows\|Result"

echo ""
echo "=== CSV Aggregated Results ==="
echo "# Comparative performance data for spreadsheet analysis"

# Run both benchmarks and combine CSV output
{
    ./bin/spmv_bench matrix/example100x100.mtx --mode=csr --output-format=csv 2>/dev/null | grep -v "Loading\|Matrix loaded\|Performing\|Executing\|SpMV completed\|Cleaning\|checksum\|Kernel time"
    ./bin/spmv_bench matrix/stencil_100x100.mtx --mode=stencil5 --output-format=csv 2>/dev/null | grep -v "Loading\|Matrix loaded\|ELL WIDTH\|Performing\|Executing\|SpMV completed\|Cleaning\|checksum\|Kernel time\|Matrix rows\|Result" | tail -n +2
}

echo ""
echo "=== Performance Summary ==="
echo "Use the JSON/CSV data above for:"
echo "- Automated visualization scripts"  
echo "- Performance tracking dashboards"
echo "- Spreadsheet analysis and charts"
echo "- CI/CD performance regression testing"