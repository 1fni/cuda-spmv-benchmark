#!/bin/bash

# Automated benchmarking and visualization script
# Usage: ./benchmark_and_visualize.sh <matrix_file> [output_prefix]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <matrix_file> [output_prefix]"
    echo "  matrix_file: Input Matrix Market file"
    echo "  output_prefix: Optional prefix for output files (default: benchmark_results)"
    exit 1
fi

MATRIX_FILE="$1"
OUTPUT_PREFIX="${2:-benchmark_results}"

if [ ! -f "$MATRIX_FILE" ]; then
    echo "Error: Matrix file $MATRIX_FILE not found"
    exit 1
fi

# Check if benchmark binary exists
if [ ! -f "./bin/spmv_bench" ]; then
    echo "Error: spmv_bench not found. Please run 'make' first."
    exit 1
fi

echo "=== SpMV Benchmark and Visualization Pipeline ==="
echo "Matrix file: $MATRIX_FILE"
echo "Output prefix: $OUTPUT_PREFIX"
echo

# Create results directory
mkdir -p results
cd results

# Test all SpMV implementations
OPERATORS=("csr" "stencil5" "ellpack")
CSV_FILE="${OUTPUT_PREFIX}.csv"
JSON_DIR="${OUTPUT_PREFIX}_json"

echo "Running benchmarks..."
echo "operator,gflops,bandwidth" > $CSV_FILE

mkdir -p $JSON_DIR

for op in "${OPERATORS[@]}"; do
    echo "  Testing $op format..."
    
    # Run benchmark and capture JSON output
    JSON_FILE="${JSON_DIR}/${op}_results.json"
    ../bin/spmv_bench "../$MATRIX_FILE" --mode=$op --output-format=json --output-file=$JSON_FILE
    
    if [ $? -eq 0 ] && [ -f $JSON_FILE ]; then
        # Extract metrics from JSON using standard tools
        gflops=$(grep -o '"gflops": [0-9.]*' $JSON_FILE | sed 's/"gflops": //')
        bandwidth=$(grep -o '"bandwidth_gb_s": [0-9.]*' $JSON_FILE | sed 's/"bandwidth_gb_s": //')
        
        if [ -n "$gflops" ] && [ -n "$bandwidth" ]; then
            echo "$op,$gflops,$bandwidth" >> $CSV_FILE
            echo "    GFLOPS: $gflops, Bandwidth: ${bandwidth} GB/s"
        else
            echo "    Warning: Could not extract metrics for $op"
        fi
    else
        echo "    Error: Benchmark failed for $op"
    fi
done

echo

# Generate performance charts
if [ -f $CSV_FILE ] && [ -s $CSV_FILE ] && [ $(wc -l < $CSV_FILE) -gt 1 ]; then
    echo "Generating performance visualization..."
    
    # Get matrix info for chart title
    MATRIX_BASENAME=$(basename "$MATRIX_FILE" .mtx)
    CHART_TITLE="SpMV Performance Comparison: $MATRIX_BASENAME"
    
    # Generate charts
    ../scripts/generate_performance_chart.sh $CSV_FILE "${OUTPUT_PREFIX}_performance" "$CHART_TITLE"
    
    if [ $? -eq 0 ]; then
        echo "Charts generated successfully!"
        echo
    else
        echo "Warning: Chart generation failed"
    fi
else
    echo "Error: No valid benchmark data to visualize"
    exit 1
fi

# Generate summary report
REPORT_FILE="${OUTPUT_PREFIX}_report.txt"
echo "=== SpMV Performance Analysis Report ===" > $REPORT_FILE
echo "Matrix: $MATRIX_FILE" >> $REPORT_FILE
echo "Generated: $(date)" >> $REPORT_FILE
echo >> $REPORT_FILE

# Parse first JSON file to get matrix info
FIRST_JSON=$(ls ${JSON_DIR}/*.json | head -1)
if [ -f "$FIRST_JSON" ]; then
    echo "=== Matrix Information ===" >> $REPORT_FILE
    grep -o '"rows":[0-9]*' "$FIRST_JSON" | sed 's/"rows":/Rows: /' >> $REPORT_FILE
    grep -o '"cols":[0-9]*' "$FIRST_JSON" | sed 's/"cols":/Columns: /' >> $REPORT_FILE  
    grep -o '"nnz":[0-9]*' "$FIRST_JSON" | sed 's/"nnz":/Non-zeros: /' >> $REPORT_FILE
    grep -o '"density":[0-9.]*' "$FIRST_JSON" | sed 's/"density":/Density: /' >> $REPORT_FILE
    echo >> $REPORT_FILE
fi

echo "=== Performance Results ===" >> $REPORT_FILE
echo "Format     | GFLOPS | Bandwidth (GB/s) | Relative Performance" >> $REPORT_FILE
echo "-----------|--------|------------------|--------------------" >> $REPORT_FILE

# Calculate relative performance (compared to first entry)
BASELINE_GFLOPS=""
while IFS=',' read -r op gflops bandwidth; do
    if [ "$op" != "operator" ]; then  # Skip header
        if [ -z "$BASELINE_GFLOPS" ]; then
            BASELINE_GFLOPS=$gflops
            relative="1.00x (baseline)"
        else
            relative=$(echo "scale=2; $gflops / $BASELINE_GFLOPS" | bc)x
        fi
        printf "%-10s | %6.2f | %14.2f | %s\n" "$op" "$gflops" "$bandwidth" "$relative" >> $REPORT_FILE
    fi
done < $CSV_FILE

# Add recommendations
echo >> $REPORT_FILE
echo "=== Recommendations ===" >> $REPORT_FILE

# Find best performer
BEST_OP=""
BEST_GFLOPS=0
while IFS=',' read -r op gflops bandwidth; do
    if [ "$op" != "operator" ]; then
        if (( $(echo "$gflops > $BEST_GFLOPS" | bc -l) )); then
            BEST_GFLOPS=$gflops
            BEST_OP=$op
        fi
    fi
done < $CSV_FILE

echo "Best performing format: $BEST_OP ($BEST_GFLOPS GFLOPS)" >> $REPORT_FILE

# Add format-specific recommendations
case $BEST_OP in
    "stencil5")
        echo "Recommendation: Optimal for structured grid problems (finite difference, stencil operations)" >> $REPORT_FILE
        ;;
    "csr")
        echo "Recommendation: Good general-purpose format for irregular sparse patterns" >> $REPORT_FILE
        ;;
    "ellpack")
        echo "Recommendation: Efficient for matrices with regular row structures" >> $REPORT_FILE
        ;;
esac

echo
echo "=== Results Summary ==="
echo "Data files:"
echo "  - CSV data: results/$CSV_FILE"
echo "  - JSON details: results/$JSON_DIR/"
echo "  - Analysis report: results/$REPORT_FILE"
echo
echo "Visualization files:"
echo "  - PNG chart: results/${OUTPUT_PREFIX}_performance.png"
echo "  - SVG chart: results/${OUTPUT_PREFIX}_performance.svg"
echo
echo "Performance summary:"
cat results/$REPORT_FILE | grep -A 10 "=== Performance Results ==="

cd ..
echo
echo "Benchmark and visualization pipeline completed!"
echo "Check the 'results/' directory for all generated files."