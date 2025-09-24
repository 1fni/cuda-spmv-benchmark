#!/bin/bash

# VastAI Multimode Benchmark Script
# Tests all SpMV implementations with a single matrix load (no reload between modes)
# Optimized for VastAI instances with automatic GPU detection and sizing

set -e

echo "🚀 VastAI Multimode GPU Benchmark Pipeline"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "bin/spmv_bench" ] && [ ! -f "Makefile" ]; then
    echo "❌ Not in project root or binaries not built"
    echo "Run from project root after 'make'"
    exit 1
fi

# Build if binary doesn't exist
if [ ! -f "bin/spmv_bench" ]; then
    echo -e "${BLUE}🔧 Building binaries...${NC}"
    make
    if [ $? -ne 0 ]; then
        echo "❌ Build failed"
        exit 1
    fi
fi

# Step 1: GPU Detection and Configuration
echo -e "${BLUE}🔍 Step 1: GPU Detection & Sizing${NC}"
echo "Detecting GPU configuration..."
./scripts/detect_gpu_config.sh

if [ ! -f "/tmp/gpu_config.env" ]; then
    echo "❌ GPU detection failed"
    exit 1
fi

source /tmp/gpu_config.env
echo -e "${GREEN}✅ GPU detected: $GPU_NAME${NC}"
echo -e "${GREEN}✅ Optimal matrix size: ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE}${NC}"

# Step 2: Matrix Generation
MATRIX_FILE="matrix/vastai_multimode_${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE}.mtx"
echo ""
echo -e "${BLUE}🔧 Step 2: Matrix Generation${NC}"

if [ -f "$MATRIX_FILE" ]; then
    echo "Matrix file already exists: $MATRIX_FILE"
    MATRIX_SIZE_MB=$(du -m "$MATRIX_FILE" | cut -f1)
    echo "Size: ${MATRIX_SIZE_MB}MB"
else
    echo "Generating ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE} stencil matrix..."
    echo "⏱️  This may take 1-3 minutes for large matrices..."
    echo "🔄 Progress will be shown below:"
    echo ""
    ./bin/generate_matrix $MAX_MATRIX_SIZE "$MATRIX_FILE" 2>&1 | tee /tmp/matrix_generation.log
    
    if [ $? -eq 0 ]; then
        MATRIX_SIZE_MB=$(du -m "$MATRIX_FILE" | cut -f1)
        echo -e "${GREEN}✅ Matrix generated successfully: ${MATRIX_SIZE_MB}MB${NC}"
    else
        echo "❌ Matrix generation failed"
        exit 1
    fi
fi

# Step 3: Multimode Benchmark (Single Matrix Load)
echo ""
echo -e "${BLUE}🏁 Step 3: Multimode Benchmark (Single Load)${NC}"
echo "Testing all SpMV implementations with single matrix load..."

# Generate result prefix with GPU name and timestamp
RESULT_PREFIX="vastai_multimode_$(echo "$GPU_NAME" | tr ' -' '_' | tr '[:upper:]' '[:lower:]')_$(date +%Y%m%d_%H%M)"

# Available modes (stencil5-shared excluded due to shared memory issues on large matrices)
ALL_MODES="csr,ellpack-naive,stencil5,stencil5-opt,stencil5-coarsened"

echo "Testing modes: $ALL_MODES"
echo "Matrix: $MATRIX_FILE"
echo "Result prefix: $RESULT_PREFIX"
echo ""

# Create results directory
mkdir -p results
cd results

# Run multimode benchmark (loads matrix once, tests all modes)
echo -e "${YELLOW}Running multimode benchmark...${NC}"
echo "This will test all 5 SpMV implementations with a single matrix load."

JSON_OUTPUT_FILE="${RESULT_PREFIX}_multimode.json"
HUMAN_OUTPUT_FILE="${RESULT_PREFIX}_multimode_report.txt"

# Run the multimode benchmark with live terminal output and file logging
../bin/spmv_bench "../$MATRIX_FILE" --mode=$ALL_MODES 2>&1 | tee "$HUMAN_OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Multimode benchmark completed successfully!${NC}"
    
    # Extract metrics from multimode output to generate individual JSON files
    echo ""
    echo -e "${BLUE}📊 Generating individual JSON reports from multimode data...${NC}"
    
    JSON_DIR="${RESULT_PREFIX}_json"
    mkdir -p "$JSON_DIR"
    
    # Parse the multimode output to extract individual metrics
    echo "   ➤ Parsing multimode results from report file..."
    
    # Extract performance metrics from the multimode report
    IFS=',' read -ra MODES <<< "$ALL_MODES"
    for mode in "${MODES[@]}"; do
        echo "  Extracting JSON for $mode..."
        JSON_FILE="${JSON_DIR}/${mode}_results.json"
        
        # Extract metrics from multimode report using grep and awk
        # Look for the performance metrics section for this operator
        EXEC_TIME=$(grep -A 10 "Operator: $mode" "$HUMAN_OUTPUT_FILE" | grep "Execution time:" | awk '{print $3}' | sed 's/ms//')
        GFLOPS=$(grep -A 10 "Operator: $mode" "$HUMAN_OUTPUT_FILE" | grep "GFLOPS:" | awk '{print $2}')
        BANDWIDTH=$(grep -A 10 "Operator: $mode" "$HUMAN_OUTPUT_FILE" | grep "Memory bandwidth:" | awk '{print $3}' | sed 's/GB\/s//')
        
        # Use default values if extraction fails
        EXEC_TIME=${EXEC_TIME:-0.0}
        GFLOPS=${GFLOPS:-0.0}  
        BANDWIDTH=${BANDWIDTH:-0.0}
        
        # Create JSON with extracted metrics
        cat > "$JSON_FILE" << EOF
{
  "operator": "$mode",
  "matrix_info": {
    "filename": "$MATRIX_FILE",
    "rows": $(grep "Matrix loaded:" "$HUMAN_OUTPUT_FILE" | head -1 | awk '{print $3}'),
    "cols": $(grep "Matrix loaded:" "$HUMAN_OUTPUT_FILE" | head -1 | awk '{print $5}'),
    "nnz": $(grep "Matrix loaded:" "$HUMAN_OUTPUT_FILE" | head -1 | awk '{print $7}')
  },
  "execution_time_ms": $EXEC_TIME,
  "gflops": $GFLOPS,
  "bandwidth_gb_s": $BANDWIDTH,
  "note": "Metrics extracted from multimode execution - no matrix reload"
}
EOF
    done
    
    echo -e "${GREEN}✅ JSON placeholders generated (extracted from multimode run)${NC}"
    
    # Generate CSV summary for compatibility with existing visualization tools
    CSV_FILE="${RESULT_PREFIX}_summary.csv"
    echo "operator,time_ms,gflops,bandwidth" > "$CSV_FILE"
    
    for mode in "${MODES[@]}"; do
        JSON_FILE="${JSON_DIR}/${mode}_results.json"
        if [ -f "$JSON_FILE" ]; then
            time_ms=$(grep -o '"execution_time_ms": [0-9.]*' "$JSON_FILE" | sed 's/"execution_time_ms": //')
            gflops=$(grep -o '"gflops": [0-9.]*' "$JSON_FILE" | sed 's/"gflops": //')
            bandwidth=$(grep -o '"bandwidth_gb_s": [0-9.]*' "$JSON_FILE" | sed 's/"bandwidth_gb_s": //')
            
            if [ -n "$time_ms" ] && [ -n "$gflops" ] && [ -n "$bandwidth" ]; then
                echo "$mode,$time_ms,$gflops,$bandwidth" >> "$CSV_FILE"
            fi
        fi
    done
    
    # Generate performance visualization
    if [ -f "$CSV_FILE" ] && [ $(wc -l < "$CSV_FILE") -gt 1 ]; then
        echo -e "${BLUE}📈 Generating performance charts...${NC}"
        
        MATRIX_BASENAME=$(basename "$MATRIX_FILE" .mtx)
        CHART_TITLE="SpMV Multimode Performance: $MATRIX_BASENAME"
        
        ../scripts/generate_performance_chart.sh "$CSV_FILE" "${RESULT_PREFIX}_performance" "$CHART_TITLE"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Charts generated successfully!${NC}"
        else
            echo -e "${YELLOW}Warning: Chart generation failed${NC}"
        fi
    fi
    
else
    echo -e "${RED}❌ Multimode benchmark failed${NC}"
    cat "$HUMAN_OUTPUT_FILE"
    cd ..
    exit 1
fi

cd ..

echo ""
echo -e "${GREEN}🎉 VastAI Multimode Benchmark Pipeline Completed Successfully!${NC}"
echo ""
echo -e "${YELLOW}📊 Results Summary:${NC}"
echo "- Matrix: ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE} (${MATRIX_SIZE_MB}MB)"
echo "- GPU: $GPU_NAME"
echo "- Modes tested: 5 ($(echo $ALL_MODES | tr ',' ' '))"
echo "- Single matrix load: ✅ (no reloading between modes)"
echo ""

echo -e "${YELLOW}📁 Generated Files:${NC}"
echo "Main results:"
ls -la results/${RESULT_PREFIX}*

echo ""
echo "JSON details:"
if [ -d "results/${RESULT_PREFIX}_json" ]; then
    ls -la results/${RESULT_PREFIX}_json/
fi

echo ""
echo -e "${YELLOW}📈 Performance Comparison:${NC}"
if [ -f "results/$HUMAN_OUTPUT_FILE" ]; then
    echo "View detailed comparison in: results/$HUMAN_OUTPUT_FILE"
    echo ""
    echo "Quick summary from multimode run:"
    grep -A 20 "Performance Metrics" "results/$HUMAN_OUTPUT_FILE" | head -15
fi

echo ""
echo -e "${YELLOW}⬇️ Download Instructions:${NC}"
echo "Use 'scp' or VastAI file sync to download results/ directory"
echo "Key files:"
echo "  - Multimode report: results/${RESULT_PREFIX}_multimode_report.txt"
echo "  - CSV summary: results/${RESULT_PREFIX}_summary.csv"
echo "  - JSON details: results/${RESULT_PREFIX}_json/"
echo "  - Performance charts: results/${RESULT_PREFIX}_performance.{png,svg}"

echo ""
echo -e "${GREEN}🚀 Advantages of Multimode Benchmark:${NC}"
echo "  ✅ Single matrix load (faster execution)"
echo "  ✅ Consistent memory usage across modes"
echo "  ✅ Fair performance comparison"
echo "  ✅ Reduced I/O overhead"
echo "  ✅ Compatible with existing visualization tools"
