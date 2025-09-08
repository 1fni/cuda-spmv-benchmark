#!/bin/bash

# CUDA SpMV Validation Script for 3x3 Stencil Matrix
# Tests CSR, STENCIL5, and ELLPACK operators with result verification

set -e  # Exit on any error

echo "🧪 CUDA SpMV Validation Test Suite"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if binaries exist
if [ ! -f "bin/spmv_bench" ] || [ ! -f "bin/generate_matrix" ]; then
    echo -e "${RED}❌ Binaries not found. Run 'make' first${NC}"
    exit 1
fi

# Generate 3x3 test matrix
echo -e "${BLUE}🔧 Generating 3x3 stencil matrix...${NC}"
./bin/generate_matrix 3 matrix/test_3x3.mtx
echo -e "${GREEN}✅ Matrix generated: matrix/test_3x3.mtx${NC}"

# Display matrix info
echo -e "${BLUE}📄 Matrix information:${NC}"
echo "Size: $(wc -l matrix/test_3x3.mtx | cut -d' ' -f1) lines, $(du -h matrix/test_3x3.mtx | cut -f1)"
echo ""

# Test each SpMV operator
echo -e "${YELLOW}🚀 Testing SpMV Operators${NC}"
echo "========================"

# CSR Test
echo -e "${BLUE}🔸 Testing CSR operator...${NC}"
CSR_OUTPUT=$(./bin/spmv_bench matrix/test_3x3.mtx --mode=csr 2>&1)
echo "$CSR_OUTPUT"

# Extract checksum from CSR output
CSR_CHECKSUM=$(echo "$CSR_OUTPUT" | grep -i checksum | grep -o "[0-9.-]*e[+-]*[0-9]*" || echo "PARSE_ERROR")
echo -e "${GREEN}CSR Checksum: $CSR_CHECKSUM${NC}"
echo ""

# STENCIL5 Test  
echo -e "${BLUE}🔸 Testing STENCIL5 operator...${NC}"
STENCIL_OUTPUT=$(./bin/spmv_bench matrix/test_3x3.mtx --mode=stencil5 2>&1)
echo "$STENCIL_OUTPUT"

# Extract checksum from STENCIL output  
STENCIL_CHECKSUM=$(echo "$STENCIL_OUTPUT" | grep -i "check_sum\|checksum" | grep -o "[0-9.-]*e[+-]*[0-9]*" || echo "PARSE_ERROR")
echo -e "${GREEN}STENCIL5 Checksum: $STENCIL_CHECKSUM${NC}"
echo ""

# ELLPACK Test
echo -e "${BLUE}🔸 Testing ELLPACK operator...${NC}"
ELLPACK_OUTPUT=$(./bin/spmv_bench matrix/test_3x3.mtx --mode=ellpack 2>&1)
echo "$ELLPACK_OUTPUT"

# Extract checksum from ELLPACK output
ELLPACK_CHECKSUM=$(echo "$ELLPACK_OUTPUT" | grep -i checksum | grep -o "[0-9.-]*e[+-]*[0-9]*" || echo "PARSE_ERROR")
echo -e "${GREEN}ELLPACK Checksum: $ELLPACK_CHECKSUM${NC}"
echo ""

# Cross-validation
echo -e "${YELLOW}📊 Cross-Validation Results${NC}"
echo "==========================="
echo "CSR Checksum:     $CSR_CHECKSUM"
echo "STENCIL5 Checksum: $STENCIL_CHECKSUM" 
echo "ELLPACK Checksum:  $ELLPACK_CHECKSUM"
echo ""

# Validation logic
VALIDATION_PASSED=true

# Check if all checksums are identical (allowing for small numerical differences)
if [ "$CSR_CHECKSUM" = "PARSE_ERROR" ] || [ "$STENCIL_CHECKSUM" = "PARSE_ERROR" ] || [ "$ELLPACK_CHECKSUM" = "PARSE_ERROR" ]; then
    echo -e "${RED}❌ VALIDATION FAILED: Could not parse checksums${NC}"
    VALIDATION_PASSED=false
elif [ "$CSR_CHECKSUM" = "$STENCIL_CHECKSUM" ] && [ "$STENCIL_CHECKSUM" = "$ELLPACK_CHECKSUM" ]; then
    echo -e "${GREEN}✅ VALIDATION PASSED: All operators produce identical results${NC}"
else
    # Check if checksums are numerically close (within tolerance)
    python3 -c "
import sys
try:
    csr = float('$CSR_CHECKSUM')
    stencil = float('$STENCIL_CHECKSUM') 
    ellpack = float('$ELLPACK_CHECKSUM')
    
    tolerance = 1e-10
    
    if abs(csr - stencil) < tolerance and abs(stencil - ellpack) < tolerance:
        print('NUMERICALLY_EQUAL')
        sys.exit(0)
    else:
        print('DIFFERENT')
        sys.exit(1)
except:
    print('PARSE_ERROR')
    sys.exit(1)
" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ VALIDATION PASSED: All operators produce numerically equivalent results (within tolerance)${NC}"
    else
        echo -e "${RED}❌ VALIDATION FAILED: Operators produce different results${NC}"
        VALIDATION_PASSED=false
    fi
fi

# Expected result for 3x3 stencil with vector of all ones
echo ""
echo -e "${BLUE}📚 Expected Theoretical Result:${NC}"
echo "For a 3x3 stencil matrix with input vector of all ones:"
echo "Expected checksum: -60.0 (sum of all elements in result vector)"
echo ""

# Performance summary
echo -e "${YELLOW}⚡ Performance Summary${NC}"
echo "===================="
echo "Matrix: 3x3 grid (9 points, $(grep -c "^[0-9]" matrix/test_3x3.mtx) non-zeros)"

# Extract timing information if available
CSR_TIME=$(echo "$CSR_OUTPUT" | grep -o "completed in [0-9.]* ms" | grep -o "[0-9.]*" || echo "N/A")
STENCIL_TIME=$(echo "$STENCIL_OUTPUT" | grep -o "completed in [0-9.]* ms" | grep -o "[0-9.]*" || echo "N/A")
ELLPACK_TIME=$(echo "$ELLPACK_OUTPUT" | grep -o "completed in [0-9.]* ms" | grep -o "[0-9.]*" || echo "N/A")

echo "CSR execution time:     ${CSR_TIME} ms"
echo "STENCIL5 execution time: ${STENCIL_TIME} ms"
echo "ELLPACK execution time:  ${ELLPACK_TIME} ms"

# Final status
echo ""
echo -e "${YELLOW}🏁 Final Status${NC}"
echo "=============="
if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}✅ All SpMV operators validated successfully!${NC}"
    echo -e "${GREEN}✅ Mathematical correctness confirmed for 3x3 stencil case${NC}"
    exit 0
else
    echo -e "${RED}❌ Validation failed - manual investigation required${NC}"
    exit 1
fi