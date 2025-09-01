#!/bin/bash

# Generate performance comparison charts using gnuplot
# Usage: ./generate_performance_chart.sh <data_file> <output_name> [chart_title]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_file> <output_name> [chart_title]"
    echo "  data_file: CSV file with operator,gflops,bandwidth columns"
    echo "  output_name: Output filename (without extension)"
    echo "  chart_title: Optional chart title"
    exit 1
fi

DATA_FILE="$1"
OUTPUT_NAME="$2"
CHART_TITLE="${3:-SpMV Performance Comparison}"

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file $DATA_FILE not found"
    exit 1
fi

# Check if gnuplot is available
if ! command -v gnuplot &> /dev/null; then
    echo "Error: gnuplot not found. Please install gnuplot."
    exit 1
fi

# Create gnuplot script for performance comparison
cat > performance_chart.gp << EOF
#!/usr/bin/gnuplot

# Set CSV data format
set datafile separator ','

# Set output format and filename
set terminal pngcairo enhanced color font "Arial,12" size 1200,800
set output "${OUTPUT_NAME}.png"

# Chart styling
set title "${CHART_TITLE}" font "Arial,16" 
set xlabel "SpMV Implementation" font "Arial,12"
set ylabel "Performance" font "Arial,12"

# Grid and styling
set grid ytics
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.8 border -1
set boxwidth 0.8

# Color scheme
set palette defined (1 '#1f77b4', 2 '#ff7f0e', 3 '#2ca02c')

# Margins and layout
set lmargin 10
set rmargin 5
set bmargin 5
set tmargin 3

# Y-axis formatting
set format y "%.1f"

# X-axis labels rotation for better readability
set xtics rotate by -30

# Multi-plot layout for GFLOPS and Bandwidth
set multiplot layout 2,1 title "${CHART_TITLE}" font "Arial,18"

# Top plot: GFLOPS
set ylabel "GFLOPS" font "Arial,12"
set title "GFLOPS Performance" font "Arial,14"
set yrange [0:*]

# Set colors for bars
set style histogram clustered gap 1
set style fill solid 0.7

plot '$DATA_FILE' every ::1 using 2:xtic(1) with histogram title "GFLOPS" lc rgb '#1f77b4'

# Bottom plot: Memory Bandwidth
set ylabel "Memory Bandwidth (GB/s)" font "Arial,12" 
set title "Memory Bandwidth Utilization" font "Arial,14"
set xlabel "SpMV Implementation" font "Arial,12"

plot '$DATA_FILE' every ::1 using 3:xtic(1) with histogram title "Bandwidth" lc rgb '#ff7f0e'

unset multiplot
EOF

# Generate the chart
gnuplot performance_chart.gp

if [ $? -eq 0 ]; then
    echo "Performance chart generated: ${OUTPUT_NAME}.png"
else
    echo "Error generating chart"
    exit 1
fi

# Generate SVG version as well
cat > performance_chart_svg.gp << EOF
#!/usr/bin/gnuplot

# Set CSV data format
set datafile separator ','

set terminal svg enhanced size 1200,800 font "Arial,12"
set output "${OUTPUT_NAME}.svg"

# Chart styling
set title "${CHART_TITLE}" font "Arial,16"
set xlabel "SpMV Implementation" font "Arial,12"
set ylabel "Performance" font "Arial,12"

# Grid and styling
set grid ytics
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.8 border -1
set boxwidth 0.8

# Margins and layout
set lmargin 10
set rmargin 5
set bmargin 5
set tmargin 3

# Y-axis formatting
set format y "%.1f"
set xtics rotate by -30

# Multi-plot for both metrics
set multiplot layout 2,1 title "${CHART_TITLE}" font "Arial,18"

# GFLOPS plot
set ylabel "GFLOPS" font "Arial,12"
set title "GFLOPS Performance" font "Arial,14"
set yrange [0:*]

plot '$DATA_FILE' every ::1 using 2:xtic(1) with histogram title "GFLOPS" lc rgb '#1f77b4'

# Bandwidth plot
set ylabel "Memory Bandwidth (GB/s)" font "Arial,12"
set title "Memory Bandwidth Utilization" font "Arial,14" 
set xlabel "SpMV Implementation" font "Arial,12"

plot '$DATA_FILE' every ::1 using 3:xtic(1) with histogram title "Bandwidth" lc rgb '#ff7f0e'

unset multiplot
EOF

gnuplot performance_chart_svg.gp

if [ $? -eq 0 ]; then
    echo "SVG version generated: ${OUTPUT_NAME}.svg"
fi

# Clean up temporary files
rm -f performance_chart.gp performance_chart_svg.gp

echo "Chart generation complete!"
echo "Files created:"
echo "  - ${OUTPUT_NAME}.png (for presentations/reports)"
echo "  - ${OUTPUT_NAME}.svg (vector format for publications)"