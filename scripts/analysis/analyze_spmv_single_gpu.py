#!/usr/bin/env python3
"""
Analyze single-GPU SpMV format comparison (CSR vs STENCIL5)
Reads results from results_spmv_a100_manual.json
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Read results
with open('docs/results_spmv_a100_manual.json') as f:
    data = json.load(f)

gpu = data['benchmark_info']['gpu']
results = data['results']

# Extract data by format
sizes = [r['size'] for r in results]
csr_times = [r['csr']['time_ms'] for r in results]
stencil_times = [r['stencil5']['time_ms'] for r in results]
csr_gflops = [r['csr']['gflops'] for r in results]
stencil_gflops = [r['stencil5']['gflops'] for r in results]
csr_bw = [r['csr']['bandwidth_gb_s'] for r in results]
stencil_bw = [r['stencil5']['bandwidth_gb_s'] for r in results]
speedups = [r['speedup'] for r in results]

# Create 4-panel comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

colors_csr = '#2E86AB'
colors_stencil = '#F18F01'

# Plot 1: Execution time
ax1.plot(sizes, csr_times, marker='o', linestyle='-', label='CSR (cuSPARSE)',
         color=colors_csr, linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white')
ax1.plot(sizes, stencil_times, marker='s', linestyle='-', label='STENCIL5 (Custom)',
         color=colors_stencil, linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white')
ax1.set_xlabel('Matrix Size (N×N)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
ax1.set_title(f'SpMV Execution Time - {gpu}', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=12, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)

# Plot 2: GFLOPS
ax2.plot(sizes, csr_gflops, marker='o', linestyle='-', label='CSR (cuSPARSE)',
         color=colors_csr, linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white')
ax2.plot(sizes, stencil_gflops, marker='s', linestyle='-', label='STENCIL5 (Custom)',
         color=colors_stencil, linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white')
ax2.set_xlabel('Matrix Size (N×N)', fontsize=13, fontweight='bold')
ax2.set_ylabel('GFLOPS', fontsize=13, fontweight='bold')
ax2.set_title('Computational Performance', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=12, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(bottom=0)

# Plot 3: Memory bandwidth
ax3.plot(sizes, csr_bw, marker='o', linestyle='-', label='CSR (cuSPARSE)',
         color=colors_csr, linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white')
ax3.plot(sizes, stencil_bw, marker='s', linestyle='-', label='STENCIL5 (Custom)',
         color=colors_stencil, linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white')
ax3.set_xlabel('Matrix Size (N×N)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Memory Bandwidth (GB/s)', fontsize=13, fontweight='bold')
ax3.set_title('Memory Performance', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=12, frameon=True, shadow=True)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(bottom=0)

# Plot 4: Speedup
ax4.plot(sizes, speedups, marker='o', linestyle='-',
         color='#C73E1D', linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white')
ax4.axhline(y=1.0, color='k', linestyle='--', linewidth=2, alpha=0.6, label='Baseline (CSR)')
ax4.set_xlabel('Matrix Size (N×N)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Speedup vs CSR', fontsize=13, fontweight='bold')
ax4.set_title('STENCIL5 Speedup over CSR', fontsize=14, fontweight='bold', pad=15)
ax4.legend(fontsize=12, frameon=True, shadow=True)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_ylim([0, max(speedups) * 1.3])

# Add value labels on speedup plot
for x, y in zip(sizes, speedups):
    ax4.text(x, y + 0.05, f'{y:.2f}×', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('spmv_format_comparison_a100.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Generated: spmv_format_comparison_a100.png")

# Print summary table
print("\n" + "="*100)
print(f"Single-GPU SpMV Format Comparison - {gpu}")
print("="*100)
print(f"{'Size':<10} {'Format':<15} {'Time (ms)':<12} {'GFLOPS':<12} {'Bandwidth (GB/s)':<18} {'Speedup':<10}")
print("-"*100)

for r in results:
    size = r['size']
    unknowns = r['unknowns']

    # CSR row
    print(f"{size:<10} {'CSR':<15} {r['csr']['time_ms']:<12.3f} "
          f"{r['csr']['gflops']:<12.1f} {r['csr']['bandwidth_gb_s']:<18.1f} {'1.00×':<10}")

    # STENCIL5 row
    print(f"{size:<10} {'STENCIL5':<15} {r['stencil5']['time_ms']:<12.3f} "
          f"{r['stencil5']['gflops']:<12.1f} {r['stencil5']['bandwidth_gb_s']:<18.1f} "
          f"{r['speedup']:.2f}×")

    print(f"{'':10} {'('}{unknowns:,} unknowns, {r['nnz']:,} nnz)")
    print("-"*100)

print("="*100)
print(f"\nAverage speedup (STENCIL5 vs CSR): {np.mean(speedups):.2f}×")
print(f"Bandwidth improvement: {np.mean([s['bandwidth_gb_s'] for s in [r['stencil5'] for r in results]]) / np.mean([s['bandwidth_gb_s'] for s in [r['csr'] for r in results]]):.2f}×")
print("="*100)
