#!/usr/bin/env python3
"""
Generate multi-GPU scaling plots for 15k×15k stencil (225M unknowns).
Outputs: scaling_main_a100.png, scaling_detailed_a100.png

Uses linear x-axis so visual spacing reflects actual GPU count ratios.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Data: 15000×15000 stencil, 14 CG iterations, A100-SXM4-80GB ---
gpus = [1, 2, 4, 8]
times_ms = [300.1, 152.5, 77.7, 40.4]
speedups = [1.00, 1.97, 3.86, 7.43]
efficiency = [100.0, 98.4, 96.5, 92.9]
overhead = [0.0, 100 - 98.4, 100 - 96.5, 100 - 92.9]  # comm overhead %
iterations = 14
iter_cost = [t / iterations for t in times_ms]

# Ideal reference
ideal_speedup = [1, 2, 4, 8]

# --- Style ---
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

color_time = '#2E86AB'
color_measured = '#A23B72'
color_ideal = '#888888'
color_efficiency = '#F18F01'
color_overhead = ['#B8D4E3', '#A23B72', '#F18F01', '#E74C3C']

# ============================================================================
# Figure 1: scaling_main_a100.png (3-panel overview)
# ============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Total Time
ax1.plot(gpus, times_ms, 'o-', color=color_time, linewidth=2.5, markersize=8, zorder=5)
for g, t in zip(gpus, times_ms):
    ax1.annotate(f'{t:.1f}ms', (g, t), textcoords='offset points',
                 xytext=(8, 8), fontsize=10, fontweight='bold')
ax1.set_xlabel('Number of GPUs', fontweight='bold')
ax1.set_ylabel('Total Time (ms)', fontweight='bold')
ax1.set_title('CG Solver Total Time\n15000×15000 stencil, 225M unknowns',
              fontweight='bold')
ax1.set_xticks(gpus)
ax1.set_xlim(0, 9)
ax1.set_ylim(0, 340)
ax1.grid(True, alpha=0.3, linestyle='--')

# Panel 2: Speedup
ax2.plot(gpus, ideal_speedup, '--', color=color_ideal, linewidth=2,
         label='Ideal (linear)', zorder=3)
ax2.plot(gpus, speedups, 'o-', color=color_measured, linewidth=2.5,
         markersize=8, label='Measured', zorder=5)
for g, s in zip(gpus, speedups):
    ax2.annotate(f'{s:.2f}×', (g, s), textcoords='offset points',
                 xytext=(-15, -18), fontsize=10, fontweight='bold',
                 color=color_measured,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor=color_measured, alpha=0.9))
ax2.set_xlabel('Number of GPUs', fontweight='bold')
ax2.set_ylabel('Speedup', fontweight='bold')
ax2.set_title('Strong Scaling Speedup\nA100-SXM4-80GB GPUs', fontweight='bold')
ax2.set_xticks(gpus)
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 9)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--')

# Panel 3: Parallel Efficiency
ax3.plot(gpus, efficiency, 'o-', color=color_efficiency, linewidth=2.5,
         markersize=8, zorder=5)
ax3.axhline(y=100, color=color_ideal, linestyle='--', linewidth=2,
            label='Ideal (100%)', zorder=3)
for g, e in zip(gpus, efficiency):
    ax3.annotate(f'{e:.1f}%', (g, e), textcoords='offset points',
                 xytext=(8, 8), fontsize=10, fontweight='bold')
ax3.set_xlabel('Number of GPUs', fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
ax3.set_title('Parallel Efficiency\nSpeedup / # GPUs × 100', fontweight='bold')
ax3.set_xticks(gpus)
ax3.set_xlim(0, 9)
ax3.set_ylim(0, 110)
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('docs/figures/scaling_main_a100.png', dpi=300, bbox_inches='tight')
print('Generated: docs/figures/scaling_main_a100.png')
plt.close()

# ============================================================================
# Figure 2: scaling_detailed_a100.png (4-panel detailed)
# ============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

bar_colors = ['#5DA5DA', '#FAA43A', '#F17CB0', '#B276B2']
gpu_labels = ['1', '2', '4', '8']
x_cat = np.arange(len(gpus))  # categorical positions for bar charts

# Panel 1: Total Solver Time (bar chart - categorical x)
bars = ax1.bar(x_cat, times_ms, width=0.6, color=bar_colors,
               edgecolor='black', linewidth=0.8)
for i, t in enumerate(times_ms):
    ax1.text(i, t + 5, f'{t:.1f}ms\n({t/iterations:.1f}ms/iter)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xlabel('Number of GPUs', fontweight='bold')
ax1.set_ylabel('Total Time (ms)', fontweight='bold')
ax1.set_title(f'Total Solver Time\n15000×15000 Stencil, {iterations} Iterations',
              fontweight='bold')
ax1.set_xticks(x_cat)
ax1.set_xticklabels(gpu_labels)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Panel 2: Speedup with scaling gap
ax2.fill_between(gpus, speedups, ideal_speedup, alpha=0.15, color=color_measured,
                 label='Scaling gap', zorder=2)
ax2.plot(gpus, ideal_speedup, '--', color=color_ideal, linewidth=2,
         label='Ideal (linear)', zorder=3)
ax2.plot(gpus, speedups, 'o-', color=color_measured, linewidth=2.5,
         markersize=8, label='Measured', zorder=5)
for g, s in zip(gpus, speedups):
    ax2.annotate(f'{s:.2f}×', (g, s), textcoords='offset points',
                 xytext=(-15, -18), fontsize=10, fontweight='bold',
                 color=color_measured,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFCC',
                           edgecolor=color_measured, alpha=0.9))
ax2.set_xlabel('Number of GPUs', fontweight='bold')
ax2.set_ylabel('Speedup', fontweight='bold')
ax2.set_title('Strong Scaling Speedup\nA100-SXM4-80GB', fontweight='bold')
ax2.set_xticks(gpus)
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 9)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--')

# Panel 3: Parallel Efficiency (bar chart - categorical x)
bars3 = ax3.bar(x_cat, efficiency, width=0.6, color=bar_colors,
                edgecolor='black', linewidth=0.8)
ax3.axhline(y=100, color=color_ideal, linestyle='--', linewidth=2,
            label='Ideal (100%)', zorder=3)
for i, e in enumerate(efficiency):
    ax3.text(i, e + 1, f'{e:.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
ax3.set_xlabel('Number of GPUs', fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
ax3.set_title('Parallel Efficiency\n(Speedup / GPUs) × 100', fontweight='bold')
ax3.set_xticks(x_cat)
ax3.set_xticklabels(gpu_labels)
ax3.set_ylim(0, 110)
ax3.legend(loc='lower left')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

# Panel 4: Communication Overhead (bar chart - categorical x)
bars4 = ax4.bar(x_cat, overhead, width=0.6, color=bar_colors,
                edgecolor='black', linewidth=0.8)
for i, o in enumerate(overhead):
    ax4.text(i, o + 0.3, f'{o:.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
ax4.set_xlabel('Number of GPUs', fontweight='bold')
ax4.set_ylabel('Communication Overhead (%)', fontweight='bold')
ax4.set_title('Estimated Communication Overhead\n100% - Efficiency', fontweight='bold')
ax4.set_xticks(x_cat)
ax4.set_xticklabels(gpu_labels)
ax4.set_ylim(0, 15)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_axisbelow(True)

plt.tight_layout()
plt.savefig('docs/figures/scaling_detailed_a100.png', dpi=300, bbox_inches='tight')
print('Generated: docs/figures/scaling_detailed_a100.png')
plt.close()

print('\nDone: scaling plots with linear x-axis')
