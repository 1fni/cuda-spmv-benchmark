#!/usr/bin/env python3
"""
Generate problem-size scaling plots for 3 stencil sizes (10k, 15k, 20k).
Outputs: problem_size_scaling_overview.png, problem_size_scaling_detailed.png

Uses linear x-axis so visual spacing reflects actual GPU count ratios.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Data: A100-SXM4-80GB, 14 CG iterations all configs ---
gpus = [1, 2, 4, 8]
iterations = 14

# 10k×10k stencil (100M unknowns)
times_10k = [133.9, 68.7, 35.7, 19.3]
speedups_10k = [t0 / t for t0, t in zip([133.9]*4, times_10k)]
efficiency_10k = [s / g * 100 for s, g in zip(speedups_10k, gpus)]

# 15k×15k stencil (225M unknowns)
times_15k = [300.1, 152.5, 77.7, 40.4]
speedups_15k = [t0 / t for t0, t in zip([300.1]*4, times_15k)]
efficiency_15k = [s / g * 100 for s, g in zip(speedups_15k, gpus)]

# 20k×20k stencil (400M unknowns)
times_20k = [531.4, 269.3, 136.3, 71.0]
speedups_20k = [t0 / t for t0, t in zip([531.4]*4, times_20k)]
efficiency_20k = [s / g * 100 for s, g in zip(speedups_20k, gpus)]

# Iteration costs
iter_10k = [t / iterations for t in times_10k]
iter_15k = [t / iterations for t in times_15k]
iter_20k = [t / iterations for t in times_20k]

# --- Style ---
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

color_10k = '#2E86AB'   # Blue
color_15k = '#A23B72'   # Purple
color_20k = '#F18F01'   # Orange
color_ideal = '#888888'

# ============================================================================
# Figure 1: problem_size_scaling_overview.png (3-panel)
# ============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Total Time
ax1.plot(gpus, times_10k, 'o--', color=color_10k, linewidth=2, markersize=7,
         label='10k×10k (100M unknowns)')
ax1.plot(gpus, times_15k, 's-', color=color_15k, linewidth=2, markersize=7,
         label='15k×15k (225M unknowns)')
ax1.plot(gpus, times_20k, '^-', color=color_20k, linewidth=2, markersize=7,
         label='20k×20k (400M unknowns)')
ax1.set_xlabel('Number of GPUs', fontweight='bold')
ax1.set_ylabel('Total Time (ms)', fontweight='bold')
ax1.set_title('Strong Scaling: Total Time', fontweight='bold')
ax1.set_xticks(gpus)
ax1.set_xlim(0, 9)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')

# Panel 2: Speedup
ax2.plot(gpus, gpus, '--', color=color_ideal, linewidth=2, label='Ideal (100%)')
ax2.plot(gpus, speedups_10k, 'o--', color=color_10k, linewidth=2, markersize=7,
         label='10k×10k (100M unknowns)')
ax2.plot(gpus, speedups_15k, 's-', color=color_15k, linewidth=2, markersize=7,
         label='15k×15k (225M unknowns)')
ax2.plot(gpus, speedups_20k, '^-', color=color_20k, linewidth=2, markersize=7,
         label='20k×20k (400M unknowns)')
ax2.set_xlabel('Number of GPUs', fontweight='bold')
ax2.set_ylabel('Speedup', fontweight='bold')
ax2.set_title('Strong Scaling: Speedup', fontweight='bold')
ax2.set_xticks(gpus)
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 9)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')

# Panel 3: Efficiency
ax3.plot(gpus, efficiency_10k, 'o--', color=color_10k, linewidth=2, markersize=7,
         label='10k×10k (100M unknowns)')
ax3.plot(gpus, efficiency_15k, 's-', color=color_15k, linewidth=2, markersize=7,
         label='15k×15k (225M unknowns)')
ax3.plot(gpus, efficiency_20k, '^-', color=color_20k, linewidth=2, markersize=7,
         label='20k×20k (400M unknowns)')
ax3.axhline(y=100, color=color_ideal, linestyle='--', linewidth=2, label='Ideal')
ax3.set_xlabel('Number of GPUs', fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
ax3.set_title('Strong Scaling: Efficiency', fontweight='bold')
ax3.set_xticks(gpus)
ax3.set_xlim(0, 9)
ax3.set_ylim(80, 105)
ax3.legend(loc='lower left', fontsize=9)
ax3.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('docs/figures/problem_size_scaling_overview.png', dpi=300,
            bbox_inches='tight')
print('Generated: docs/figures/problem_size_scaling_overview.png')
plt.close()

# ============================================================================
# Figure 2: problem_size_scaling_detailed.png (4-panel)
# ============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

bar_colors = ['#5DA5DA', '#FAA43A', '#F17CB0', '#B276B2']
gpu_labels = ['1', '2', '4', '8']
x_cat = np.arange(len(gpus))  # categorical positions for bar charts

# Panel 1: Time by GPU Count (bar chart - categorical x)
bars1 = ax1.bar(x_cat, times_15k, width=0.6, color=bar_colors,
                edgecolor='black', linewidth=0.8)
for i, t in enumerate(times_15k):
    ax1.text(i, t + 5, f'{t:.1f} ms', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
ax1.set_xlabel('Number of GPUs', fontweight='bold')
ax1.set_ylabel('Total Time (ms)', fontweight='bold')
ax1.set_title('Time by GPU Count (15k×15k matrix)', fontweight='bold')
ax1.set_xticks(x_cat)
ax1.set_xticklabels(gpu_labels)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Panel 2: Speedup Comparison (all 3 sizes, line plot - linear x)
ax2.plot(gpus, gpus, '--', color=color_ideal, linewidth=2, label='Ideal')
ax2.plot(gpus, speedups_10k, 'o-', color=color_10k, linewidth=2, markersize=7,
         label='10k×10k (100M unknowns)')
ax2.plot(gpus, speedups_15k, 's-', color=color_15k, linewidth=2, markersize=7,
         label='15k×15k (225M unknowns)')
ax2.plot(gpus, speedups_20k, '^-', color=color_20k, linewidth=2, markersize=7,
         label='20k×20k (400M unknowns)')
ax2.set_xlabel('Number of GPUs', fontweight='bold')
ax2.set_ylabel('Speedup', fontweight='bold')
ax2.set_title('Speedup Comparison', fontweight='bold')
ax2.set_xticks(gpus)
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 9)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')

# Panel 3: Parallel Efficiency (grouped bar chart - categorical x)
width_bar = 0.2
offsets = [-width_bar, 0, width_bar]

for i, (eff, color, label) in enumerate([
    (efficiency_10k, color_10k, '10k×10k (100M unknowns)'),
    (efficiency_15k, color_15k, '15k×15k (225M unknowns)'),
    (efficiency_20k, color_20k, '20k×20k (400M unknowns)'),
]):
    ax3.bar(x_cat + offsets[i], eff, width=width_bar, color=color,
            edgecolor='black', linewidth=0.5, label=label)

ax3.axhline(y=100, color=color_ideal, linestyle='--', linewidth=1.5)
ax3.set_xlabel('Number of GPUs', fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
ax3.set_title('Parallel Efficiency Comparison', fontweight='bold')
ax3.set_xticks(x_cat)
ax3.set_xticklabels(gpu_labels)
ax3.set_ylim(0, 110)
ax3.legend(loc='lower left', fontsize=9)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

# Panel 4: Iteration Cost vs GPU Count (line plot - linear x)
ax4.plot(gpus, iter_10k, 'o-', color=color_10k, linewidth=2, markersize=7,
         label='10k×10k (100M unknowns)')
ax4.plot(gpus, iter_15k, 's-', color=color_15k, linewidth=2, markersize=7,
         label='15k×15k (225M unknowns)')
ax4.plot(gpus, iter_20k, '^-', color=color_20k, linewidth=2, markersize=7,
         label='20k×20k (400M unknowns)')
ax4.set_xlabel('Number of GPUs', fontweight='bold')
ax4.set_ylabel('Time per Iteration (ms)', fontweight='bold')
ax4.set_title('Iteration Cost vs GPU Count', fontweight='bold')
ax4.set_xticks(gpus)
ax4.set_xlim(0, 9)
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('docs/figures/problem_size_scaling_detailed.png', dpi=300,
            bbox_inches='tight')
print('Generated: docs/figures/problem_size_scaling_detailed.png')
plt.close()

print('\nDone: problem size scaling plots with linear x-axis')
