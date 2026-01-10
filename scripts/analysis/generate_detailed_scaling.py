#!/usr/bin/env python3
"""
Generate additional detailed scaling graphs
"""
import matplotlib.pyplot as plt
import numpy as np

gpus = [1, 2, 4, 8]
time_ms = [299.53, 152.23, 77.72, 40.30]
iterations = [14, 14, 14, 14]

# Create single larger figure with better layout
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. Time breakdown (top left)
ax1 = fig.add_subplot(gs[0, 0])
time_per_iter = [t/14 for t in time_ms]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax1.bar(range(len(gpus)), time_ms, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Number of GPUs', fontsize=11, fontweight='bold')
ax1.set_ylabel('Total Time (ms)', fontsize=11, fontweight='bold')
ax1.set_title('Total Solver Time\n15000×15000 Stencil, 14 Iterations', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(gpus)))
ax1.set_xticklabels(gpus)
ax1.grid(axis='y', alpha=0.3)
for i, (bar, t) in enumerate(zip(bars, time_ms)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{t:.1f}ms\n({time_per_iter[i]:.1f}ms/iter)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Speedup comparison (top right)
ax2 = fig.add_subplot(gs[0, 1])
speedup = [time_ms[0]/t for t in time_ms]
ideal = gpus
ax2.plot(gpus, speedup, 'o-', linewidth=3, markersize=10, color='#A23B72', label='Measured', zorder=3)
ax2.plot(gpus, ideal, '--', linewidth=2, color='#666666', label='Ideal (linear)', zorder=2)
ax2.fill_between(gpus, speedup, ideal, alpha=0.2, color='red', label='Scaling gap', zorder=1)
ax2.set_xlabel('Number of GPUs', fontsize=11, fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=11, fontweight='bold')
ax2.set_title('Strong Scaling Speedup\nA100-SXM4-80GB', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(gpus)
for g, s in zip(gpus, speedup):
    ax2.annotate(f'{s:.2f}×', (g, s), textcoords="offset points", 
                 xytext=(0,8), ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 3. Efficiency (bottom left)
ax3 = fig.add_subplot(gs[1, 0])
efficiency = [s/g * 100 for s, g in zip(speedup, gpus)]
bars = ax3.bar(range(len(gpus)), efficiency, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axhline(y=100, linestyle='--', color='#666666', linewidth=2, label='Ideal (100%)', zorder=1)
ax3.set_xlabel('Number of GPUs', fontsize=11, fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='bold')
ax3.set_title('Parallel Efficiency\n(Speedup / GPUs) × 100', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticks(range(len(gpus)))
ax3.set_xticklabels(gpus)
ax3.set_ylim([0, 110])
for i, (bar, e) in enumerate(zip(bars, efficiency)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{e:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Communication overhead estimation (bottom right)
ax4 = fig.add_subplot(gs[1, 1])
overhead_pct = [(1 - s/g) * 100 for s, g in zip(speedup, gpus)]
bars = ax4.bar(range(len(gpus)), overhead_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Number of GPUs', fontsize=11, fontweight='bold')
ax4.set_ylabel('Communication Overhead (%)', fontsize=11, fontweight='bold')
ax4.set_title('Estimated Communication Overhead\n100% - Efficiency', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.set_xticks(range(len(gpus)))
ax4.set_xticklabels(gpus)
ax4.set_ylim([0, 15])
for i, (bar, o) in enumerate(zip(bars, overhead_pct)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{o:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.savefig('scaling_detailed_a100.png', dpi=300, bbox_inches='tight')
print("✅ Generated: scaling_detailed_a100.png")
print("\n" + "="*60)
print("Detailed Scaling Analysis")
print("="*60)
print(f"{'GPUs':<6} {'Time/iter (ms)':<16} {'Overhead (%)':<14} {'Efficiency (%)':<14}")
print("-"*60)
for g, tpi, o, e in zip(gpus, time_per_iter, overhead_pct, efficiency):
    print(f"{g:<6} {tpi:<16.2f} {o:<14.1f} {e:<14.1f}")
