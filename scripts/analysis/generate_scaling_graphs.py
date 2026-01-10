#!/usr/bin/env python3
"""
Generate scaling graphs for A100 benchmarks - Main branch only
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Data from A100 benchmarks (15000×15000, 225M unknowns)
gpus = [1, 2, 4, 8]
time_ms = [299.53, 152.23, 77.72, 40.30]
iterations = [14, 14, 14, 14]

# Calculate metrics
time_per_iter = [t/i for t, i in zip(time_ms, iterations)]
speedup = [time_ms[0]/t for t in time_ms]
efficiency = [s/g * 100 for s, g in zip(speedup, gpus)]
ideal_speedup = gpus

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# 1. Total Time
ax1.plot(gpus, time_ms, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Measured')
ax1.set_xlabel('Number of GPUs', fontsize=12)
ax1.set_ylabel('Total Time (ms)', fontsize=12)
ax1.set_title('CG Solver Total Time\n15000×15000 stencil, 225M unknowns', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(gpus)
for i, (g, t) in enumerate(zip(gpus, time_ms)):
    ax1.annotate(f'{t:.1f}ms', (g, t), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=10)

# 2. Speedup
ax2.plot(gpus, speedup, 'o-', linewidth=2, markersize=8, color='#A23B72', label='Measured')
ax2.plot(gpus, ideal_speedup, '--', linewidth=2, color='#666666', label='Ideal (linear)')
ax2.set_xlabel('Number of GPUs', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('Strong Scaling Speedup\nA100-SXM4-80GB GPUs', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(gpus)
for i, (g, s) in enumerate(zip(gpus, speedup)):
    ax2.annotate(f'{s:.2f}×', (g, s), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=10)

# 3. Parallel Efficiency
ax3.plot(gpus, efficiency, 'o-', linewidth=2, markersize=8, color='#F18F01')
ax3.axhline(y=100, linestyle='--', color='#666666', linewidth=2, label='Ideal (100%)')
ax3.set_xlabel('Number of GPUs', fontsize=12)
ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax3.set_title('Parallel Efficiency\nSpeedup / # GPUs × 100', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(gpus)
ax3.set_ylim([0, 110])
for i, (g, e) in enumerate(zip(gpus, efficiency)):
    ax3.annotate(f'{e:.1f}%', (g, e), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('scaling_main_a100.png', dpi=300, bbox_inches='tight')
print("✅ Generated: scaling_main_a100.png")

# Generate summary table
print("\n" + "="*60)
print("Strong Scaling Results - Main Branch (A100)")
print("="*60)
print(f"{'GPUs':<8} {'Time (ms)':<12} {'Speedup':<12} {'Efficiency':<12}")
print("-"*60)
for g, t, s, e in zip(gpus, time_ms, speedup, efficiency):
    print(f"{g:<8} {t:<12.2f} {s:<12.2f}× {e:<12.1f}%")
print("="*60)
print(f"Matrix: 15000×15000 stencil (225M unknowns, 1.125B nnz)")
print(f"Convergence: 14 iterations (consistent across all GPU counts)")
print(f"Hardware: NVIDIA A100-SXM4-80GB")
