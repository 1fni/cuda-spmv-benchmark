#!/usr/bin/env python3
"""
Generate roofline plot comparing cuSPARSE CSR vs Custom Stencil SpMV.

Data extracted from Nsight Compute profiles (RTX 4060 Laptop GPU, 7000x7000 matrix).
"""

import matplotlib.pyplot as plt
import numpy as np

# RTX 4060 Laptop GPU (AD107) specifications (from nvidia-smi / datasheet)
PEAK_MEMORY_BW = 256  # GB/s (theoretical peak)
PEAK_FP64 = 112       # GFLOP/s (FP64 is 1/64 of FP32 on consumer GPUs)

# Problem: 7000x7000 stencil = 49M rows, 245M nnz, 490M FLOPs per SpMV
GFLOPS_PER_SPMV = 0.49

# Measured data from NCU profiles
kernels = {
    'cuSPARSE CSR': {
        'duration_ms': 22.99,
        'memory_throughput_pct': 67.0,
        'color': '#e74c3c',  # Red
        'marker': 's',
    },
    'Custom Stencil': {
        'duration_ms': 11.25,
        'memory_throughput_pct': 95.0,
        'color': '#27ae60',  # Green
        'marker': 'o',
    },
}

def compute_metrics(kernel_data):
    """Compute performance and arithmetic intensity from NCU data."""
    duration_s = kernel_data['duration_ms'] / 1000.0
    perf_gflops = GFLOPS_PER_SPMV / duration_s

    # Effective memory bandwidth from throughput percentage
    effective_bw = PEAK_MEMORY_BW * (kernel_data['memory_throughput_pct'] / 100.0)

    # Arithmetic intensity = GFLOP/s / GB/s = FLOP/byte
    ai = perf_gflops / effective_bw

    return ai, perf_gflops

def main():
    fig, ax = plt.subplots(figsize=(10, 7))

    # Roofline boundaries
    ai_range = np.logspace(-2, 2, 500)

    # Memory roof (slope = memory bandwidth)
    memory_roof = PEAK_MEMORY_BW * ai_range

    # Compute roof (flat line at peak FP64)
    compute_roof = np.full_like(ai_range, PEAK_FP64)

    # Combined roofline (minimum of memory and compute)
    roofline = np.minimum(memory_roof, compute_roof)

    # Ridge point (where memory and compute roofs meet)
    ridge_ai = PEAK_FP64 / PEAK_MEMORY_BW

    # Plot roofline
    ax.loglog(ai_range, roofline, 'b-', linewidth=2.5, label='Roofline', zorder=1)

    # Fill regions
    ax.fill_between(ai_range, roofline, 0.1, alpha=0.1, color='blue')

    # Add roof labels (positioned to avoid overlap with data)
    ax.annotate(f'DRAM BW: {PEAK_MEMORY_BW} GB/s (spec)',
                xy=(0.015, 2.5), fontsize=9, color='darkblue', rotation=45)
    ax.annotate(f'Peak FP64: {PEAK_FP64} GFLOP/s',
                xy=(1.5, PEAK_FP64 * 1.15), fontsize=9, color='darkblue')

    # Plot kernel points
    for name, data in kernels.items():
        ai, perf = compute_metrics(data)
        ax.scatter(ai, perf, s=200, c=data['color'], marker=data['marker'],
                   edgecolors='black', linewidths=1.5, zorder=5, label=name)

        # Add annotation positioned to the right of the roofline slope
        if name == 'Custom Stencil':
            offset = (50, 10)
        else:
            offset = (60, -15)
        ax.annotate(f'{name}\n{perf:.1f} GFLOP/s',
                    xy=(ai, perf), xytext=offset, textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=data['color'], alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

    # Formatting
    ax.set_xlim(0.01, 10)
    ax.set_ylim(1, 200)
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12, fontweight='bold')
    ax.set_title('Roofline Analysis: SpMV Kernels on RTX 4060 Laptop GPU\n(7000×7000 5-point stencil matrix)',
                 fontsize=14, fontweight='bold')

    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=11)

    # Add insight text box
    textstr = '\n'.join([
        'Both kernels are memory-bound',
        'Stencil achieves 2× higher throughput',
        'by eliminating index indirection'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Add data source note
    ax.text(0.98, 0.02, 'Data: Nsight Compute',
            transform=ax.transAxes, fontsize=8, color='gray',
            horizontalalignment='right', verticalalignment='bottom')

    plt.tight_layout()

    # Save figure
    output_path = 'docs/figures/roofline_spmv_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')

    # Also save to profiling/images for reference
    output_path2 = 'profiling/images/roofline_spmv_comparison.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path2}')

if __name__ == '__main__':
    main()
