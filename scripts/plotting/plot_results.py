#!/usr/bin/env python3
"""
plot_results.py - Generate benchmark figures from JSON results

Usage:
    python scripts/plotting/plot_results.py
    python scripts/plotting/plot_results.py --results-dir=results/json --output-dir=results/figures

Generates:
    - cg_speedup.png: Custom CG vs AmgX comparison
    - cg_scaling.png: Multi-GPU scaling efficiency
    - spmv_comparison.png: SpMV kernel comparison
"""

import json
import os
import sys
import glob
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy required")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'custom': '#2ecc71',      # Green
    'amgx': '#e74c3c',        # Red
    'cusparse': '#3498db',    # Blue
    'stencil': '#9b59b6',     # Purple
}


def load_json_results(results_dir):
    """Load all JSON result files from directory."""
    results = {}
    json_files = glob.glob(os.path.join(results_dir, '*.json'))

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                name = Path(filepath).stem
                results[name] = data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {filepath}: {e}")

    return results


def plot_cg_speedup(results, output_dir):
    """Generate CG speedup comparison chart (Custom vs AmgX)."""
    # Find matching pairs (custom vs amgx for same config)
    custom_results = {k: v for k, v in results.items() if 'amgx' not in k.lower()}
    amgx_results = {k: v for k, v in results.items() if 'amgx' in k.lower()}

    if not custom_results or not amgx_results:
        print("Skipping CG speedup plot: need both custom and AmgX results")
        return

    # Group by configuration (single vs multi-gpu)
    configs = []
    custom_times = []
    amgx_times = []
    speedups = []

    for config_type in ['single', 'mgpu']:
        custom_key = next((k for k in custom_results if config_type in k), None)
        amgx_key = next((k for k in amgx_results if config_type in k), None)

        if custom_key and amgx_key:
            custom_time = custom_results[custom_key]['timing']['median_ms']
            amgx_time = amgx_results[amgx_key]['timing']['median_ms']
            speedup = amgx_time / custom_time

            label = 'Single-GPU' if config_type == 'single' else 'Multi-GPU'
            configs.append(label)
            custom_times.append(custom_time)
            amgx_times.append(amgx_time)
            speedups.append(speedup)

    if not configs:
        print("Skipping CG speedup plot: no matching configurations found")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, amgx_times, width, label='AmgX (NVIDIA)', color=COLORS['amgx'])
    bars2 = ax.bar(x + width/2, custom_times, width, label='Custom CG', color=COLORS['custom'])

    # Add speedup annotations
    for i, (bar1, bar2, speedup) in enumerate(zip(bars1, bars2, speedups)):
        ax.annotate(f'{speedup:.2f}x faster',
                    xy=(x[i], max(bar1.get_height(), bar2.get_height())),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color=COLORS['custom'])

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('CG Solver Performance: Custom vs AmgX', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(amgx_times) * 1.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cg_speedup.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_scaling(results, output_dir):
    """Generate multi-GPU scaling efficiency chart."""
    # Find multi-GPU results with different GPU counts
    mgpu_results = [(k, v) for k, v in results.items()
                    if 'mgpu' in k.lower() and 'amgx' not in k.lower()]

    if len(mgpu_results) < 2:
        print("Skipping scaling plot: need results with multiple GPU counts")
        return

    # Sort by GPU count (extract from filename or data)
    # Assuming format like "cg_mgpu_5000_4gpu"
    def get_gpu_count(name):
        for part in name.split('_'):
            if 'gpu' in part:
                try:
                    return int(part.replace('gpu', ''))
                except ValueError:
                    pass
        return 1

    mgpu_results.sort(key=lambda x: get_gpu_count(x[0]))

    gpu_counts = [get_gpu_count(name) for name, _ in mgpu_results]
    times = [data['timing']['median_ms'] for _, data in mgpu_results]

    # Calculate speedup relative to smallest GPU count
    base_time = times[0] * gpu_counts[0]  # Estimate single-GPU time
    speedups = [base_time / t for t in times]
    ideal_speedup = gpu_counts
    efficiency = [s / i * 100 for s, i in zip(speedups, ideal_speedup)]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Speedup plot
    ax1.plot(gpu_counts, speedups, 'o-', color=COLORS['custom'],
             linewidth=2, markersize=8, label='Measured')
    ax1.plot(gpu_counts, ideal_speedup, '--', color='gray',
             linewidth=2, label='Ideal (linear)')
    ax1.set_xlabel('Number of GPUs', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Multi-GPU Scaling', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_xticks(gpu_counts)
    ax1.grid(True, alpha=0.3)

    # Efficiency plot
    bars = ax2.bar(gpu_counts, efficiency, color=COLORS['custom'], alpha=0.8)
    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Number of GPUs', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title('Scaling Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(gpu_counts)
    ax2.set_ylim(0, 110)

    # Add efficiency labels on bars
    for bar, eff in zip(bars, efficiency):
        ax2.annotate(f'{eff:.1f}%',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cg_scaling.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_spmv_comparison(results, output_dir):
    """Generate SpMV kernel comparison chart."""
    # Look for SpMV results (might be in different format)
    spmv_results = {k: v for k, v in results.items() if 'spmv' in k.lower()}

    if not spmv_results:
        print("Skipping SpMV comparison plot: no SpMV results found")
        return

    # If we have per-mode data in results
    # This depends on how SpMV benchmark exports JSON
    print("SpMV comparison plot: TODO (depends on spmv_bench JSON format)")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark figures')
    parser.add_argument('--results-dir', default='results/json',
                        help='Directory containing JSON results')
    parser.add_argument('--output-dir', default='results/figures',
                        help='Directory for output figures')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_json_results(args.results_dir)

    if not results:
        print("No JSON results found. Run benchmarks first:")
        print("  ./scripts/run_all.sh")
        return 1

    print(f"Found {len(results)} result files")
    for name in results:
        print(f"  - {name}")
    print()

    # Generate plots
    plot_cg_speedup(results, args.output_dir)
    plot_scaling(results, args.output_dir)
    plot_spmv_comparison(results, args.output_dir)

    print(f"\nFigures saved to: {args.output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
