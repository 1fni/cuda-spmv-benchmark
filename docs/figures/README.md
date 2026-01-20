# Showcase Figures

Curated figures for the README and documentation. These are the **final, polished** visualizations.

## Contents

- `performance_summary_horizontal.png` - Main performance overview
- `scaling_main_a100.png` - Multi-GPU scaling chart
- `scaling_detailed_a100.png` - Detailed scaling breakdown
- `spmv_format_comparison_a100.png` - SpMV kernel comparison
- `custom_vs_amgx_overview.png` - AmgX comparison
- `problem_size_scaling_overview.png` - Problem size analysis

## vs `results/figures/`

| Directory | Purpose | Git tracked |
|-----------|---------|-------------|
| `docs/figures/` | Curated showcase figures | ✅ Yes |
| `results/figures/` | Auto-generated from benchmarks | ❌ No |

## Workflow

1. Run benchmarks: `./scripts/run_all.sh`
2. Generate plots: `python scripts/plotting/plot_results.py`
3. Review outputs in `results/figures/`
4. Copy selected figures here for showcase
5. Commit to git
