# Benchmark Scripts Guide

Cinq scripts principaux pour évaluer les performances du solver CG multi-GPU et SpMV standalone.

---

## 1. Single-GPU Format Comparison

**Script**: `benchmark_single_gpu_formats.sh`

**Objectif**: Comparer CSR (cuSPARSE) vs STENCIL5-OPT (custom kernel) sur single-GPU

**Test**:
- Formats: CSR, STENCIL5-OPT
- Tailles: 5k, 7.5k, 10k, 15k, 20k (25M à 400M unknowns)
- Hardware: 1 GPU

**Usage**:
```bash
cd /path/to/cuda-spmv-benchmark
./scripts/benchmarking/benchmark_single_gpu_formats.sh

# Résultats dans: results_single_gpu_formats_[GPU]_[DATE]/
cd results_single_gpu_formats_*/
python3 analyze_formats.py
```

**Output**:
- `format_comparison.png` : 4 panels (time, GFLOPS, bandwidth, speedup)
- JSON per configuration
- Summary table avec speedup STENCIL5-OPT vs CSR

**Résultats attendus (H100 NVL)** :
- CSR: ~320 GFLOPS, 2575 GB/s
- STENCIL5-OPT: ~480 GFLOPS, 3645 GB/s
- Speedup: **1.49×**

---

## 2. Strong Scaling (Multi-GPU)

**Script**: `benchmark_problem_sizes.sh`

**Objectif**: Tester strong scaling (même problème, plus de GPUs = plus rapide)

**Test**:
- GPU counts: 1, 2, 4, 8
- Tailles: 10k, 15k, 20k (fixe pour chaque GPU count)
- Métrique: Speedup et parallel efficiency

**Usage**:
```bash
./scripts/benchmarking/benchmark_problem_sizes.sh

# Résultats dans: results_problem_size_scaling_[GPU]_[DATE]/
cd results_problem_size_scaling_*/
python3 analyze_scaling.py
```

**Output**:
- `strong_scaling_analysis.png` : 4 panels (time, speedup, efficiency, time/iter)
- JSON/CSV per configuration
- Summary table avec speedup et efficiency

**Résultats attendus (8× A100)** :
- 10k×10k: 6.94× speedup, 86.8% efficiency
- 15k×15k: 7.43× speedup, 92.9% efficiency
- 20k×20k: **7.48× speedup, 93.5% efficiency**

---

## 3. Weak Scaling (Multi-GPU)

**Script**: `benchmark_weak_scaling.sh`

**Objectif**: Tester weak scaling (constant work per GPU, temps constant idéal)

**Test**:
- 1 GPU: 5k×5k (25M unknowns)
- 2 GPUs: 7071×7071 (~50M unknowns)
- 4 GPUs: 10k×10k (100M unknowns)
- 8 GPUs: 14142×14142 (~200M unknowns)
- Métrique: Efficiency (temps constant = 100%)

**Usage**:
```bash
./scripts/benchmarking/benchmark_weak_scaling.sh

# Résultats dans: results_weak_scaling_[GPU]_[DATE]/
cd results_weak_scaling_*/
python3 analyze_weak_scaling.py
```

**Output**:
- `weak_scaling_analysis.png` : 2 panels (time vs GPUs, efficiency bars)
- JSON/CSV per configuration
- Summary table avec efficiency

**Résultats attendus (8× A100)** :
- Temps devrait rester ~constant (baseline 1 GPU)
- Efficiency > 80% pour weak scaling correct
- Montre si l'overhead communication domine

---

## 4. Multi-GPU SpMV Strong Scaling

**Script**: `benchmark_mgpu_spmv_strong.sh`

**Objectif**: Tester strong scaling du SpMV seul (pas CG complet)

**Test**:
- GPU counts: 1, 2, 4, 8
- Tailles: 10k, 15k, 20k (fixe pour chaque GPU count)
- Métrique: Temps SpMV par iteration (extrait du CG solver avec --timers)

**Usage**:
```bash
./scripts/benchmarking/benchmark_mgpu_spmv_strong.sh

# Résultats dans: results_mgpu_spmv_strong_[GPU]_[DATE]/
cd results_mgpu_spmv_strong_*/
python3 analyze_spmv_scaling.py
```

**Output**:
- `mgpu_spmv_strong_scaling.png` : 4 panels (time/iter, speedup, efficiency, total)
- JSON/CSV per configuration
- Summary table avec speedup SpMV seul

**Différence avec CG complet**:
- CG inclut: SpMV + BLAS1 + réductions + communications
- SpMV seul: Juste le kernel SpMV (25-30% du temps total CG)
- Permet d'isoler performance SpMV pure

---

## 5. Multi-GPU SpMV Weak Scaling

**Script**: `benchmark_mgpu_spmv_weak.sh`

**Objectif**: Tester weak scaling du SpMV seul (constant work per GPU)

**Test**:
- 1 GPU: 5k×5k (25M unknowns)
- 2 GPUs: 7071×7071 (~50M unknowns)
- 4 GPUs: 10k×10k (100M unknowns)
- 8 GPUs: 14142×14142 (~200M unknowns)
- Métrique: Temps SpMV par iteration constant = 100% efficiency

**Usage**:
```bash
./scripts/benchmarking/benchmark_mgpu_spmv_weak.sh

# Résultats dans: results_mgpu_spmv_weak_[GPU]_[DATE]/
cd results_mgpu_spmv_weak_*/
python3 analyze_spmv_weak.py
```

**Output**:
- `mgpu_spmv_weak_scaling.png` : 2 panels (time vs GPUs, efficiency bars)
- JSON/CSV per configuration
- Summary table avec efficiency SpMV

**Note importante**:
- SpMV temps extrait du CG solver via `--timers` flag
- Temps per iteration rapporté (moyenne sur toutes les itérations CG)
- Permet de mesurer overhead communication SpMV seul

---

## Comparaison Strong vs Weak Scaling

| Type | Taille problème | Métrique | Idéal |
|------|----------------|----------|-------|
| **Strong** | Fixe (ex: 15k×15k) | Speedup = T₁/Tₙ | Speedup = N GPUs |
| **Weak** | Proportionnelle (25M/GPU) | Efficiency = T₁/Tₙ × 100% | Efficiency = 100% |

**Strong scaling** : "Plus de GPUs = plus rapide"
- Test : Problème fixe sur 1, 2, 4, 8 GPUs
- Idéal : 8 GPUs = 8× plus rapide
- Reality : 7-7.5× typique (communication overhead)

**Weak scaling** : "Plus de GPUs = problèmes plus gros en même temps"
- Test : Work constant per GPU (25M unknowns/GPU)
- Idéal : Temps constant quel que soit GPU count
- Reality : Légère augmentation (overhead MPI AllReduce)

---

## Workflow Showcase Complet

```bash
# ===== Single-GPU Benchmarks =====
# 1. Format comparison CSR vs STENCIL5-OPT (pour README hero section)
./scripts/benchmarking/benchmark_single_gpu_formats.sh
cd results_single_gpu_formats_*/
python3 analyze_formats.py
cp format_comparison.png ../docs/figures/
cd ..

# ===== Multi-GPU CG Solver Benchmarks =====
# 2. Strong scaling CG complet (showcase principal)
./scripts/benchmarking/benchmark_problem_sizes.sh
cd results_problem_size_scaling_*/
python3 analyze_scaling.py
cp strong_scaling_analysis.png ../docs/figures/
cd ..

# 3. Weak scaling CG complet (optionnel)
./scripts/benchmarking/benchmark_weak_scaling.sh
cd results_weak_scaling_*/
python3 analyze_weak_scaling.py
cp weak_scaling_analysis.png ../docs/figures/
cd ..

# ===== Multi-GPU SpMV Standalone Benchmarks =====
# 4. Strong scaling SpMV seul (pour isoler performance SpMV)
./scripts/benchmarking/benchmark_mgpu_spmv_strong.sh
cd results_mgpu_spmv_strong_*/
python3 analyze_spmv_scaling.py
cp mgpu_spmv_strong_scaling.png ../docs/figures/
cd ..

# 5. Weak scaling SpMV seul (optionnel)
./scripts/benchmarking/benchmark_mgpu_spmv_weak.sh
cd results_mgpu_spmv_weak_*/
python3 analyze_spmv_weak.py
cp mgpu_spmv_weak_scaling.png ../docs/figures/
cd ..
```

---

## Configuration

Tous les scripts utilisent les mêmes conventions :
- **RUNS=10** : Nombre de runs per config (median reporté)
- **BRANCH="main"** : Branche git à tester
- **Auto-detection** : GPU name, date pour nommage résultats
- **Matrix generation** : Automatique si fichier manquant

---

## Troubleshooting

**Build fails** :
```bash
# Vérifier cibles Makefile
make cg_solver_mgpu_stencil  # Multi-GPU
make spmv_bench              # Single-GPU
make generate_matrix         # Génération matrices
```

**Out of memory** :
- Réduire tailles matrices (MATRIX_SIZES)
- Utiliser moins de GPUs
- Vérifier `nvidia-smi` pour mémoire disponible

**MPI errors** :
```bash
# Vérifier GPU count disponible
nvidia-smi --list-gpus

# Test MPI simple
mpirun -np 2 hostname
```

**Python errors** :
```bash
# Install matplotlib
pip3 install matplotlib numpy

# Test JSON parsing
jq . results_*.json
```

---

## Output Files

Chaque script génère :
- `results_[type]_[GPU]_[DATE]/` : Dossier résultats
- `*.json` : Métriques détaillées per config
- `*.csv` : Export CSV pour spreadsheet
- `summary.txt` : Log complet de tous les runs
- `analyze_*.py` : Script Python pour génération plots
- `*_analysis.png` : Visualisation finale (300 DPI)

---

*Created: 2026-01-09*
*Author: Stephane Bouhrour*
