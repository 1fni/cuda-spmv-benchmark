# VastAI - Commandes CG + Kokkos Benchmark

## 1. Setup Initial (à exécuter une fois)

```bash
# Sur le nœud VastAI
cd /workspace
git clone https://github.com/1fni/cuda-spmv-benchmark.git
cd cuda-spmv-benchmark

# Lancer l'installation complète (Kokkos + build)
bash scripts/vastai_cg_setup.sh
```

Ce script fait :
- Clone le repo
- Build le projet
- Détecte le GPU et calcule la taille max de matrice
- Installe Kokkos + Kokkos-Kernels
- Build les exécutables CG (le nôtre + Kokkos)

---

## 2. Génération de la Matrice (remplir le nœud)

```bash
# Charger la config GPU détectée
source /tmp/gpu_config.env

# Vérifier la taille max
echo "Max matrix size: ${MAX_MATRIX_SIZE}x${MAX_MATRIX_SIZE}"

# Générer la matrice principale (remplira le disque)
./bin/generate_matrix ${MAX_MATRIX_SIZE} matrix/stencil_max.mtx
```

**Note** : Ne pas générer la matrice Kokkos tout de suite (le nœud est déjà plein).

---

## 3. Tests CG - Notre Solveur

### Mode 1: CSR-Direct (stencil optimisé)

```bash
# Run fonctionnel
./bin/cg_test matrix/stencil_max.mtx --mode=stencil5-csr-direct

# Profiling Nsight Compute
ncu --set full \
    --export profiling_results/cg_stencil5_csr_direct \
    --force-overwrite \
    --kernel-regex="spmv|SpMV" \
    ./bin/cg_test matrix/stencil_max.mtx --mode=stencil5-csr-direct
```

### Mode 2: CSR (baseline cuSPARSE)

```bash
# Run fonctionnel
./bin/cg_test matrix/stencil_max.mtx --mode=csr

# Profiling Nsight Compute
ncu --set full \
    --export profiling_results/cg_csr \
    --force-overwrite \
    --kernel-regex="spmv|SpMV" \
    ./bin/cg_test matrix/stencil_max.mtx --mode=csr
```

---

## 4. Test Kokkos CG (MANUEL - après libération d'espace)

### Étape 1: Libérer de l'espace

```bash
# Option A: Supprimer temporairement notre matrice
mv matrix/stencil_max.mtx /tmp/stencil_max.mtx.bak

# Option B: Compresser
gzip matrix/stencil_max.mtx  # Crée stencil_max.mtx.gz
```

### Étape 2: Générer matrice Kokkos

```bash
source /tmp/gpu_config.env
./external/generate_kokkos_matrix ${MAX_MATRIX_SIZE} matrix/stencil_max_kokkos.mtx
```

### Étape 3: Run Kokkos CG

```bash
# Run fonctionnel
./external/kokkos_cg_struct matrix/stencil_max_kokkos.mtx

# Profiling Nsight Compute
ncu --set full \
    --export profiling_results/kokkos_cg_struct \
    --force-overwrite \
    --kernel-regex="spmv|Kokkos" \
    ./external/kokkos_cg_struct matrix/stencil_max_kokkos.mtx
```

### Étape 4: Restaurer notre matrice (si backup)

```bash
mv /tmp/stencil_max.mtx.bak matrix/stencil_max.mtx
# ou
gunzip matrix/stencil_max.mtx.gz
```

---

## 5. Récupération des Résultats

### Fichiers générés

```bash
profiling_results/
├── cg_stencil5_csr_direct.ncu-rep   # Nsight profile stencil-direct
├── cg_csr.ncu-rep                    # Nsight profile CSR baseline
└── kokkos_cg_struct.ncu-rep          # Nsight profile Kokkos
```

### Copier depuis VastAI vers local

```bash
# Sur votre machine locale
scp -r root@<vastai-ip>:/workspace/cuda-spmv-benchmark/profiling_results ./vastai_cg_results/
```

### Visualiser les profils

```bash
# Sur machine locale (avec Nsight Compute installé)
ncu-ui vastai_cg_results/cg_stencil5_csr_direct.ncu-rep
ncu-ui vastai_cg_results/cg_csr.ncu-rep
ncu-ui vastai_cg_results/kokkos_cg_struct.ncu-rep
```

---

## 6. Métriques Clés à Extraire

### Depuis les outputs texte

```bash
# Convergence et temps total
grep -E "Converged:|Iterations:|Time-to-solution:" <output>

# Breakdown temps
grep -E "SpMV:|BLAS1:|Reductions:" <output>

# Performances SpMV
grep -E "FLOPS:|Bandwidth:" <output>
```

### Depuis Nsight Compute

Métriques importantes :
- **SM Throughput** : Utilisation GPU compute
- **DRAM Throughput** : Bande passante mémoire
- **Achieved Occupancy** : Efficacité threads
- **Memory Efficiency** : Taux de hit cache L1/L2
- **Duration** : Temps d'exécution kernel

---

## 7. Script Automatique Partiel (optionnel)

Si tu veux automatiser juste les runs (pas la génération matrice) :

```bash
#!/bin/bash
# vastai_run_all_cg.sh
set -e

MATRIX="matrix/stencil_max.mtx"
OUTDIR="profiling_results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# Notre CG - stencil-direct
echo "Running CG stencil-direct..."
./bin/cg_test "$MATRIX" --mode=stencil5-csr-direct | tee "$OUTDIR/cg_stencil_direct.txt"
ncu --set full --export "$OUTDIR/cg_stencil_direct" --kernel-regex="spmv" \
    ./bin/cg_test "$MATRIX" --mode=stencil5-csr-direct > "$OUTDIR/cg_stencil_direct_ncu.log" 2>&1

# Notre CG - CSR
echo "Running CG CSR..."
./bin/cg_test "$MATRIX" --mode=csr | tee "$OUTDIR/cg_csr.txt"
ncu --set full --export "$OUTDIR/cg_csr" --kernel-regex="spmv" \
    ./bin/cg_test "$MATRIX" --mode=csr > "$OUTDIR/cg_csr_ncu.log" 2>&1

echo "Results in: $OUTDIR"
```

Puis manuellement faire Kokkos après avoir géré l'espace disque.

---

## Résumé Workflow

```
1. Setup (1 fois)     → bash scripts/vastai_cg_setup.sh
2. Générer matrice    → ./bin/generate_matrix ${MAX_MATRIX_SIZE} matrix/stencil_max.mtx
3. Test CG direct     → ./bin/cg_test + ncu
4. Test CG CSR        → ./bin/cg_test + ncu
5. Libérer espace     → mv/gzip matrice
6. Générer Kokkos     → ./external/generate_kokkos_matrix
7. Test Kokkos        → ./external/kokkos_cg_struct + ncu
8. Récupérer résultats → scp profiling_results/
```
