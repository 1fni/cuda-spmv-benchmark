# Kokkos Comparison - Simple Plan

## Setup
```bash
git clone https://github.com/kokkos/kokkos.git external/kokkos
git clone https://github.com/kokkos/kokkos-kernels.git external/kokkos-kernels

cd external/kokkos
mkdir build && cd build
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON ..
make -j8
```

## Benchmark executable
```cpp
// external/kokkos_spmv_bench.cpp
// Generate 5-point stencil, run SpMV, output time
```

## Compare script
```bash
# Run both, same matrix size
./bin/spmv_bench matrix.mtx --mode=stencil5-opt
./external/kokkos_bench --grid-size=N

# Compare times
```

## Deliverable
Simple table: grid size, our time, kokkos time, speedup
