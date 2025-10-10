# Industry Reference Implementations

Standalone benchmarks for validating custom SpMV kernels against production-grade libraries.

## Structure

All reference implementations are:
- **Compiled separately** from main project
- **Use identical optimization flags** (`-O3 --ptxas-options=-O3 --allow-expensive-optimizations`)
- **Standalone binaries** for fair comparison

## AmgX (NVIDIA Production Solver)

NVIDIA's production-grade linear algebra library.

```bash
# Build AmgX benchmark
cd external
make -f Makefile_amgx

# Run comparison
./amgx_bench ../matrix/stencil_512x512.mtx
```

**Requirements**: AmgX library installed at `external/amgx/`

## Kokkos (Performance Portability Framework)

Structured stencil SpMV with Kokkos portable kernels.

```bash
# Generate Kokkos-ordered matrix
./generate_kokkos_matrix 512 ../matrix/stencil_512x512_kokkos.mtx

# Build and run benchmark
make -f Makefile_kokkos kokkos_struct
./kokkos_struct ../matrix/stencil_512x512_kokkos.mtx
```

**Requirements**: Kokkos and KokkosSparse installed at `external/kokkos-install/`

## Compilation Flags

**All implementations use**:
```makefile
CXXFLAGS = -std=c++17 -O3 --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true
```

This ensures fair comparison with custom kernels in `src/spmv/`.
