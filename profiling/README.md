# Profiling Data

Nsight Systems profiles for performance analysis and showcase.

## Contents

### `nsys/` - Nsight Systems Timelines

| Profile | Description |
|---------|-------------|
| `mpi_1ranks_profile_10000.nsys-rep` | Custom CG, 1 GPU, 10k×10k matrix |
| `mpi_2ranks_profile_10000.nsys-rep` | Custom CG, 2 GPUs, 10k×10k matrix |
| `amgx_1ranks_profile_10000.nsys-rep` | AmgX CG, 1 GPU, 10k×10k matrix |
| `amgx_2ranks_profile_10000.nsys-rep` | AmgX CG, 2 GPUs, 10k×10k matrix |

## Viewing Profiles

```bash
# Open in Nsight Systems GUI
nsys-ui profiling/nsys/mpi_2ranks_profile_10000.nsys-rep
```

## Generating New Profiles

```bash
# Profile custom CG (2 GPUs)
nsys profile --trace=cuda,mpi,nvtx -o profiling/nsys/custom_2gpu \
    mpirun -np 2 ./bin/cg_solver_mgpu_stencil matrix/stencil_10000x10000.mtx

# Profile AmgX (2 GPUs)
nsys profile --trace=cuda,mpi,nvtx -o profiling/nsys/amgx_2gpu \
    mpirun -np 2 ./external/benchmarks/amgx/amgx_cg_solver_mgpu matrix/stencil_10000x10000.mtx
```

## Key Observations

- Custom CG shows better compute/communication overlap
- AmgX has more kernel launches (generic CSR vs optimized stencil)
- MPI staging (D2H → MPI → H2D) visible in custom implementation
