AMGX_DIR ?= $(CURDIR)/../../amgx

CXX = nvcc
CXXFLAGS = -std=c++17 -O3 --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true -I$(AMGX_DIR)/include
LDFLAGS = -L$(AMGX_DIR)/lib -lamgx -lcudart -lcusparse -lcublas -lcusolver -lcuda

# MPI flags for multi-GPU
MPICXX = mpic++
MPI_CXXFLAGS = $(CXXFLAGS) -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LDFLAGS = $(LDFLAGS) -lmpi -lmpi_cxx

all: amgx_spmv_bench amgx_cg_solver amgx_cg_solver_mgpu

amgx_spmv_bench: amgx_spmv_stencil.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

amgx_cg_solver: amgx_cg_solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

amgx_cg_solver_mgpu: amgx_cg_solver_mgpu.cpp
	$(CXX) $(MPI_CXXFLAGS) $< -o $@ $(MPI_LDFLAGS)

clean:
	rm -f amgx_spmv_bench amgx_cg_solver amgx_cg_solver_mgpu

.PHONY: all clean
