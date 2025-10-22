AMGX_DIR = $(CURDIR)/../../amgx

CXX = nvcc
CXXFLAGS = -std=c++17 -O3 --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true -I$(AMGX_DIR)/include
LDFLAGS = -L$(AMGX_DIR)/lib -lamgx -lcudart -lcusparse -lcublas -lcusolver -lcuda

all: amgx_spmv_bench amgx_cg_solver

amgx_spmv_bench: amgx_spmv_stencil.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

amgx_cg_solver: amgx_cg_solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f amgx_spmv_bench amgx_cg_solver

.PHONY: all clean
