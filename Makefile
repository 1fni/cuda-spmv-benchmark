# Directories
SRC_DIR := src
INC_DIR := include
OBJ_DIR := build/$(BUILD_TYPE)
BIN_DIR := bin/$(BUILD_TYPE)
MAT_DIR := matrix
RES_DIR := results

# Build type (default: release)
BUILD_TYPE ?= release

# Compiler + flags
NVCC := nvcc
MPICXX := mpic++
MPI_INCLUDES := -I/usr/lib/x86_64-linux-gnu/openmpi/include

ifeq ($(BUILD_TYPE),debug)
    NVCCFLAGS := -g -G -O0 -std=c++11
else
    NVCCFLAGS := -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true -std=c++11
endif

# Base includes and libraries
INCLUDES := -I$(INC_DIR) -I$(INC_DIR)/solvers
LDFLAGS := -lcusparse -lcublas
CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -lcudart

# Sources / objets
CU_SRCS := $(shell find $(SRC_DIR) -name '*.cu')
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

# SpMV benchmark: exclude generator, CG solver, and multi-GPU sources
CU_SPMV_SRCS := $(filter-out $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/main/cg_test.cu $(SRC_DIR)/main/test_mgpu_cg.cu $(SRC_DIR)/main/test_mgpu_cg_partitioned.cu $(SRC_DIR)/solvers/cg_solver_mgpu.cu $(SRC_DIR)/solvers/cg_solver_mgpu_partitioned.cu, $(CU_SRCS))
CU_SPMV_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SPMV_SRCS))

# Matrix generator
CU_GEN_SRCS := $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/io/io.cu
CU_GEN_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_GEN_SRCS))

# Binaries
BIN_SPMV := $(BIN_DIR)/spmv_bench
BIN_GEN  := $(BIN_DIR)/generate_matrix
BIN_CG   := $(BIN_DIR)/cg_test

# CG solver test: exclude generator, spmv_bench, and multi-GPU sources
CU_CG_SRCS := $(filter-out $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/main/main.cu $(SRC_DIR)/main/test_mgpu_cg.cu $(SRC_DIR)/main/test_mgpu_cg_partitioned.cu $(SRC_DIR)/solvers/cg_solver_mgpu.cu $(SRC_DIR)/solvers/cg_solver_mgpu_partitioned.cu, $(CU_SRCS))
CU_CG_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_CG_SRCS))

# PHONY targets
.PHONY: all clean test_mgpu_part

# Main target
all: $(BIN_SPMV) $(BIN_GEN) $(BIN_CG)

# SpMV benchmark binary
$(BIN_SPMV): $(CU_SPMV_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Matrix generator binary
$(BIN_GEN): $(CU_GEN_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

# CG solver binary
$(BIN_CG): $(CU_CG_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Compile CUDA sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Multi-GPU CG Solver with MPI
# ============================================================================

# Multi-GPU partitioned solver binary
BIN_MGPU_PART := $(BIN_DIR)/test_mgpu_cg_partitioned

# MPI objects (shared utilities)
OBJ_MGPU_IO := $(OBJ_DIR)/mgpu/io.o
OBJ_MGPU_CSR := $(OBJ_DIR)/mgpu/spmv_csr.o
OBJ_MGPU_STENCIL := $(OBJ_DIR)/mgpu/spmv_stencil_csr_direct.o

# Partitioned solver objects
OBJ_MGPU_PART_MAIN := $(OBJ_DIR)/mgpu/test_mgpu_cg_partitioned.o
OBJ_MGPU_PART_SOLVER := $(OBJ_DIR)/mgpu/cg_solver_mgpu_partitioned.o

# Compile MPI sources with NVCC + MPI headers
$(OBJ_DIR)/mgpu/%.o: $(SRC_DIR)/main/%.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/%.o: $(SRC_DIR)/solvers/%.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/io.o: $(SRC_DIR)/io/io.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_csr.o: $(SRC_DIR)/spmv/spmv_cusparse_csr.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_stencil_csr_direct.o: $(SRC_DIR)/spmv/spmv_stencil_csr_direct.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

# Link partitioned solver with MPI
$(BIN_MGPU_PART): $(OBJ_MGPU_PART_MAIN) $(OBJ_MGPU_PART_SOLVER) $(OBJ_MGPU_IO) $(OBJ_MGPU_CSR) $(OBJ_MGPU_STENCIL)
	@mkdir -p $(BIN_DIR)
	$(MPICXX) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

# Convenience target
test_mgpu_part: $(BIN_MGPU_PART)

# Clean
clean:
	rm -rf build bin results

