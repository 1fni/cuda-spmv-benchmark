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

ifeq ($(BUILD_TYPE),debug)
    NVCCFLAGS := -g -G -O0 -std=c++11
else
    NVCCFLAGS := -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true -std=c++11
endif

# Base includes and libraries
INCLUDES := -I$(INC_DIR) -I$(INC_DIR)/solvers
LDFLAGS := -lcusparse -lcublas

# Sources / objets
CU_SRCS := $(shell find $(SRC_DIR) -name '*.cu')
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

# SpMV : exclut generate_matrix, cg_test, test_mgpu_cg, cg_solver_mgpu
CU_SPMV_SRCS := $(filter-out $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/main/cg_test.cu $(SRC_DIR)/main/test_mgpu_cg.cu $(SRC_DIR)/solvers/cg_solver_mgpu.cu, $(CU_SRCS))
CU_SPMV_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SPMV_SRCS))

# Générateur : juste generate_matrix + io
CU_GEN_SRCS := $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/io/io.cu
CU_GEN_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_GEN_SRCS))

# Binaries
BIN_SPMV := $(BIN_DIR)/spmv_bench
BIN_GEN  := $(BIN_DIR)/generate_matrix
BIN_CG   := $(BIN_DIR)/cg_test

# CG test: exclut generate_matrix, spmv_bench main, test_mgpu_cg, cg_solver_mgpu
CU_CG_SRCS := $(filter-out $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/main/main.cu $(SRC_DIR)/main/test_mgpu_cg.cu $(SRC_DIR)/solvers/cg_solver_mgpu.cu, $(CU_SRCS))
CU_CG_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_CG_SRCS))

# PHONY
.PHONY: all clean run test_mgpu

# Main target
all: $(BIN_SPMV) $(BIN_GEN) $(BIN_CG)

# Link SpMV
$(BIN_SPMV): $(CU_SPMV_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Link générateur
$(BIN_GEN): $(CU_GEN_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

# Link CG test
$(BIN_CG): $(CU_CG_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Compile objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Run bench (example)
run: $(BIN_SPMV)
	@mkdir -p $(MAT_DIR) $(RES_DIR)
	$(BIN_SPMV) $(MAT_DIR)/example.mtx --mode=csr | tee $(RES_DIR)/run_$(BUILD_TYPE).log

# Multi-GPU test with MPI+NCCL
BIN_MGPU := $(BIN_DIR)/test_mgpu_cg

# MPI objects
OBJ_MGPU_MAIN := $(OBJ_DIR)/mgpu/test_mgpu_cg.o
OBJ_MGPU_SOLVER := $(OBJ_DIR)/mgpu/cg_solver_mgpu.o
OBJ_MGPU_IO := $(OBJ_DIR)/mgpu/io.o
OBJ_MGPU_CSR := $(OBJ_DIR)/mgpu/spmv_csr.o
OBJ_MGPU_STENCIL := $(OBJ_DIR)/mgpu/spmv_stencil_csr_direct.o

# Compile MPI sources with nvcc + MPI headers
$(OBJ_DIR)/mgpu/test_mgpu_cg.o: $(SRC_DIR)/main/test_mgpu_cg.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I/usr/lib/x86_64-linux-gnu/openmpi/include -c $< -o $@

$(OBJ_DIR)/mgpu/cg_solver_mgpu.o: $(SRC_DIR)/solvers/cg_solver_mgpu.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I/usr/lib/x86_64-linux-gnu/openmpi/include -c $< -o $@

$(OBJ_DIR)/mgpu/io.o: $(SRC_DIR)/io/io.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I/usr/lib/x86_64-linux-gnu/openmpi/include -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_csr.o: $(SRC_DIR)/spmv/spmv_cusparse_csr.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I/usr/lib/x86_64-linux-gnu/openmpi/include -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_stencil_csr_direct.o: $(SRC_DIR)/spmv/spmv_stencil_csr_direct.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I/usr/lib/x86_64-linux-gnu/openmpi/include -c $< -o $@

# Link with mpic++
$(BIN_MGPU): $(OBJ_MGPU_MAIN) $(OBJ_MGPU_SOLVER) $(OBJ_MGPU_IO) $(OBJ_MGPU_CSR) $(OBJ_MGPU_STENCIL)
	@mkdir -p $(BIN_DIR)
	mpic++ $^ -o $@ $(LDFLAGS) -lnccl -L/usr/local/cuda/lib64 -lcudart

# Convenience target
test_mgpu: $(BIN_MGPU)

# Multi-GPU partitioned test
BIN_MGPU_PART := $(BIN_DIR)/test_mgpu_cg_partitioned

# Partitioned MPI objects
OBJ_MGPU_PART_MAIN := $(OBJ_DIR)/mgpu/test_mgpu_cg_partitioned.o
OBJ_MGPU_PART_SOLVER := $(OBJ_DIR)/mgpu/cg_solver_mgpu_partitioned.o

# Compile partitioned sources
$(OBJ_DIR)/mgpu/test_mgpu_cg_partitioned.o: $(SRC_DIR)/main/test_mgpu_cg_partitioned.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I/usr/lib/x86_64-linux-gnu/openmpi/include -c $< -o $@

$(OBJ_DIR)/mgpu/cg_solver_mgpu_partitioned.o: $(SRC_DIR)/solvers/cg_solver_mgpu_partitioned.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I/usr/lib/x86_64-linux-gnu/openmpi/include -c $< -o $@

# Link partitioned solver
$(BIN_MGPU_PART): $(OBJ_MGPU_PART_MAIN) $(OBJ_MGPU_PART_SOLVER) $(OBJ_MGPU_IO) $(OBJ_MGPU_CSR) $(OBJ_MGPU_STENCIL)
	@mkdir -p $(BIN_DIR)
	mpic++ $^ -o $@ $(LDFLAGS) -lnccl -L/usr/local/cuda/lib64 -lcudart

# Convenience target
test_mgpu_part: $(BIN_MGPU_PART)

# Clean
clean:
	rm -rf build bin results

