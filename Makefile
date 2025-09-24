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
INCLUDES := -I$(INC_DIR)
LDFLAGS := -lcusparse -lcublas

# AmgX integration (optional)
AMGX_DIR ?= /usr/local
ifneq ($(wildcard $(AMGX_DIR)/include/amgx_c.h),)
    INCLUDES += -I$(AMGX_DIR)/include
    LDFLAGS += -lcusolver -lamgx -L$(AMGX_DIR)/lib
    NVCCFLAGS += -DWITH_AMGX
    $(info AmgX found at $(AMGX_DIR) - enabling amgx-stencil mode)
else
    $(warning AmgX not found at $(AMGX_DIR) - skipping amgx-stencil mode)
    $(warning To enable AmgX: run ./scripts/install_amgx.sh or set AMGX_DIR)
endif

# Sources / objets
CU_SRCS := $(shell find $(SRC_DIR) -name '*.cu')
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

# SpMV : exclut generate_matrix
CU_SPMV_SRCS := $(filter-out $(SRC_DIR)/matrix/generate_matrix.cu, $(CU_SRCS))
CU_SPMV_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SPMV_SRCS))

# Générateur : juste generate_matrix + io
CU_GEN_SRCS := $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/io/io.cu
CU_GEN_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_GEN_SRCS))

# Binaries
BIN_SPMV := $(BIN_DIR)/spmv_bench
BIN_GEN  := $(BIN_DIR)/generate_matrix

# PHONY
.PHONY: all clean run

# Main target
all: $(BIN_SPMV) $(BIN_GEN)

# Link SpMV
$(BIN_SPMV): $(CU_SPMV_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Link générateur
$(BIN_GEN): $(CU_GEN_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

# Compile objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Run bench (example)
run: $(BIN_SPMV)
	@mkdir -p $(MAT_DIR) $(RES_DIR)
	$(BIN_SPMV) $(MAT_DIR)/example.mtx --mode=csr | tee $(RES_DIR)/run_$(BUILD_TYPE).log

# Clean
clean:
	rm -rf build bin results

