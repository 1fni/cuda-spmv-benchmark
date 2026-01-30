cmake_minimum_required(VERSION 3.18)

# Set CUDA architecture BEFORE project() for CI compatibility
# Auto-detect from environment or use native (requires CMake 3.24+)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to common architectures, excluding deprecated ones for CUDA 13+
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

project(SpMVTests LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type (default: Release, matching Makefile default)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(CUDA REQUIRED)

# Find Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Fallback: try to find gtest manually
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h
        PATHS /usr/include /usr/local/include)
    find_library(GTEST_LIBRARY gtest
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
    
    if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY)
        set(GTEST_LIBRARIES ${GTEST_LIBRARY})
        message(STATUS "Found Google Test: ${GTEST_LIBRARIES}")
    else()
        message(FATAL_ERROR "Google Test not found. Install with: sudo apt-get install libgtest-dev")
    endif()
else()
    set(GTEST_LIBRARIES GTest::gtest)
endif()

# Find cuSPARSE and cuBLAS (same as Makefile)
find_library(CUSPARSE_LIBRARY cusparse HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

if(NOT CUSPARSE_LIBRARY OR NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "cuSPARSE or cuBLAS not found")
endif()

# Include directories (same as Makefile: -I../include)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIR}
)

# Collect C/CUDA source files from main project (exclude main files and MPI sources)
file(GLOB_RECURSE PROJECT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu"
)
# Remove main files since we have our own main
list(FILTER PROJECT_SOURCES EXCLUDE REGEX "main.cu$")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX "generate_matrix.cu$")
# Remove files with main() functions
list(FILTER PROJECT_SOURCES EXCLUDE REGEX "src/main/cg_solver.cu$")
# Remove MPI-dependent sources (require mpi.h)
list(FILTER PROJECT_SOURCES EXCLUDE REGEX "cg_solver_mgpu_stencil.cu$")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX "cg_solver_mgpu_partitioned.cu$")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX "spmv_stencil_partitioned_halo_kernel.cu$")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX "benchmark_stats_mgpu_partitioned.cu$")

# Add the test executable
add_executable(spmv_tests
    test_main.cpp
    test_wrapper_basic.cpp
    test_helpers_demo.cpp
    test_cg_solver.cpp
    wrappers/spmv_wrapper.cpp
    helpers/cuda_test_utils.cpp
    helpers/matrix_fixtures.cpp
    helpers/performance_benchmarks.cpp
    ${PROJECT_SOURCES}
)

# Link libraries (same as Makefile: -lcusparse -lcublas)
target_link_libraries(spmv_tests
    ${GTEST_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUSPARSE_LIBRARY}
    ${CUBLAS_LIBRARY}
    pthread
)

# CUDA compilation flags - EXACTLY matching Makefile
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug flags: -g -G -O0 (same as Makefile)
    target_compile_options(spmv_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G -O0>
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
    )
else()
    # Release flags: -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true
    target_compile_options(spmv_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true>
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
    )
endif()

# Enable testing
enable_testing()
add_test(NAME SpMVTests COMMAND spmv_tests)

# Set CUDA properties
set_property(TARGET spmv_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "cuSPARSE library: ${CUSPARSE_LIBRARY}")
message(STATUS "cuBLAS library: ${CUBLAS_LIBRARY}")
message(STATUS "Google Test libraries: ${GTEST_LIBRARIES}")