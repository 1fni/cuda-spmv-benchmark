#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>

// Custom main with CUDA initialization
int main(int argc, char **argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize CUDA context
    int device_count;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(cuda_status) << std::endl;
        return -1;
    }
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return -1;
    }
    
    // Set device 0 as default
    cudaSetDevice(0);
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Run all tests
    int result = RUN_ALL_TESTS();
    
    // Clean up CUDA context
    cudaDeviceReset();
    
    return result;
}