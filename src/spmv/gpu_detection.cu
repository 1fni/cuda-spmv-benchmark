#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>
#include "spmv.h"

int get_gpu_properties(BenchmarkMetrics* metrics) {
    cudaDeviceProp prop;
    int device_id;
    
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // Copy GPU specifications
    strncpy(metrics->gpu_info.name, prop.name, sizeof(metrics->gpu_info.name) - 1);
    metrics->gpu_info.name[sizeof(metrics->gpu_info.name) - 1] = '\0';
    
    metrics->gpu_info.memory_mb = prop.totalGlobalMem / (1024 * 1024);
    snprintf(metrics->gpu_info.compute_capability, sizeof(metrics->gpu_info.compute_capability), 
             "%d.%d", prop.major, prop.minor);
    metrics->gpu_info.multiprocessor_count = prop.multiProcessorCount;
    metrics->gpu_info.max_threads_per_block = prop.maxThreadsPerBlock;
    metrics->gpu_info.memory_clock_khz = prop.memoryClockRate;
    metrics->gpu_info.graphics_clock_mhz = prop.clockRate / 1000;
    
    return 0;
}