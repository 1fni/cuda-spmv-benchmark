#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <string.h>
#include <unistd.h>
#include "spmv.h"

static void get_cpu_info(char* cpu_model, size_t size) {
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        strncpy(cpu_model, "Unknown", size);
        return;
    }
    
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "model name", 10) == 0) {
            char* colon = strchr(line, ':');
            if (colon) {
                colon += 2; // Skip ": "
                strncpy(cpu_model, colon, size - 1);
                cpu_model[size - 1] = '\0';
                // Remove newline
                char* newline = strchr(cpu_model, '\n');
                if (newline) *newline = '\0';
                break;
            }
        }
    }
    fclose(fp);
}

static int get_system_ram_gb() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (pages * page_size) / (1024 * 1024 * 1024);
}

static void get_nvidia_smi_info(BenchmarkMetrics* metrics) {
    FILE* fp = popen("nvidia-smi --query-gpu=temperature.gpu,temperature.memory,power.draw,power.limit,persistence_mode --format=csv,noheader,nounits", "r");
    if (fp) {
        int temp_gpu, temp_mem, power_draw, power_limit;
        char persistence[16];
        if (fscanf(fp, "%d, %d, %d, %d, %15s", &temp_gpu, &temp_mem, &power_draw, &power_limit, persistence) == 5) {
            metrics->gpu_info.current_temp_c = temp_gpu;
            metrics->gpu_info.max_temp_c = (temp_mem > temp_gpu) ? temp_mem : temp_gpu;
            metrics->gpu_info.power_draw_w = power_draw;
            metrics->gpu_info.power_limit_w = power_limit;
            strncpy(metrics->gpu_info.persistence_mode, persistence, sizeof(metrics->gpu_info.persistence_mode) - 1);
        }
        pclose(fp);
    }
    
    // Get PCIe info
    fp = popen("nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits", "r");
    if (fp) {
        int pcie_gen, pcie_width;
        if (fscanf(fp, "%d, %d", &pcie_gen, &pcie_width) == 2) {
            snprintf(metrics->gpu_info.pcie_generation, sizeof(metrics->gpu_info.pcie_generation), "Gen%d", pcie_gen);
            metrics->gpu_info.pcie_link_width = pcie_width;
        }
        pclose(fp);
    }
}

int get_gpu_properties(BenchmarkMetrics* metrics) {
    cudaDeviceProp prop;
    int device_id;
    
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // Basic GPU properties
    strncpy(metrics->gpu_info.name, prop.name, sizeof(metrics->gpu_info.name) - 1);
    metrics->gpu_info.name[sizeof(metrics->gpu_info.name) - 1] = '\0';
    
    metrics->gpu_info.memory_mb = prop.totalGlobalMem / (1024 * 1024);
    snprintf(metrics->gpu_info.compute_capability, sizeof(metrics->gpu_info.compute_capability), 
             "%d.%d", prop.major, prop.minor);
    metrics->gpu_info.multiprocessor_count = prop.multiProcessorCount;
    metrics->gpu_info.max_threads_per_block = prop.maxThreadsPerBlock;
    metrics->gpu_info.memory_clock_khz = prop.memoryClockRate;
    metrics->gpu_info.graphics_clock_mhz = prop.clockRate / 1000;
    
    // CUDA/Driver versions
    CUDA_CHECK(cudaRuntimeGetVersion(&metrics->gpu_info.cuda_runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&metrics->gpu_info.cuda_driver_version));
    
    // cuSPARSE version
    cusparseHandle_t handle;
    if (cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS) {
        int cusparse_version;
        if (cusparseGetVersion(handle, &cusparse_version) == CUSPARSE_STATUS_SUCCESS) {
            metrics->gpu_info.cusparse_version = cusparse_version;
        }
        cusparseDestroy(handle);
    }
    
    // System info
    get_cpu_info(metrics->gpu_info.cpu_model, sizeof(metrics->gpu_info.cpu_model));
    metrics->gpu_info.system_ram_gb = get_system_ram_gb();
    
    // GPU runtime state via nvidia-smi
    get_nvidia_smi_info(metrics);
    
    return 0;
}