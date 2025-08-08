#include "cuda_test_utils.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace CudaTestUtils {

// ================================
// Vector Comparison Functions  
// ================================

bool vectors_near(const std::vector<double>& a, 
                  const std::vector<double>& b,
                  const ToleranceConfig& config) {
    if (a.size() != b.size()) return false;
    
    for (size_t i = 0; i < a.size(); ++i) {
        double abs_diff = std::abs(a[i] - b[i]);
        double rel_diff = 0.0;
        
        if (std::abs(a[i]) > 0.0 || std::abs(b[i]) > 0.0) {
            rel_diff = abs_diff / std::max(std::abs(a[i]), std::abs(b[i]));
        }
        
        if (abs_diff > config.absolute_tol && rel_diff > config.relative_tol) {
            return false;
        }
    }
    return true;
}

ComparisonResult compare_vectors_detailed(const std::vector<double>& a,
                                         const std::vector<double>& b,
                                         const ToleranceConfig& config,
                                         size_t max_errors) {
    ComparisonResult result;
    result.total_elements = std::min(a.size(), b.size());
    result.failed_elements = 0;
    result.max_absolute_error = 0.0;
    result.max_relative_error = 0.0;
    result.passed = true;
    
    if (a.size() != b.size()) {
        result.passed = false;
        return result;
    }
    
    for (size_t i = 0; i < a.size(); ++i) {
        double abs_diff = std::abs(a[i] - b[i]);
        double rel_diff = 0.0;
        
        if (std::abs(a[i]) > 0.0 || std::abs(b[i]) > 0.0) {
            rel_diff = abs_diff / std::max(std::abs(a[i]), std::abs(b[i]));
        }
        
        // Update maximum errors
        result.max_absolute_error = std::max(result.max_absolute_error, abs_diff);
        result.max_relative_error = std::max(result.max_relative_error, rel_diff);
        
        // Check if this element fails tolerance
        if (abs_diff > config.absolute_tol && rel_diff > config.relative_tol) {
            result.failed_elements++;
            result.passed = false;
            
            // Store error location (up to max_errors)
            if (max_errors == 0 || result.error_indices.size() < max_errors) {
                result.error_indices.push_back(i);
            }
        }
    }
    
    return result;
}

// ================================
// Vector Generation Functions
// ================================

std::vector<double> generate_test_vector(size_t size, 
                                        TestVectorPattern pattern,
                                        unsigned int seed) {
    std::vector<double> vec(size);
    fill_test_vector(vec, pattern, seed);
    return vec;
}

void fill_test_vector(std::vector<double>& vec,
                     TestVectorPattern pattern, 
                     unsigned int seed) {
    std::mt19937 gen(seed);
    
    switch (pattern) {
        case TestVectorPattern::ONES:
            std::fill(vec.begin(), vec.end(), 1.0);
            break;
            
        case TestVectorPattern::ZEROS:
            std::fill(vec.begin(), vec.end(), 0.0);
            break;
            
        case TestVectorPattern::INCREMENTAL:
            for (size_t i = 0; i < vec.size(); ++i) {
                vec[i] = static_cast<double>(i + 1);
            }
            break;
            
        case TestVectorPattern::ALTERNATING:
            for (size_t i = 0; i < vec.size(); ++i) {
                vec[i] = (i % 2 == 0) ? 1.0 : -1.0;
            }
            break;
            
        case TestVectorPattern::RANDOM_UNIFORM: {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (double& val : vec) {
                val = dist(gen);
            }
            break;
        }
        
        case TestVectorPattern::RANDOM_NORMAL: {
            std::normal_distribution<double> dist(0.0, 1.0);
            for (double& val : vec) {
                val = dist(gen);
            }
            break;
        }
        
        case TestVectorPattern::SPARSE_RANDOM: {
            std::uniform_real_distribution<double> value_dist(0.0, 1.0);
            std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
            const double sparsity_prob = 0.1; // 10% non-zero
            
            for (double& val : vec) {
                if (prob_dist(gen) < sparsity_prob) {
                    val = value_dist(gen);
                } else {
                    val = 0.0;
                }
            }
            break;
        }
    }
}

// ================================  
// Checksum and Validation Functions
// ================================

double compute_checksum(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double val : vec) {
        sum += val;
    }
    return sum;
}

double compute_l2_norm(const std::vector<double>& vec) {
    double sum_squares = 0.0;
    for (double val : vec) {
        sum_squares += val * val;
    }
    return std::sqrt(sum_squares);
}

bool validate_checksum(const std::vector<double>& result,
                      double expected_checksum,
                      double tolerance) {
    double actual_checksum = compute_checksum(result);
    return std::abs(actual_checksum - expected_checksum) <= tolerance;
}

// ================================
// Performance Measurement Functions  
// ================================

GpuTimer::GpuTimer() : started_(false) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void GpuTimer::start() {
    cudaEventRecord(start_);
    started_ = true;
}

void GpuTimer::stop() {
    if (started_) {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }
}

float GpuTimer::elapsed_ms() const {
    if (!started_) return 0.0f;
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
}

double compute_spmv_gflops(size_t nnz, double time_seconds) {
    // SpMV: y = A*x requires 2*nnz operations (1 multiply + 1 add per non-zero)
    double operations = 2.0 * static_cast<double>(nnz);
    return (operations / time_seconds) / 1e9; // Convert to GFLOPS
}

double compute_memory_bandwidth(size_t matrix_size_bytes,
                               size_t vector_size_bytes, 
                               double time_seconds) {
    // Total memory transferred: matrix + input vector + output vector
    size_t total_bytes = matrix_size_bytes + 2 * vector_size_bytes;
    double gb_transferred = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
    return gb_transferred / time_seconds;
}

// ================================
// Error Handling and Debugging
// ================================

void print_vector_stats(const std::vector<double>& vec,
                       const std::string& name,
                       size_t max_elements) {
    if (vec.empty()) {
        std::cout << name << ": [empty vector]" << std::endl;
        return;
    }
    
    // Compute statistics
    double min_val = *std::min_element(vec.begin(), vec.end());
    double max_val = *std::max_element(vec.begin(), vec.end());
    double sum = compute_checksum(vec);
    double mean = sum / static_cast<double>(vec.size());
    double l2_norm = compute_l2_norm(vec);
    
    std::cout << name << " statistics:" << std::endl;
    std::cout << "  Size: " << vec.size() << std::endl;
    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "  Sum: " << sum << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  L2 norm: " << l2_norm << std::endl;
    
    // Print first few elements
    if (max_elements > 0) {
        size_t print_count = std::min(max_elements, vec.size());
        std::cout << "  First " << print_count << " elements: [";
        for (size_t i = 0; i < print_count; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(6) << vec[i];
        }
        if (vec.size() > max_elements) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
    }
}

MemoryInfo get_gpu_memory_info() {
    MemoryInfo info;
    size_t free_bytes, total_bytes;
    
    cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status == cudaSuccess) {
        info.total_mb = total_bytes / (1024 * 1024);
        info.free_mb = free_bytes / (1024 * 1024);
        info.used_mb = info.total_mb - info.free_mb;
    } else {
        info.total_mb = info.free_mb = info.used_mb = 0;
    }
    
    return info;
}

} // namespace CudaTestUtils