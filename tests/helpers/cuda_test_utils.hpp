/**
 * @file cuda_test_utils.hpp
 * @brief CUDA-specific utilities for SpMV testing and validation
 *
 * Provides GPU-friendly comparison functions, vector generators, and performance 
 * measurement tools specifically designed for rigorous SpMV testing in HPC environments.
 */

#ifndef CUDA_TEST_UTILS_HPP
#define CUDA_TEST_UTILS_HPP

#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

namespace CudaTestUtils {

    /**
     * @brief Performance metrics for SpMV operations
     */
    struct PerformanceMetrics {
        double kernel_time_ms;          ///< Kernel execution time in milliseconds
        double memory_bandwidth_gb_s;   ///< Effective memory bandwidth in GB/s
        double effective_gflops;        ///< Effective GFLOPS achieved
        size_t memory_footprint_mb;     ///< Total memory footprint in MB
        
        PerformanceMetrics() : kernel_time_ms(0), memory_bandwidth_gb_s(0), 
                              effective_gflops(0), memory_footprint_mb(0) {}
    };
    
    /**
     * @brief Test vector patterns for systematic testing
     */
    enum class TestVectorPattern {
        ONES,           ///< All elements = 1.0
        ZEROS,          ///< All elements = 0.0  
        INCREMENTAL,    ///< Elements = [1, 2, 3, ..., n]
        ALTERNATING,    ///< Elements = [1, -1, 1, -1, ...]
        RANDOM_UNIFORM, ///< Uniform random in [0, 1]
        RANDOM_NORMAL,  ///< Normal distribution N(0, 1)
        SPARSE_RANDOM   ///< Mostly zeros with random non-zeros
    };
    
    /**
     * @brief Tolerance levels for numerical comparisons
     */
    struct ToleranceConfig {
        double absolute_tol;  ///< Absolute tolerance
        double relative_tol;  ///< Relative tolerance
        
        ToleranceConfig(double abs_tol = 1e-10, double rel_tol = 1e-12) 
            : absolute_tol(abs_tol), relative_tol(rel_tol) {}
            
        // Predefined tolerance configurations
        static ToleranceConfig strict() { return ToleranceConfig(1e-12, 1e-14); }
        static ToleranceConfig standard() { return ToleranceConfig(1e-10, 1e-12); }
        static ToleranceConfig relaxed() { return ToleranceConfig(1e-8, 1e-10); }
    };

    // ================================
    // Vector Comparison Functions  
    // ================================
    
    /**
     * @brief Compare two vectors with configurable tolerance
     * @param a First vector
     * @param b Second vector  
     * @param config Tolerance configuration
     * @return true if vectors are equal within tolerance
     */
    bool vectors_near(const std::vector<double>& a, 
                      const std::vector<double>& b,
                      const ToleranceConfig& config = ToleranceConfig::standard());
    
    /**
     * @brief Element-wise comparison with detailed error reporting
     * @param a First vector
     * @param b Second vector
     * @param config Tolerance configuration
     * @param max_errors Maximum number of errors to report (0 = all)
     * @return Detailed comparison result with error statistics
     */
    struct ComparisonResult {
        bool passed;
        size_t total_elements;
        size_t failed_elements; 
        double max_absolute_error;
        double max_relative_error;
        std::vector<size_t> error_indices;  // First few error locations
    };
    
    ComparisonResult compare_vectors_detailed(const std::vector<double>& a,
                                             const std::vector<double>& b,
                                             const ToleranceConfig& config = ToleranceConfig::standard(),
                                             size_t max_errors = 10);
    
    // ================================
    // Vector Generation Functions
    // ================================
    
    /**
     * @brief Generate test vector with specified pattern
     * @param size Vector size
     * @param pattern Test pattern to generate
     * @param seed Random seed (for random patterns)
     * @return Generated test vector
     */
    std::vector<double> generate_test_vector(size_t size, 
                                           TestVectorPattern pattern,
                                           unsigned int seed = 42);
    
    /**
     * @brief Fill existing vector with test pattern  
     * @param vec Vector to fill
     * @param pattern Test pattern to use
     * @param seed Random seed (for random patterns)
     */
    void fill_test_vector(std::vector<double>& vec,
                         TestVectorPattern pattern, 
                         unsigned int seed = 42);
    
    // ================================  
    // Checksum and Validation Functions
    // ================================
    
    /**
     * @brief Compute stable checksum of vector
     * @param vec Input vector
     * @return Checksum (sum of all elements)
     */
    double compute_checksum(const std::vector<double>& vec);
    
    /**
     * @brief Compute L2 norm of vector
     * @param vec Input vector  
     * @return L2 norm
     */
    double compute_l2_norm(const std::vector<double>& vec);
    
    /**
     * @brief Validate SpMV result against expected checksum
     * @param result SpMV result vector
     * @param expected_checksum Expected checksum value
     * @param tolerance Absolute tolerance for checksum comparison
     * @return true if checksum matches expectation
     */
    bool validate_checksum(const std::vector<double>& result,
                          double expected_checksum,
                          double tolerance = 1e-10);
    
    // ================================
    // Performance Measurement Functions  
    // ================================
    
    /**
     * @brief RAII timer for GPU operations
     */
    class GpuTimer {
    private:
        cudaEvent_t start_, stop_;
        bool started_;
        
    public:
        GpuTimer();
        ~GpuTimer();
        
        void start();
        void stop(); 
        float elapsed_ms() const;
    };
    
    /**
     * @brief Compute theoretical GFLOPS for SpMV operation
     * @param nnz Number of non-zero elements
     * @param time_seconds Execution time in seconds
     * @return Theoretical GFLOPS (2*nnz operations / time)
     */
    double compute_spmv_gflops(size_t nnz, double time_seconds);
    
    /**
     * @brief Compute memory bandwidth for SpMV operation
     * @param matrix_size_bytes Matrix data size in bytes
     * @param vector_size_bytes Input/output vector sizes in bytes  
     * @param time_seconds Execution time in seconds
     * @return Memory bandwidth in GB/s
     */
    double compute_memory_bandwidth(size_t matrix_size_bytes,
                                   size_t vector_size_bytes, 
                                   double time_seconds);
    
    // ================================
    // Error Handling and Debugging
    // ================================
    
    /**
     * @brief Print vector statistics for debugging
     * @param vec Vector to analyze
     * @param name Vector name for output
     * @param max_elements Maximum elements to print (0 = print all)
     */
    void print_vector_stats(const std::vector<double>& vec,
                           const std::string& name,
                           size_t max_elements = 10);
    
    /**
     * @brief Check CUDA memory usage and report
     * @return Current GPU memory usage statistics  
     */
    struct MemoryInfo {
        size_t total_mb;
        size_t free_mb; 
        size_t used_mb;
    };
    
    MemoryInfo get_gpu_memory_info();
    
} // namespace CudaTestUtils

#endif // CUDA_TEST_UTILS_HPP