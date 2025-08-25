/**
 * @file performance_benchmarks.hpp
 * @brief Performance benchmarking utilities for SpMV operations
 *
 * Provides performance measurement, comparison, and analysis tools
 * specifically designed for evaluating SpMV implementations in HPC environments.
 */

#ifndef PERFORMANCE_BENCHMARKS_HPP
#define PERFORMANCE_BENCHMARKS_HPP

#include "cuda_test_utils.hpp"
#include "../wrappers/spmv_wrapper.hpp"
#include "matrix_fixtures.hpp"
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <functional>

namespace PerformanceBenchmarks {

    /**
     * @brief Performance measurement result
     */
    struct BenchmarkResult {
        std::string operator_name;          ///< SpMV operator name
        std::string matrix_description;     ///< Matrix description
        
        // Timing metrics
        double kernel_time_ms;              ///< Kernel execution time
        double total_time_ms;               ///< Total operation time (including transfers)
        double min_time_ms;                 ///< Minimum time over multiple runs
        double max_time_ms;                 ///< Maximum time over multiple runs
        double avg_time_ms;                 ///< Average time over multiple runs
        double std_dev_ms;                  ///< Standard deviation of times
        
        // Performance metrics
        double effective_gflops;            ///< Effective GFLOPS achieved
        double theoretical_gflops;          ///< Theoretical peak GFLOPS for this operation
        double gflops_efficiency;           ///< effective_gflops / theoretical_gflops
        double memory_bandwidth_gb_s;       ///< Effective memory bandwidth
        double arithmetic_intensity;        ///< FLOPS per byte transferred
        
        // Matrix characteristics
        int matrix_rows;                    ///< Matrix dimensions
        int matrix_cols;
        int matrix_nnz;                     ///< Number of non-zeros
        double matrix_density;              ///< Sparsity ratio
        
        // Resource usage
        size_t gpu_memory_used_mb;          ///< GPU memory footprint
        double cpu_memory_used_mb;          ///< CPU memory usage
        
        // Quality metrics
        bool correctness_passed;            ///< Did correctness check pass?
        double numerical_error;             ///< Error compared to reference
        
        BenchmarkResult() : kernel_time_ms(0), total_time_ms(0), min_time_ms(0), max_time_ms(0),
                           avg_time_ms(0), std_dev_ms(0), effective_gflops(0), theoretical_gflops(0),
                           gflops_efficiency(0), memory_bandwidth_gb_s(0), arithmetic_intensity(0),
                           matrix_rows(0), matrix_cols(0), matrix_nnz(0), matrix_density(0),
                           gpu_memory_used_mb(0), cpu_memory_used_mb(0),
                           correctness_passed(false), numerical_error(0) {}
    };
    
    /**
     * @brief Configuration for benchmark runs
     */
    struct BenchmarkConfig {
        int num_warmup_runs;                ///< Number of warmup iterations
        int num_measurement_runs;           ///< Number of measurement iterations
        bool measure_correctness;           ///< Perform correctness validation
        bool measure_memory_usage;          ///< Track memory consumption
        bool detailed_timing;               ///< Collect detailed timing statistics
        CudaTestUtils::TestVectorPattern input_pattern;  ///< Input vector pattern
        unsigned int random_seed;           ///< Seed for reproducible tests
        
        BenchmarkConfig() : num_warmup_runs(3), num_measurement_runs(10),
                           measure_correctness(true), measure_memory_usage(true),
                           detailed_timing(true), 
                           input_pattern(CudaTestUtils::TestVectorPattern::ONES),
                           random_seed(42) {}
                           
        // Predefined configurations
        static BenchmarkConfig quick() {
            BenchmarkConfig config;
            config.num_warmup_runs = 1;
            config.num_measurement_runs = 3;
            config.detailed_timing = false;
            return config;
        }
        
        static BenchmarkConfig comprehensive() {
            BenchmarkConfig config;
            config.num_warmup_runs = 5;
            config.num_measurement_runs = 20;
            config.measure_memory_usage = true;
            config.detailed_timing = true;
            return config;
        }
        
        static BenchmarkConfig stress_test() {
            BenchmarkConfig config;
            config.num_warmup_runs = 2;
            config.num_measurement_runs = 100;
            config.measure_correctness = false;  // Skip for stress testing
            return config;
        }
    };

    // ================================
    // Single Operator Benchmarking
    // ================================
    
    /**
     * @brief Benchmark a single SpMV operator on a matrix
     * @param operator_name Name of SpMV operator ("csr", "stencil5", "ellpack")
     * @param matrix Matrix to test with
     * @param config Benchmark configuration
     * @return Benchmark results
     */
    BenchmarkResult benchmark_operator(const std::string& operator_name,
                                      const MatrixDataWrapper& matrix,
                                      const BenchmarkConfig& config = BenchmarkConfig());
    
    /**
     * @brief Benchmark operator with custom input vector
     * @param operator_name SpMV operator name  
     * @param matrix Test matrix
     * @param input_vector Custom input vector
     * @param config Benchmark configuration
     * @return Benchmark results
     */
    BenchmarkResult benchmark_operator_with_input(const std::string& operator_name,
                                                  const MatrixDataWrapper& matrix,
                                                  const std::vector<double>& input_vector,
                                                  const BenchmarkConfig& config = BenchmarkConfig());
    
    // ================================
    // Comparative Benchmarking
    // ================================
    
    /**
     * @brief Compare multiple operators on the same matrix
     * @param operator_names List of operators to compare
     * @param matrix Test matrix  
     * @param config Benchmark configuration
     * @return Results for each operator
     */
    std::map<std::string, BenchmarkResult> 
    compare_operators(const std::vector<std::string>& operator_names,
                     const MatrixDataWrapper& matrix,
                     const BenchmarkConfig& config = BenchmarkConfig());
    
    /**
     * @brief Benchmark single operator across multiple matrices
     * @param operator_name SpMV operator to test
     * @param matrices Vector of test matrices with descriptions
     * @param config Benchmark configuration
     * @return Results for each matrix
     */
    std::vector<BenchmarkResult>
    benchmark_across_matrices(const std::string& operator_name,
                             const std::vector<std::pair<std::unique_ptr<MatrixDataWrapper>, std::string>>& matrices,
                             const BenchmarkConfig& config = BenchmarkConfig());
    
    // ================================
    // Scalability Analysis
    // ================================
    
    /**
     * @brief Analyze performance scaling with matrix size
     * @param operator_name SpMV operator to test
     * @param matrix_sizes Vector of matrix sizes to test
     * @param matrix_generator Function to generate matrix of given size
     * @param config Benchmark configuration
     * @return Scaling analysis results
     */
    struct ScalingResult {
        std::vector<int> sizes;
        std::vector<double> times_ms;
        std::vector<double> gflops;
        std::vector<double> bandwidths;
        double scaling_efficiency;          ///< How well performance scales
    };
    
    ScalingResult analyze_scaling(const std::string& operator_name,
                                 const std::vector<int>& matrix_sizes,
                                 std::function<std::unique_ptr<MatrixDataWrapper>(int)> matrix_generator,
                                 const BenchmarkConfig& config = BenchmarkConfig());
    
    // ================================
    // Performance Regression Testing
    // ================================
    
    /**
     * @brief Performance regression test suite
     */
    class RegressionTester {
    private:
        std::map<std::string, BenchmarkResult> baseline_results_;
        double performance_tolerance_;      ///< Acceptable performance regression (e.g., 0.1 = 10%)
        
    public:
        RegressionTester(double tolerance = 0.1) : performance_tolerance_(tolerance) {}
        
        /**
         * @brief Set baseline performance results
         * @param baseline_file Path to baseline results file (JSON format)
         * @return true if baseline loaded successfully
         */
        bool load_baseline(const std::string& baseline_file);
        
        /**
         * @brief Save current results as new baseline
         * @param baseline_file Output path for baseline results
         * @param results Current benchmark results
         * @return true if save successful
         */
        bool save_baseline(const std::string& baseline_file,
                          const std::map<std::string, BenchmarkResult>& results);
        
        /**
         * @brief Check for performance regressions
         * @param current_results Current benchmark results
         * @return Regression analysis report
         */
        struct RegressionReport {
            bool passed;
            std::vector<std::string> regressions;    ///< List of regressed tests
            std::vector<std::string> improvements;   ///< List of improved tests
            double worst_regression_pct;             ///< Worst regression percentage
            double best_improvement_pct;             ///< Best improvement percentage
        };
        
        RegressionReport check_regressions(const std::map<std::string, BenchmarkResult>& current_results);
    };
    
    // ================================
    // Result Analysis and Reporting
    // ================================
    
    /**
     * @brief Generate performance summary report
     * @param results Benchmark results to summarize
     * @param output_stream Output stream for report
     */
    void generate_performance_report(const std::map<std::string, BenchmarkResult>& results,
                                    std::ostream& output_stream);
    
    /**
     * @brief Export results to CSV format for analysis
     * @param results Benchmark results
     * @param csv_filename Output CSV file path
     * @return true if export successful
     */
    bool export_results_csv(const std::map<std::string, BenchmarkResult>& results,
                           const std::string& csv_filename);
    
    /**
     * @brief Export results to JSON format
     * @param results Benchmark results  
     * @param json_filename Output JSON file path
     * @return true if export successful
     */
    bool export_results_json(const std::map<std::string, BenchmarkResult>& results,
                            const std::string& json_filename);
    
    /**
     * @brief Print comparison table to console
     * @param results Results from multiple operators/matrices
     * @param sort_by_performance Sort results by performance metric
     */
    void print_comparison_table(const std::map<std::string, BenchmarkResult>& results,
                               bool sort_by_performance = true);
    
    // ================================
    // Utility Functions
    // ================================
    
    /**
     * @brief Estimate theoretical peak GFLOPS for current GPU
     * @return Theoretical peak GFLOPS
     */
    double get_theoretical_peak_gflops();
    
    /**
     * @brief Estimate theoretical memory bandwidth for current GPU  
     * @return Theoretical peak memory bandwidth in GB/s
     */
    double get_theoretical_memory_bandwidth();
    
    /**
     * @brief Compute arithmetic intensity for SpMV operation
     * @param nnz Number of non-zeros in matrix
     * @param matrix_size_bytes Size of matrix data in bytes
     * @param vector_size_bytes Size of input/output vectors in bytes
     * @return Arithmetic intensity (FLOPS per byte)
     */
    double compute_arithmetic_intensity(int nnz, size_t matrix_size_bytes, size_t vector_size_bytes);

} // namespace PerformanceBenchmarks

#endif // PERFORMANCE_BENCHMARKS_HPP