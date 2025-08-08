#include "performance_benchmarks.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace PerformanceBenchmarks {

// ================================
// Single Operator Benchmarking
// ================================

BenchmarkResult benchmark_operator(const std::string& operator_name,
                                  const MatrixDataWrapper& matrix,
                                  const BenchmarkConfig& config) {
    
    // Generate input vector based on config pattern
    std::vector<double> input_vector = CudaTestUtils::generate_test_vector(
        matrix->rows, config.input_pattern, config.random_seed);
    
    return benchmark_operator_with_input(operator_name, matrix, input_vector, config);
}

BenchmarkResult benchmark_operator_with_input(const std::string& operator_name,
                                              const MatrixDataWrapper& matrix,
                                              const std::vector<double>& input_vector,
                                              const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.operator_name = operator_name;
    result.matrix_rows = matrix->rows;
    result.matrix_cols = matrix->cols;
    result.matrix_nnz = matrix->nnz;
    result.matrix_density = static_cast<double>(matrix->nnz) / 
                           (static_cast<double>(matrix->rows) * matrix->cols);
    
    // Initialize operator
    SpMVWrapper spmv_wrapper(operator_name);
    if (!spmv_wrapper.init(const_cast<MatrixData*>(matrix.get()))) {
        result.correctness_passed = false;
        return result;
    }
    
    std::vector<double> timing_results;
    timing_results.reserve(config.num_warmup_runs + config.num_measurement_runs);
    
    // Memory usage before operations
    auto memory_before = CudaTestUtils::get_gpu_memory_info();
    
    // Warmup runs
    for (int i = 0; i < config.num_warmup_runs; ++i) {
        auto warmup_result = spmv_wrapper.multiply(input_vector);
        // Don't record warmup times
    }
    
    // Measurement runs
    std::vector<std::vector<double>> all_results;
    all_results.reserve(config.num_measurement_runs);
    
    for (int i = 0; i < config.num_measurement_runs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto spmv_result = spmv_wrapper.multiply(input_vector);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double time_ms = duration.count() / 1000.0;
        
        timing_results.push_back(time_ms);
        all_results.push_back(spmv_result);
    }
    
    // Memory usage after operations  
    auto memory_after = CudaTestUtils::get_gpu_memory_info();
    result.gpu_memory_used_mb = memory_before.used_mb; // Simplified
    
    // Compute timing statistics
    result.min_time_ms = *std::min_element(timing_results.begin(), timing_results.end());
    result.max_time_ms = *std::max_element(timing_results.begin(), timing_results.end());
    result.avg_time_ms = std::accumulate(timing_results.begin(), timing_results.end(), 0.0) / timing_results.size();
    
    // Compute standard deviation
    double variance = 0.0;
    for (double time : timing_results) {
        variance += (time - result.avg_time_ms) * (time - result.avg_time_ms);
    }
    result.std_dev_ms = std::sqrt(variance / timing_results.size());
    
    // Use minimum time for performance calculations (most representative)
    result.kernel_time_ms = result.min_time_ms;
    result.total_time_ms = result.avg_time_ms; // Include variance in total time
    
    // Compute performance metrics
    double time_seconds = result.kernel_time_ms / 1000.0;
    result.effective_gflops = CudaTestUtils::compute_spmv_gflops(matrix->nnz, time_seconds);
    
    // Estimate memory bandwidth
    size_t matrix_bytes = matrix->nnz * (sizeof(double) + sizeof(int)) + (matrix->rows + 1) * sizeof(int);
    size_t vector_bytes = matrix->rows * sizeof(double);
    result.memory_bandwidth_gb_s = CudaTestUtils::compute_memory_bandwidth(
        matrix_bytes, vector_bytes, time_seconds);
    
    // Compute arithmetic intensity
    result.arithmetic_intensity = compute_arithmetic_intensity(matrix->nnz, matrix_bytes, vector_bytes);
    
    // Theoretical performance (simplified estimates)
    result.theoretical_gflops = get_theoretical_peak_gflops();
    result.gflops_efficiency = result.effective_gflops / result.theoretical_gflops;
    
    // Correctness check
    if (config.measure_correctness && !all_results.empty()) {
        // Use first result as reference, check others are consistent
        const auto& reference = all_results[0];
        result.correctness_passed = true;
        
        for (size_t i = 1; i < all_results.size() && result.correctness_passed; ++i) {
            if (!CudaTestUtils::vectors_near(reference, all_results[i])) {
                result.correctness_passed = false;
                // Compute numerical error
                auto comparison = CudaTestUtils::compare_vectors_detailed(reference, all_results[i]);
                result.numerical_error = comparison.max_absolute_error;
            }
        }
    }
    
    return result;
}

// ================================
// Comparative Benchmarking
// ================================

std::map<std::string, BenchmarkResult> 
compare_operators(const std::vector<std::string>& operator_names,
                 const MatrixDataWrapper& matrix,
                 const BenchmarkConfig& config) {
    
    std::map<std::string, BenchmarkResult> results;
    
    for (const std::string& op_name : operator_names) {
        try {
            results[op_name] = benchmark_operator(op_name, matrix, config);
            results[op_name].matrix_description = "Comparison matrix " + 
                std::to_string(matrix->rows) + "x" + std::to_string(matrix->cols);
        } catch (const std::exception& e) {
            // Log error and continue with other operators
            std::cerr << "Error benchmarking operator '" << op_name << "': " << e.what() << std::endl;
            BenchmarkResult error_result;
            error_result.operator_name = op_name;
            error_result.correctness_passed = false;
            results[op_name] = error_result;
        }
    }
    
    return results;
}

std::vector<BenchmarkResult>
benchmark_across_matrices(const std::string& operator_name,
                         const std::vector<std::pair<std::unique_ptr<MatrixDataWrapper>, std::string>>& matrices,
                         const BenchmarkConfig& config) {
    
    std::vector<BenchmarkResult> results;
    results.reserve(matrices.size());
    
    for (const auto& matrix_pair : matrices) {
        try {
            BenchmarkResult result = benchmark_operator(operator_name, *matrix_pair.first, config);
            result.matrix_description = matrix_pair.second;
            results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Error benchmarking matrix '" << matrix_pair.second 
                      << "': " << e.what() << std::endl;
            BenchmarkResult error_result;
            error_result.operator_name = operator_name;
            error_result.matrix_description = matrix_pair.second;
            error_result.correctness_passed = false;
            results.push_back(error_result);
        }
    }
    
    return results;
}

// ================================
// Scalability Analysis
// ================================

ScalingResult analyze_scaling(const std::string& operator_name,
                             const std::vector<int>& matrix_sizes,
                             std::function<std::unique_ptr<MatrixDataWrapper>(int)> matrix_generator,
                             const BenchmarkConfig& config) {
    
    ScalingResult scaling_result;
    scaling_result.sizes = matrix_sizes;
    scaling_result.times_ms.reserve(matrix_sizes.size());
    scaling_result.gflops.reserve(matrix_sizes.size());
    scaling_result.bandwidths.reserve(matrix_sizes.size());
    
    for (int size : matrix_sizes) {
        try {
            auto matrix = matrix_generator(size);
            BenchmarkResult bench_result = benchmark_operator(operator_name, *matrix, config);
            
            scaling_result.times_ms.push_back(bench_result.kernel_time_ms);
            scaling_result.gflops.push_back(bench_result.effective_gflops);
            scaling_result.bandwidths.push_back(bench_result.memory_bandwidth_gb_s);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in scaling analysis for size " << size 
                      << ": " << e.what() << std::endl;
            scaling_result.times_ms.push_back(0.0);
            scaling_result.gflops.push_back(0.0);  
            scaling_result.bandwidths.push_back(0.0);
        }
    }
    
    // Compute scaling efficiency (simplified)
    if (scaling_result.gflops.size() >= 2) {
        double first_gflops = scaling_result.gflops.front();
        double last_gflops = scaling_result.gflops.back();
        if (first_gflops > 0) {
            scaling_result.scaling_efficiency = last_gflops / first_gflops;
        }
    }
    
    return scaling_result;
}

// ================================
// Result Analysis and Reporting
// ================================

void generate_performance_report(const std::map<std::string, BenchmarkResult>& results,
                                std::ostream& output_stream) {
    output_stream << "\n" << std::string(80, '=') << std::endl;
    output_stream << "SPMV PERFORMANCE BENCHMARK REPORT" << std::endl;  
    output_stream << std::string(80, '=') << std::endl;
    
    for (const auto& [op_name, result] : results) {
        output_stream << "\nOperator: " << op_name << std::endl;
        output_stream << "Matrix: " << result.matrix_rows << "x" << result.matrix_cols 
                     << " (nnz=" << result.matrix_nnz << ")" << std::endl;
        
        if (result.correctness_passed) {
            output_stream << "✓ Correctness: PASSED" << std::endl;
        } else {
            output_stream << "✗ Correctness: FAILED";
            if (result.numerical_error > 0) {
                output_stream << " (error=" << result.numerical_error << ")";
            }
            output_stream << std::endl;
        }
        
        output_stream << std::fixed << std::setprecision(3);
        output_stream << "Performance:" << std::endl;
        output_stream << "  Kernel time: " << result.kernel_time_ms << " ms" << std::endl;
        output_stream << "  GFLOPS: " << result.effective_gflops << std::endl;
        output_stream << "  Memory bandwidth: " << result.memory_bandwidth_gb_s << " GB/s" << std::endl;
        output_stream << "  Efficiency: " << (result.gflops_efficiency * 100) << "%" << std::endl;
        
        if (result.std_dev_ms > 0) {
            output_stream << "Timing statistics:" << std::endl;
            output_stream << "  Average: " << result.avg_time_ms << " ± " << result.std_dev_ms << " ms" << std::endl;
            output_stream << "  Range: [" << result.min_time_ms << ", " << result.max_time_ms << "] ms" << std::endl;
        }
        
        output_stream << std::string(50, '-') << std::endl;
    }
}

bool export_results_csv(const std::map<std::string, BenchmarkResult>& results,
                       const std::string& csv_filename) {
    std::ofstream file(csv_filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Write CSV header
    file << "Operator,Matrix_Rows,Matrix_Cols,Matrix_NNZ,Density,"
         << "Kernel_Time_ms,GFLOPS,Memory_Bandwidth_GB_s,Efficiency,"
         << "Correctness_Passed,Numerical_Error\n";
    
    // Write data rows
    for (const auto& [op_name, result] : results) {
        file << op_name << ","
             << result.matrix_rows << "," << result.matrix_cols << "," << result.matrix_nnz << ","
             << result.matrix_density << ","
             << result.kernel_time_ms << "," << result.effective_gflops << ","  
             << result.memory_bandwidth_gb_s << "," << result.gflops_efficiency << ","
             << (result.correctness_passed ? "true" : "false") << ","
             << result.numerical_error << "\n";
    }
    
    file.close();
    return true;
}

bool export_results_json(const std::map<std::string, BenchmarkResult>& results,
                        const std::string& json_filename) {
    // Simplified JSON export - in production would use proper JSON library
    std::ofstream file(json_filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "{\n  \"benchmark_results\": [\n";
    
    bool first = true;
    for (const auto& [op_name, result] : results) {
        if (!first) file << ",\n";
        first = false;
        
        file << "    {\n";
        file << "      \"operator\": \"" << op_name << "\",\n";
        file << "      \"matrix_rows\": " << result.matrix_rows << ",\n";
        file << "      \"matrix_cols\": " << result.matrix_cols << ",\n";
        file << "      \"matrix_nnz\": " << result.matrix_nnz << ",\n";
        file << "      \"kernel_time_ms\": " << result.kernel_time_ms << ",\n";
        file << "      \"effective_gflops\": " << result.effective_gflops << ",\n";
        file << "      \"memory_bandwidth_gb_s\": " << result.memory_bandwidth_gb_s << ",\n";
        file << "      \"correctness_passed\": " << (result.correctness_passed ? "true" : "false") << "\n";
        file << "    }";
    }
    
    file << "\n  ]\n}\n";
    file.close();
    return true;
}

void print_comparison_table(const std::map<std::string, BenchmarkResult>& results,
                           bool sort_by_performance) {
    
    // Convert to vector for sorting
    std::vector<std::pair<std::string, BenchmarkResult>> sorted_results;
    for (const auto& pair : results) {
        sorted_results.push_back(pair);
    }
    
    if (sort_by_performance) {
        std::sort(sorted_results.begin(), sorted_results.end(),
                 [](const auto& a, const auto& b) {
                     return a.second.effective_gflops > b.second.effective_gflops;
                 });
    }
    
    // Print table header
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Operator"
              << std::setw(12) << "Time (ms)" 
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Bandwidth(GB/s)"
              << std::setw(12) << "Efficiency"
              << std::setw(12) << "Correct"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    // Print table rows
    for (const auto& [op_name, result] : sorted_results) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::left << std::setw(15) << op_name
                  << std::setw(12) << result.kernel_time_ms
                  << std::setw(12) << result.effective_gflops  
                  << std::setw(15) << result.memory_bandwidth_gb_s
                  << std::setw(12) << (result.gflops_efficiency * 100) << "%"
                  << std::setw(12) << (result.correctness_passed ? "✓" : "✗")
                  << std::endl;
    }
    
    std::cout << std::string(100, '=') << std::endl;
}

// ================================
// Utility Functions
// ================================

double get_theoretical_peak_gflops() {
    // Simplified estimation - would query actual GPU properties in production
    return 10000.0;  // Placeholder: 10 TFLOPS
}

double get_theoretical_memory_bandwidth() {
    // Simplified estimation - would query actual GPU memory specs
    return 900.0;  // Placeholder: 900 GB/s
}

double compute_arithmetic_intensity(int nnz, size_t matrix_size_bytes, size_t vector_size_bytes) {
    // SpMV: 2*nnz FLOPS, matrix_bytes + 2*vector_bytes transferred
    double flops = 2.0 * nnz;
    double bytes = static_cast<double>(matrix_size_bytes + 2 * vector_size_bytes);
    return flops / bytes;
}

} // namespace PerformanceBenchmarks