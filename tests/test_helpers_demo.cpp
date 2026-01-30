#include <gtest/gtest.h>
#include "helpers/cuda_test_utils.hpp"
#include "helpers/matrix_fixtures.hpp"
#include "helpers/performance_benchmarks.hpp"
#include <iostream>

/**
 * @brief Test fixture for helpers demonstration
 */
class HelpersDemo : public ::testing::Test {
  protected:
    void SetUp() override {
        // Setup for helper tests
    }
};

/**
 * @brief Demonstrate CUDA test utilities
 */
TEST_F(HelpersDemo, CudaTestUtilitiesDemo) {
    std::cout << "\n=== CUDA Test Utilities Demo ===" << std::endl;

    // Generate different test vector patterns
    auto ones_vector =
        CudaTestUtils::generate_test_vector(5, CudaTestUtils::TestVectorPattern::ONES);
    auto incremental_vector =
        CudaTestUtils::generate_test_vector(5, CudaTestUtils::TestVectorPattern::INCREMENTAL);
    auto alternating_vector =
        CudaTestUtils::generate_test_vector(5, CudaTestUtils::TestVectorPattern::ALTERNATING);

    // Print vector statistics
    CudaTestUtils::print_vector_stats(ones_vector, "Ones vector", 5);
    CudaTestUtils::print_vector_stats(incremental_vector, "Incremental vector", 5);
    CudaTestUtils::print_vector_stats(alternating_vector, "Alternating vector", 5);

    // Test vector comparisons
    auto ones_copy = ones_vector;
    EXPECT_TRUE(CudaTestUtils::vectors_near(ones_vector, ones_copy));

    auto detailed_comparison = CudaTestUtils::compare_vectors_detailed(
        ones_vector, incremental_vector, CudaTestUtils::ToleranceConfig::standard());

    std::cout << "Comparison ones vs incremental:" << std::endl;
    std::cout << "  Passed: " << (detailed_comparison.passed ? "No" : "Yes (as expected)")
              << std::endl;
    std::cout << "  Failed elements: " << detailed_comparison.failed_elements << "/"
              << detailed_comparison.total_elements << std::endl;
    std::cout << "  Max absolute error: " << detailed_comparison.max_absolute_error << std::endl;

    EXPECT_FALSE(detailed_comparison.passed);           // Should fail as vectors are different
    EXPECT_EQ(detailed_comparison.failed_elements, 4);  // Only first element matches (1.0)
}

/**
 * @brief Demonstrate matrix fixtures
 */
TEST_F(HelpersDemo, MatrixFixturesDemo) {
    std::cout << "\n=== Matrix Fixtures Demo ===" << std::endl;

    // Test analytical matrices
    auto [identity_matrix, identity_result] = MatrixFixtures::identity_3x3();
    auto [diagonal_matrix, diagonal_result] = MatrixFixtures::diagonal_5x5();
    auto [tridiagonal_matrix, tridiag_result] = MatrixFixtures::tridiagonal_4x4();

    // Analyze and print matrix properties
    auto identity_props = MatrixFixtures::analyze_matrix(*identity_matrix);
    auto diagonal_props = MatrixFixtures::analyze_matrix(*diagonal_matrix);
    auto tridiag_props = MatrixFixtures::analyze_matrix(*tridiagonal_matrix);

    MatrixFixtures::print_matrix_properties(identity_props, "Identity 3x3");
    MatrixFixtures::print_matrix_properties(diagonal_props, "Diagonal 5x5");
    MatrixFixtures::print_matrix_properties(tridiag_props, "Tridiagonal 4x4");

    // Test expected results
    std::cout << "\nExpected results for input vector of ones:" << std::endl;
    std::cout << "Identity 3x3: checksum = " << identity_result.expected_checksum << std::endl;
    std::cout << "Diagonal 5x5: checksum = " << diagonal_result.expected_checksum << std::endl;
    std::cout << "Tridiagonal 4x4: checksum = " << tridiag_result.expected_checksum << std::endl;

    // Validate matrix properties
    EXPECT_EQ(identity_props.rows, 3);
    EXPECT_EQ(identity_props.nnz, 3);
    EXPECT_TRUE(identity_props.is_square);

    EXPECT_EQ(diagonal_props.rows, 5);
    EXPECT_EQ(diagonal_props.nnz, 5);
    EXPECT_DOUBLE_EQ(diagonal_result.expected_checksum, 15.0);

    EXPECT_EQ(tridiag_props.rows, 4);
    EXPECT_EQ(tridiag_props.nnz, 10);  // 4 diagonal + 6 off-diagonal
    EXPECT_DOUBLE_EQ(tridiag_result.expected_checksum, 2.0);
}

/**
 * @brief Demonstrate stencil matrix generation
 */
TEST_F(HelpersDemo, StencilMatrixDemo) {
    std::cout << "\n=== Stencil Matrix Demo ===" << std::endl;

    // Generate 9x9 stencil (matches example81x81.mtx)
    auto [stencil_9x9, stencil_result] = MatrixFixtures::stencil_5point(9);
    auto stencil_props = MatrixFixtures::analyze_matrix(*stencil_9x9);

    MatrixFixtures::print_matrix_properties(stencil_props, "5-point Stencil 9x9");

    std::cout << "Expected checksum for 9x9 stencil: " << stencil_result.expected_checksum
              << std::endl;
    std::cout << "Description: " << stencil_result.description << std::endl;

    // Validate against known result
    // 9x9 grid = 81 points
    // Checksum: 4*(-6) + 28*(-7) + 49*(-8) = -24 - 196 - 392 = -612
    EXPECT_EQ(stencil_props.rows, 81);
    EXPECT_EQ(stencil_props.cols, 81);
    EXPECT_DOUBLE_EQ(stencil_result.expected_checksum, -612.0);

    // Generate larger stencil for performance testing
    auto [stencil_5x5, stencil_5x5_result] = MatrixFixtures::stencil_5point(5);
    auto stencil_5x5_props = MatrixFixtures::analyze_matrix(*stencil_5x5);

    std::cout << "\n5x5 stencil properties:" << std::endl;
    std::cout << "  Size: " << stencil_5x5_props.rows << "x" << stencil_5x5_props.cols << std::endl;
    std::cout << "  NNZ: " << stencil_5x5_props.nnz << std::endl;
    std::cout << "  Expected checksum: " << stencil_5x5_result.expected_checksum << std::endl;

    EXPECT_EQ(stencil_5x5_props.rows, 25);  // 5x5 grid = 25 points
}

/**
 * @brief Demonstrate performance benchmarking (quick test)
 */
TEST_F(HelpersDemo, PerformanceBenchmarkDemo) {
    std::cout << "\n=== Performance Benchmark Demo ===" << std::endl;

    // Use quick benchmark config for demo
    auto config = PerformanceBenchmarks::BenchmarkConfig::quick();

    // Test with 3x3 stencil matrix
    auto [test_matrix, analytical_result] = MatrixFixtures::stencil_5point(3);

    // Benchmark CSR operator
    std::cout << "Benchmarking CSR operator on 3x3 stencil..." << std::endl;
    auto csr_result = PerformanceBenchmarks::benchmark_operator("csr", *test_matrix, config);

    std::cout << "CSR Results:" << std::endl;
    std::cout << "  Kernel time: " << csr_result.kernel_time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << csr_result.effective_gflops << std::endl;
    std::cout << "  Memory bandwidth: " << csr_result.memory_bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "  Correctness: " << (csr_result.correctness_passed ? "PASSED" : "FAILED")
              << std::endl;

    // Test operator comparison
    std::vector<std::string> operators = {"csr", "stencil5"};
    auto comparison_results =
        PerformanceBenchmarks::compare_operators(operators, *test_matrix, config);

    std::cout << "\n=== Operator Comparison ===" << std::endl;
    PerformanceBenchmarks::print_comparison_table(comparison_results, true);

    // Validate results
    EXPECT_TRUE(csr_result.correctness_passed);
    EXPECT_GT(csr_result.effective_gflops, 0.0);
    EXPECT_EQ(comparison_results.size(), 2);

    // Both operators should produce similar results
    if (comparison_results.count("csr") && comparison_results.count("stencil5")) {
        auto csr_res = comparison_results["csr"];
        auto stencil_res = comparison_results["stencil5"];

        EXPECT_TRUE(csr_res.correctness_passed);
        EXPECT_TRUE(stencil_res.correctness_passed);

        std::cout << "\nCross-validation: CSR vs STENCIL5 performance ratio = "
                  << (stencil_res.effective_gflops / csr_res.effective_gflops) << std::endl;
    }
}

/**
 * @brief Demonstrate GPU memory monitoring
 */
TEST_F(HelpersDemo, MemoryMonitoringDemo) {
    std::cout << "\n=== GPU Memory Monitoring Demo ===" << std::endl;

    auto memory_info = CudaTestUtils::get_gpu_memory_info();

    std::cout << "GPU Memory Status:" << std::endl;
    std::cout << "  Total: " << memory_info.total_mb << " MB" << std::endl;
    std::cout << "  Free: " << memory_info.free_mb << " MB" << std::endl;
    std::cout << "  Used: " << memory_info.used_mb << " MB" << std::endl;
    std::cout << "  Usage: " << std::fixed << std::setprecision(1)
              << (100.0 * memory_info.used_mb / memory_info.total_mb) << "%" << std::endl;

    EXPECT_GT(memory_info.total_mb, 0);
    EXPECT_GE(memory_info.free_mb, 0);
    EXPECT_GE(memory_info.used_mb, 0);
    EXPECT_EQ(memory_info.total_mb, memory_info.free_mb + memory_info.used_mb);
}