#include <gtest/gtest.h>
#include "wrappers/spmv_wrapper.hpp"
#include <vector>
#include <cmath>

/**
 * @brief Test fixture for SpMV wrapper basic functionality
 */
class SpMVWrapperTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test matrices are available in ../../matrix/ directory (from build/)
        test_matrix_3x3_ = "../../matrix/example3x3.mtx";
        test_matrix_5x5_ = "../../matrix/example5x5.mtx";
    }
    
    /**
     * @brief Compute checksum of a vector  
     * @param vec Input vector
     * @return Sum of all elements
     */
    double compute_checksum(const std::vector<double>& vec) {
        double sum = 0.0;
        for (double val : vec) {
            sum += val;
        }
        return sum;
    }
    
    std::string test_matrix_3x3_;
    std::string test_matrix_5x5_;
};

/**
 * @brief Test wrapper construction with valid operators
 */
TEST_F(SpMVWrapperTest, ConstructorWithValidOperators) {
    // Test all three available operators
    EXPECT_NO_THROW({
        SpMVWrapper csr_wrapper("SPMV_CSR");
        EXPECT_EQ(csr_wrapper.get_name(), "SPMV_CSR");
        EXPECT_FALSE(csr_wrapper.is_initialized());
    });
    
    EXPECT_NO_THROW({
        SpMVWrapper stencil_wrapper("SPMV_STENCIL5");
        EXPECT_EQ(stencil_wrapper.get_name(), "SPMV_STENCIL5");
        EXPECT_FALSE(stencil_wrapper.is_initialized());
    });
    
    EXPECT_NO_THROW({
        SpMVWrapper ellpack_wrapper("SPMV_ELLPACK");
        EXPECT_EQ(ellpack_wrapper.get_name(), "SPMV_ELLPACK");
        EXPECT_FALSE(ellpack_wrapper.is_initialized());
    });
}

/**
 * @brief Test wrapper construction with invalid operator
 */
TEST_F(SpMVWrapperTest, ConstructorWithInvalidOperator) {
    EXPECT_THROW({
        SpMVWrapper invalid_wrapper("INVALID_OPERATOR");
    }, std::invalid_argument);
}

/**
 * @brief Test MatrixDataWrapper with existing test file
 */
TEST_F(SpMVWrapperTest, MatrixDataWrapperLoad) {
    EXPECT_NO_THROW({
        MatrixDataWrapper matrix(test_matrix_3x3_);
        
        // Verify matrix was loaded
        EXPECT_GT(matrix->rows, 0);
        EXPECT_GT(matrix->cols, 0);
        EXPECT_GT(matrix->nnz, 0);
        EXPECT_NE(matrix->entries, nullptr);
        
        std::cout << "Loaded matrix: " << matrix->rows << "x" << matrix->cols 
                  << " with " << matrix->nnz << " non-zeros" << std::endl;
    });
}

/**
 * @brief Test wrapper initialization with CSR operator
 */
TEST_F(SpMVWrapperTest, CSRInitialization) {
    EXPECT_NO_THROW({
        MatrixDataWrapper matrix(test_matrix_3x3_);
        SpMVWrapper csr_wrapper("SPMV_CSR");
        
        EXPECT_TRUE(csr_wrapper.init(matrix.get()));
        EXPECT_TRUE(csr_wrapper.is_initialized());
        EXPECT_EQ(csr_wrapper.get_matrix_rows(), matrix->rows);
    });
}

/**
 * @brief Test CSR correctness with 3x3 stencil matrix using checksum
 */
TEST_F(SpMVWrapperTest, CSRCorrectnessStencil3x3) {
    MatrixDataWrapper matrix(test_matrix_3x3_);
    SpMVWrapper csr_wrapper("SPMV_CSR");
    
    ASSERT_TRUE(csr_wrapper.init(matrix.get()));
    
    // Test vector: all ones
    std::vector<double> x(matrix->rows, 1.0);
    std::vector<double> y = csr_wrapper.multiply(x);
    
    // Compute checksum
    double checksum = compute_checksum(y);
    
    // Expected checksum for 3x3 5-point stencil with vector of ones:
    // Each row sums to: -4 + (number of neighbors * -1)
    // Corner points (4 total): -4 + (-1*2) = -6
    // Edge points (4 total): -4 + (-1*3) = -7  
    // Center point (1 total): -4 + (-1*4) = -8
    // Total: 4*(-6) + 4*(-7) + 1*(-8) = -24 + -28 + -8 = -60
    double expected_checksum = -60.0;
    
    EXPECT_NEAR(checksum, expected_checksum, 1e-10) 
        << "CSR checksum mismatch. Got: " << checksum 
        << ", expected: " << expected_checksum;
    
    std::cout << "CSR 3x3 checksum test passed: " << checksum << std::endl;
}

/**
 * @brief Test STENCIL5 correctness with 3x3 matrix (should match CSR)
 */
TEST_F(SpMVWrapperTest, StencilCorrectnessStencil3x3) {
    MatrixDataWrapper matrix(test_matrix_3x3_);
    SpMVWrapper stencil_wrapper("SPMV_STENCIL5");
    
    ASSERT_TRUE(stencil_wrapper.init(matrix.get()));
    
    // Test vector: all ones
    std::vector<double> x(matrix->rows, 1.0);
    std::vector<double> y = stencil_wrapper.multiply(x);
    
    // Compute checksum
    double checksum = compute_checksum(y);
    
    // Should match CSR result exactly
    double expected_checksum = -60.0;
    
    EXPECT_NEAR(checksum, expected_checksum, 1e-10)
        << "STENCIL5 checksum mismatch. Got: " << checksum 
        << ", expected: " << expected_checksum;
    
    std::cout << "STENCIL5 3x3 checksum test passed: " << checksum << std::endl;
}

/**
 * @brief Cross-validation test: CSR vs STENCIL5 should give identical results
 */
TEST_F(SpMVWrapperTest, CrossValidationCSRvsStencil) {
    MatrixDataWrapper matrix(test_matrix_3x3_);
    
    // Initialize both operators
    SpMVWrapper csr_wrapper("SPMV_CSR");
    SpMVWrapper stencil_wrapper("SPMV_STENCIL5");
    
    ASSERT_TRUE(csr_wrapper.init(matrix.get()));
    ASSERT_TRUE(stencil_wrapper.init(matrix.get()));
    
    // Test with same input vector
    std::vector<double> x(matrix->rows, 1.0);
    
    std::vector<double> y_csr = csr_wrapper.multiply(x);
    std::vector<double> y_stencil = stencil_wrapper.multiply(x);
    
    // Compare checksums
    double checksum_csr = compute_checksum(y_csr);
    double checksum_stencil = compute_checksum(y_stencil);
    
    EXPECT_NEAR(checksum_csr, checksum_stencil, 1e-12)
        << "Cross-validation failed. CSR checksum: " << checksum_csr
        << ", STENCIL5 checksum: " << checksum_stencil;
    
    // Also compare element-wise for more rigorous validation
    ASSERT_EQ(y_csr.size(), y_stencil.size());
    for (size_t i = 0; i < y_csr.size(); ++i) {
        EXPECT_NEAR(y_csr[i], y_stencil[i], 1e-12)
            << "Element " << i << " differs: CSR=" << y_csr[i] 
            << ", STENCIL5=" << y_stencil[i];
    }
    
    std::cout << "Cross-validation CSR vs STENCIL5 passed. Checksum: " 
              << checksum_csr << std::endl;
}

/**
 * @brief Test error handling for uninitialized wrapper
 */
TEST_F(SpMVWrapperTest, ErrorHandlingUninitialized) {
    SpMVWrapper csr_wrapper("SPMV_CSR");
    std::vector<double> x(3, 1.0);
    
    // Should throw when not initialized
    EXPECT_THROW({
        csr_wrapper.multiply(x);
    }, std::runtime_error);
}

/**
 * @brief Test error handling for wrong vector size
 */
TEST_F(SpMVWrapperTest, ErrorHandlingWrongVectorSize) {
    MatrixDataWrapper matrix(test_matrix_3x3_);
    SpMVWrapper csr_wrapper("SPMV_CSR");
    
    ASSERT_TRUE(csr_wrapper.init(matrix.get()));
    
    // Wrong size vector
    std::vector<double> x_wrong_size(matrix->rows + 1, 1.0);
    
    EXPECT_THROW({
        csr_wrapper.multiply(x_wrong_size);
    }, std::runtime_error);
}