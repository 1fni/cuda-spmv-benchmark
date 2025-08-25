/**
 * @file matrix_fixtures.hpp
 * @brief Matrix generators and fixtures for systematic SpMV testing
 *
 * Provides specialized matrix generators with known analytical properties
 * for SpMV correctness and performance validation.
 */

#ifndef MATRIX_FIXTURES_HPP
#define MATRIX_FIXTURES_HPP

#include <vector>
#include <memory>
#include "../wrappers/spmv_wrapper.hpp"

extern "C" {
    #include "io.h"
}

namespace MatrixFixtures {

    /**
     * @brief Expected result for analytical matrices
     */
    struct AnalyticalResult {
        double expected_checksum;       ///< Expected checksum for vector of ones
        double expected_l2_norm;        ///< Expected L2 norm of result
        std::string description;        ///< Human-readable description
        
        AnalyticalResult(double checksum, double l2_norm, const std::string& desc)
            : expected_checksum(checksum), expected_l2_norm(l2_norm), description(desc) {}
    };

    // ================================
    // Simple Analytical Matrices
    // ================================
    
    /**
     * @brief Create 3x3 identity matrix
     * @return Matrix wrapper and expected result for input vector of ones
     */
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult> 
    identity_3x3();
    
    /**
     * @brief Create 5x5 diagonal matrix with coefficients [1, 2, 3, 4, 5]
     * @return Matrix wrapper and expected result for input vector of ones
     */
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    diagonal_5x5();
    
    /**
     * @brief Create 4x4 tridiagonal matrix with -1, 2, -1 pattern
     * @return Matrix wrapper and expected result for input vector of ones
     */
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    tridiagonal_4x4();
    
    /**
     * @brief Create upper triangular 3x3 matrix
     * @return Matrix wrapper and expected result for input vector of ones
     */
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    upper_triangular_3x3();
    
    // ================================
    // Stencil Pattern Matrices
    // ================================
    
    /**
     * @brief Create 5-point stencil matrix for given grid size
     * @param grid_size Grid dimension (matrix will be grid_size^2 x grid_size^2)
     * @return Matrix wrapper and expected result
     */
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    stencil_5point(int grid_size);
    
    /**
     * @brief Create 9-point stencil matrix for given grid size
     * @param grid_size Grid dimension
     * @return Matrix wrapper and expected result
     */
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    stencil_9point(int grid_size);
    
    // ================================
    // Structured Test Matrices  
    // ================================
    
    /**
     * @brief Create banded matrix with specified bandwidth
     * @param size Matrix size (square)
     * @param bandwidth Number of diagonals (1 = diagonal, 3 = tridiagonal, etc.)
     * @param diagonal_value Value on each diagonal
     * @return Matrix wrapper and expected result
     */
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    banded_matrix(int size, int bandwidth, double diagonal_value = 1.0);
    
    /**
     * @brief Create dense block matrix (useful for performance testing)
     * @param size Matrix size (square)
     * @param block_size Size of dense blocks
     * @param fill_value Value to fill non-zero entries
     * @return Matrix wrapper and expected result
     */
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    dense_block_matrix(int size, int block_size, double fill_value = 1.0);
    
    // ================================
    // Random and Stress Test Matrices
    // ================================
    
    /**
     * @brief Create random sparse matrix with controlled sparsity
     * @param size Matrix size (square)
     * @param sparsity_ratio Fraction of non-zero elements (0.0 to 1.0)
     * @param seed Random seed for reproducibility
     * @param value_range Range for random values [min, max]
     * @return Matrix wrapper (no analytical result - use for performance/stress testing)
     */
    std::unique_ptr<MatrixDataWrapper>
    random_sparse(int size, double sparsity_ratio, unsigned int seed = 42,
                  std::pair<double, double> value_range = {-1.0, 1.0});
    
    /**
     * @brief Create pathological matrix with extreme condition number
     * @param size Matrix size
     * @return Matrix wrapper for numerical stability testing
     */
    std::unique_ptr<MatrixDataWrapper>
    ill_conditioned_matrix(int size);
    
    /**
     * @brief Create matrix with highly unbalanced row lengths (for load balancing tests)
     * @param size Matrix size
     * @param seed Random seed
     * @return Matrix wrapper
     */
    std::unique_ptr<MatrixDataWrapper>
    unbalanced_rows_matrix(int size, unsigned int seed = 42);
    
    // ================================
    // Matrix Property Analysis
    // ================================
    
    /**
     * @brief Analyze matrix properties for test validation
     */
    struct MatrixProperties {
        int rows, cols, nnz;
        double density;                 ///< nnz / (rows * cols)
        double avg_nnz_per_row;        ///< Average non-zeros per row
        double max_nnz_per_row;        ///< Maximum non-zeros in any row
        double min_nnz_per_row;        ///< Minimum non-zeros in any row
        bool is_square;
        bool is_symmetric;             ///< Approximate symmetry check
        std::string sparsity_pattern;  ///< Description of sparsity pattern
    };
    
    /**
     * @brief Analyze properties of a matrix
     * @param matrix Matrix to analyze
     * @return Detailed property analysis
     */
    MatrixProperties analyze_matrix(const MatrixDataWrapper& matrix);
    
    /**
     * @brief Print matrix properties in human-readable format
     * @param props Matrix properties to display
     * @param matrix_name Name of matrix for output
     */
    void print_matrix_properties(const MatrixProperties& props, 
                                const std::string& matrix_name);
    
    // ================================
    // Utility Functions
    // ================================
    
    /**
     * @brief Create matrix from CSR arrays (for integration with existing code)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param row_ptr CSR row pointer array
     * @param col_ind CSR column index array  
     * @param values CSR values array
     * @param nnz Number of non-zeros
     * @return Matrix wrapper
     */
    std::unique_ptr<MatrixDataWrapper>
    from_csr_arrays(int rows, int cols, 
                   const int* row_ptr, const int* col_ind, const double* values,
                   int nnz);
    
    /**
     * @brief Save matrix to Matrix Market file for external validation
     * @param matrix Matrix to save
     * @param filename Output filename
     * @return true if save successful
     */
    bool save_to_file(const MatrixDataWrapper& matrix, const std::string& filename);
    
} // namespace MatrixFixtures

#endif // MATRIX_FIXTURES_HPP