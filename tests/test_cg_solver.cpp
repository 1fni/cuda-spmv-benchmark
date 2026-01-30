/**
 * @file test_cg_solver.cpp
 * @brief Tests for CG solver functionality
 *
 * Tests the Conjugate Gradient solver with stencil matrices.
 * Note: Full multi-GPU tests require MPI and are run separately.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>

// Include project headers
#include "io.h"

/**
 * @brief Test fixture for CG solver tests
 */
class CGSolverTest : public ::testing::Test {
  protected:
    void SetUp() override { test_matrix_81x81_ = "../../matrix/example81x81.mtx"; }

    /**
     * @brief Compute L2 norm of a vector
     */
    double compute_norm(const std::vector<double>& vec) {
        double sum = 0.0;
        for (double val : vec) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Compute residual norm ||b - Ax|| / ||b||
     */
    double compute_relative_residual(const double* A_values, const int* A_row_ptr,
                                     const int* A_col_idx, int n, const double* x,
                                     const double* b) {
        std::vector<double> Ax(n, 0.0);

        // Compute Ax (CSR SpMV)
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = A_row_ptr[i]; j < A_row_ptr[i + 1]; j++) {
                sum += A_values[j] * x[A_col_idx[j]];
            }
            Ax[i] = sum;
        }

        // Compute residual r = b - Ax
        std::vector<double> r(n);
        double r_norm_sq = 0.0;
        double b_norm_sq = 0.0;
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - Ax[i];
            r_norm_sq += r[i] * r[i];
            b_norm_sq += b[i] * b[i];
        }

        return std::sqrt(r_norm_sq) / std::sqrt(b_norm_sq);
    }

    std::string test_matrix_81x81_;
};

/**
 * @brief Test matrix loading for CG solver
 */
TEST_F(CGSolverTest, MatrixLoadForCG) {
    MatrixData mat;
    int result = load_matrix_market(test_matrix_81x81_.c_str(), &mat);

    ASSERT_EQ(result, 0) << "Failed to load matrix";
    EXPECT_EQ(mat.rows, 81);
    EXPECT_EQ(mat.cols, 81);
    EXPECT_GT(mat.nnz, 0);

    std::cout << "Loaded matrix for CG: " << mat.rows << "x" << mat.cols << " with " << mat.nnz
              << " non-zeros" << std::endl;

    // Cleanup
    free(mat.entries);
}

/**
 * @brief Test CSR conversion for CG solver
 */
TEST_F(CGSolverTest, CSRConversionForCG) {
    MatrixData mat;
    ASSERT_EQ(load_matrix_market(test_matrix_81x81_.c_str(), &mat), 0);

    // Convert to CSR
    std::vector<double> values;
    std::vector<int> row_ptr(mat.rows + 1, 0);
    std::vector<int> col_idx;

    // Count nnz per row
    for (int i = 0; i < mat.nnz; i++) {
        row_ptr[mat.entries[i].row + 1]++;
    }

    // Prefix sum
    for (int i = 1; i <= mat.rows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }

    // Fill arrays
    values.resize(mat.nnz);
    col_idx.resize(mat.nnz);
    std::vector<int> row_count(mat.rows, 0);

    for (int i = 0; i < mat.nnz; i++) {
        int row = mat.entries[i].row;
        int idx = row_ptr[row] + row_count[row];
        values[idx] = mat.entries[i].value;
        col_idx[idx] = mat.entries[i].col;
        row_count[row]++;
    }

    // Verify CSR structure
    EXPECT_EQ(row_ptr[mat.rows], mat.nnz);
    EXPECT_EQ(values.size(), static_cast<size_t>(mat.nnz));
    EXPECT_EQ(col_idx.size(), static_cast<size_t>(mat.nnz));

    std::cout << "CSR conversion successful: " << mat.nnz << " non-zeros" << std::endl;

    free(mat.entries);
}

/**
 * @brief Test that stencil matrix is symmetric positive definite (required for CG)
 *
 * For a 5-point stencil Laplacian, the matrix should be:
 * - Symmetric: A[i][j] = A[j][i]
 * - Diagonally dominant (implies SPD for this type of matrix)
 */
TEST_F(CGSolverTest, MatrixIsSPD) {
    MatrixData mat;
    ASSERT_EQ(load_matrix_market(test_matrix_81x81_.c_str(), &mat), 0);

    // Check symmetry by verifying each off-diagonal entry has a transpose
    bool is_symmetric = true;
    for (int i = 0; i < mat.nnz && is_symmetric; i++) {
        int row = mat.entries[i].row;
        int col = mat.entries[i].col;
        double val = mat.entries[i].value;

        if (row != col) {  // Off-diagonal
            // Find transpose entry
            bool found = false;
            for (int j = 0; j < mat.nnz; j++) {
                if (mat.entries[j].row == col && mat.entries[j].col == row) {
                    if (std::abs(mat.entries[j].value - val) < 1e-12) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                is_symmetric = false;
                std::cout << "Missing symmetric entry for (" << row << "," << col << ")"
                          << std::endl;
            }
        }
    }

    EXPECT_TRUE(is_symmetric) << "Matrix should be symmetric for CG";

    // Check diagonal dominance
    std::vector<double> diag(mat.rows, 0.0);
    std::vector<double> off_diag_sum(mat.rows, 0.0);

    for (int i = 0; i < mat.nnz; i++) {
        int row = mat.entries[i].row;
        int col = mat.entries[i].col;
        double val = std::abs(mat.entries[i].value);

        if (row == col) {
            diag[row] = val;
        } else {
            off_diag_sum[row] += val;
        }
    }

    bool is_diag_dominant = true;
    for (int i = 0; i < mat.rows; i++) {
        if (diag[i] < off_diag_sum[i]) {
            is_diag_dominant = false;
            break;
        }
    }

    EXPECT_TRUE(is_diag_dominant) << "Matrix should be diagonally dominant";

    std::cout << "Matrix SPD check passed (symmetric + diagonally dominant)" << std::endl;

    free(mat.entries);
}

/**
 * @brief Test simple CG iteration (CPU reference)
 *
 * This tests the CG algorithm logic without GPU/MPI dependencies.
 */
TEST_F(CGSolverTest, SimpleCGIteration) {
    MatrixData mat;
    ASSERT_EQ(load_matrix_market(test_matrix_81x81_.c_str(), &mat), 0);

    int n = mat.rows;

    // Convert to CSR
    std::vector<double> values(mat.nnz);
    std::vector<int> row_ptr(n + 1, 0);
    std::vector<int> col_idx(mat.nnz);

    for (int i = 0; i < mat.nnz; i++) {
        row_ptr[mat.entries[i].row + 1]++;
    }
    for (int i = 1; i <= n; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }

    std::vector<int> row_count(n, 0);
    for (int i = 0; i < mat.nnz; i++) {
        int row = mat.entries[i].row;
        int idx = row_ptr[row] + row_count[row];
        values[idx] = mat.entries[i].value;
        col_idx[idx] = mat.entries[i].col;
        row_count[row]++;
    }

    // Setup CG: solve Ax = b where b = A * ones (so x should converge to ones)
    std::vector<double> x_true(n, 1.0);
    std::vector<double> b(n, 0.0);

    // Compute b = A * x_true
    for (int i = 0; i < n; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            b[i] += values[j] * x_true[col_idx[j]];
        }
    }

    // Initial guess x = 0
    std::vector<double> x(n, 0.0);
    std::vector<double> r(n);   // residual
    std::vector<double> p(n);   // search direction
    std::vector<double> Ap(n);  // A * p

    // r = b - Ax (since x=0, r = b)
    for (int i = 0; i < n; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }

    double rs_old = 0.0;
    for (int i = 0; i < n; i++) {
        rs_old += r[i] * r[i];
    }

    double b_norm = std::sqrt(rs_old);
    double tol = 1e-6;
    int max_iter = 100;

    std::cout << "Starting CG: initial residual norm = " << std::sqrt(rs_old) << std::endl;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        for (int i = 0; i < n; i++) {
            Ap[i] = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                Ap[i] += values[j] * p[col_idx[j]];
            }
        }

        // alpha = rs_old / (p' * Ap)
        double pAp = 0.0;
        for (int i = 0; i < n; i++) {
            pAp += p[i] * Ap[i];
        }
        double alpha = rs_old / pAp;

        // x = x + alpha * p
        // r = r - alpha * Ap
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // rs_new = r' * r
        double rs_new = 0.0;
        for (int i = 0; i < n; i++) {
            rs_new += r[i] * r[i];
        }

        // Check convergence
        if (std::sqrt(rs_new) / b_norm < tol) {
            iter++;
            break;
        }

        // p = r + (rs_new / rs_old) * p
        double beta = rs_new / rs_old;
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rs_old = rs_new;
    }

    double final_residual = compute_relative_residual(values.data(), row_ptr.data(), col_idx.data(),
                                                      n, x.data(), b.data());

    std::cout << "CG converged in " << iter << " iterations" << std::endl;
    std::cout << "Final relative residual: " << final_residual << std::endl;

    EXPECT_LT(final_residual, tol) << "CG should converge to tolerance";
    EXPECT_LT(iter, max_iter) << "CG should converge before max iterations";

    // Check solution is close to true solution
    double error = 0.0;
    for (int i = 0; i < n; i++) {
        error += (x[i] - x_true[i]) * (x[i] - x_true[i]);
    }
    error = std::sqrt(error) / std::sqrt(n);

    std::cout << "Solution error (L2 normalized): " << error << std::endl;
    EXPECT_LT(error, 1e-5) << "Solution should be close to true solution";

    free(mat.entries);
}
