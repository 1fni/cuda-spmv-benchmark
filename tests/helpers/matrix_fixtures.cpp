#include "matrix_fixtures.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <set>

namespace MatrixFixtures {

// ================================
// Helper Functions
// ================================

/**
 * @brief Create MatrixDataWrapper from entries list
 */
static std::unique_ptr<MatrixDataWrapper> 
create_from_entries(int rows, int cols, const std::vector<Entry>& entries, int grid_size = -1) {
    return std::make_unique<MatrixDataWrapper>(rows, cols, entries, grid_size);
}

// ================================
// Simple Analytical Matrices
// ================================

std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult> 
identity_3x3() {
    std::vector<Entry> entries = {
        {0, 0, 1.0},  // (1,1) = 1
        {1, 1, 1.0},  // (2,2) = 1  
        {2, 2, 1.0}   // (3,3) = 1
    };
    
    auto matrix = create_from_entries(3, 3, entries);
    
    // For input vector [1, 1, 1], result = [1, 1, 1], checksum = 3, L2 norm = sqrt(3)
    AnalyticalResult result(3.0, std::sqrt(3.0), "3x3 identity matrix");
    
    return {std::move(matrix), result};
}

std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
diagonal_5x5() {
    std::vector<Entry> entries = {
        {0, 0, 1.0},  // (1,1) = 1
        {1, 1, 2.0},  // (2,2) = 2
        {2, 2, 3.0},  // (3,3) = 3
        {3, 3, 4.0},  // (4,4) = 4
        {4, 4, 5.0}   // (5,5) = 5
    };
    
    auto matrix = create_from_entries(5, 5, entries);
    
    // For input vector [1, 1, 1, 1, 1], result = [1, 2, 3, 4, 5]
    // checksum = 15, L2 norm = sqrt(1 + 4 + 9 + 16 + 25) = sqrt(55)
    AnalyticalResult result(15.0, std::sqrt(55.0), "5x5 diagonal matrix [1,2,3,4,5]");
    
    return {std::move(matrix), result};
}

std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
tridiagonal_4x4() {
    // Tridiagonal matrix:
    //  2 -1  0  0
    // -1  2 -1  0  
    //  0 -1  2 -1
    //  0  0 -1  2
    std::vector<Entry> entries = {
        {0, 0,  2.0}, {0, 1, -1.0},                    // Row 1: 2, -1
        {1, 0, -1.0}, {1, 1,  2.0}, {1, 2, -1.0},     // Row 2: -1, 2, -1
        {2, 1, -1.0}, {2, 2,  2.0}, {2, 3, -1.0},     // Row 3: -1, 2, -1  
        {3, 2, -1.0}, {3, 3,  2.0}                     // Row 4: -1, 2
    };
    
    auto matrix = create_from_entries(4, 4, entries);
    
    // For input [1, 1, 1, 1]:
    // Row 1: 2*1 + (-1)*1 = 1
    // Row 2: (-1)*1 + 2*1 + (-1)*1 = 0  
    // Row 3: (-1)*1 + 2*1 + (-1)*1 = 0
    // Row 4: (-1)*1 + 2*1 = 1
    // Result = [1, 0, 0, 1], checksum = 2, L2 norm = sqrt(2)
    AnalyticalResult result(2.0, std::sqrt(2.0), "4x4 tridiagonal matrix");
    
    return {std::move(matrix), result};
}

std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
upper_triangular_3x3() {
    // Upper triangular:
    // 1  2  3
    // 0  4  5
    // 0  0  6
    std::vector<Entry> entries = {
        {0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 3.0},  // Row 1: 1, 2, 3
        {1, 1, 4.0}, {1, 2, 5.0},                // Row 2: 0, 4, 5
        {2, 2, 6.0}                              // Row 3: 0, 0, 6
    };
    
    auto matrix = create_from_entries(3, 3, entries);
    
    // For input [1, 1, 1]:
    // Row 1: 1*1 + 2*1 + 3*1 = 6
    // Row 2: 0*1 + 4*1 + 5*1 = 9
    // Row 3: 0*1 + 0*1 + 6*1 = 6
    // Result = [6, 9, 6], checksum = 21, L2 norm = sqrt(36 + 81 + 36) = sqrt(153)
    AnalyticalResult result(21.0, std::sqrt(153.0), "3x3 upper triangular matrix");
    
    return {std::move(matrix), result};
}

// ================================
// Stencil Pattern Matrices
// ================================

std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
stencil_5point(int grid_size) {
    std::vector<Entry> entries;
    int n = grid_size * grid_size;  // Total number of grid points
    
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            int idx = i * grid_size + j;  // Current grid point index
            
            // Center coefficient (diagonal)
            entries.push_back({idx, idx, -4.0});
            
            // North neighbor (i-1, j)
            if (i > 0) {
                int north_idx = (i-1) * grid_size + j;
                entries.push_back({idx, north_idx, 1.0});
            }
            
            // South neighbor (i+1, j)  
            if (i < grid_size - 1) {
                int south_idx = (i+1) * grid_size + j;
                entries.push_back({idx, south_idx, 1.0});
            }
            
            // West neighbor (i, j-1)
            if (j > 0) {
                int west_idx = i * grid_size + (j-1);
                entries.push_back({idx, west_idx, 1.0});
            }
            
            // East neighbor (i, j+1)
            if (j < grid_size - 1) {
                int east_idx = i * grid_size + (j+1);
                entries.push_back({idx, east_idx, 1.0});
            }
        }
    }
    
    auto matrix = create_from_entries(n, n, entries, grid_size);
    
    // For input vector of ones, compute expected checksum analytically
    double expected_checksum = 0.0;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            // Count neighbors for this point
            int num_neighbors = 0;
            if (i > 0) num_neighbors++;              // North
            if (i < grid_size - 1) num_neighbors++;  // South
            if (j > 0) num_neighbors++;              // West  
            if (j < grid_size - 1) num_neighbors++;  // East
            
            // Contribution: -4*1 + num_neighbors*1 = num_neighbors - 4
            expected_checksum += (num_neighbors - 4);
        }
    }
    
    std::string description = std::to_string(grid_size) + "x" + std::to_string(grid_size) + 
                             " 5-point stencil matrix";
    AnalyticalResult result(expected_checksum, 0.0, description);  // L2 norm not computed
    
    return {std::move(matrix), result};
}

std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
stencil_9point(int grid_size) {
    std::vector<Entry> entries;
    int n = grid_size * grid_size;
    
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            int idx = i * grid_size + j;
            
            // Center coefficient
            entries.push_back({idx, idx, -8.0});
            
            // 4-connected neighbors (weight 2)
            const int di4[] = {-1, 1, 0, 0};
            const int dj4[] = {0, 0, -1, 1};
            
            for (int k = 0; k < 4; ++k) {
                int ni = i + di4[k];
                int nj = j + dj4[k];
                if (ni >= 0 && ni < grid_size && nj >= 0 && nj < grid_size) {
                    int neighbor_idx = ni * grid_size + nj;
                    entries.push_back({idx, neighbor_idx, 2.0});
                }
            }
            
            // 8-connected neighbors (weight 1)  
            const int di8[] = {-1, -1, 1, 1};
            const int dj8[] = {-1, 1, -1, 1};
            
            for (int k = 0; k < 4; ++k) {
                int ni = i + di8[k];
                int nj = j + dj8[k];
                if (ni >= 0 && ni < grid_size && nj >= 0 && nj < grid_size) {
                    int neighbor_idx = ni * grid_size + nj;
                    entries.push_back({idx, neighbor_idx, 1.0});
                }
            }
        }
    }
    
    auto matrix = create_from_entries(n, n, entries, grid_size);
    
    std::string description = std::to_string(grid_size) + "x" + std::to_string(grid_size) + 
                             " 9-point stencil matrix";
    AnalyticalResult result(0.0, 0.0, description);  // Checksum requires detailed calculation
    
    return {std::move(matrix), result};
}

// ================================
// Structured Test Matrices  
// ================================

std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
banded_matrix(int size, int bandwidth, double diagonal_value) {
    std::vector<Entry> entries;
    int half_band = bandwidth / 2;
    
    for (int i = 0; i < size; ++i) {
        for (int j = std::max(0, i - half_band); j <= std::min(size - 1, i + half_band); ++j) {
            entries.push_back({i, j, diagonal_value});
        }
    }
    
    auto matrix = create_from_entries(size, size, entries);
    
    // Expected checksum for ones vector
    double expected_checksum = 0.0;
    for (int i = 0; i < size; ++i) {
        int row_nnz = std::min(size - 1, i + half_band) - std::max(0, i - half_band) + 1;
        expected_checksum += row_nnz * diagonal_value;
    }
    
    std::string description = std::to_string(size) + "x" + std::to_string(size) + 
                             " banded matrix (bandwidth=" + std::to_string(bandwidth) + ")";
    AnalyticalResult result(expected_checksum, 0.0, description);
    
    return {std::move(matrix), result};
}

std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
dense_block_matrix(int size, int block_size, double fill_value) {
    std::vector<Entry> entries;
    
    // Create dense blocks along the diagonal
    for (int block_start = 0; block_start < size; block_start += block_size) {
        int block_end = std::min(block_start + block_size, size);
        
        for (int i = block_start; i < block_end; ++i) {
            for (int j = block_start; j < block_end; ++j) {
                entries.push_back({i, j, fill_value});
            }
        }
    }
    
    auto matrix = create_from_entries(size, size, entries);
    
    // Expected checksum
    int num_full_blocks = size / block_size;
    int remainder_block_size = size % block_size;
    double expected_checksum = num_full_blocks * block_size * block_size * fill_value;
    if (remainder_block_size > 0) {
        expected_checksum += remainder_block_size * remainder_block_size * fill_value;
    }
    
    std::string description = std::to_string(size) + "x" + std::to_string(size) + 
                             " dense block matrix (block_size=" + std::to_string(block_size) + ")";
    AnalyticalResult result(expected_checksum, 0.0, description);
    
    return {std::move(matrix), result};
}

// ================================
// Random and Stress Test Matrices
// ================================

std::unique_ptr<MatrixDataWrapper>
random_sparse(int size, double sparsity_ratio, unsigned int seed,
              std::pair<double, double> value_range) {
    std::vector<Entry> entries;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> value_dist(value_range.first, value_range.second);
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (prob_dist(gen) < sparsity_ratio) {
                double value = value_dist(gen);
                if (std::abs(value) > 1e-15) {  // Avoid tiny values
                    entries.push_back({i, j, value});
                }
            }
        }
    }
    
    return create_from_entries(size, size, entries);
}

std::unique_ptr<MatrixDataWrapper>
ill_conditioned_matrix(int size) {
    std::vector<Entry> entries;
    
    // Create matrix with exponentially decreasing diagonal values
    for (int i = 0; i < size; ++i) {
        double diagonal_value = std::pow(10.0, -i);  // 1, 0.1, 0.01, ...
        entries.push_back({i, i, diagonal_value});
        
        // Add some off-diagonal entries to make it more interesting
        if (i > 0) {
            entries.push_back({i, i-1, 0.1 * diagonal_value});
        }
        if (i < size - 1) {
            entries.push_back({i, i+1, 0.1 * diagonal_value});
        }
    }
    
    return create_from_entries(size, size, entries);
}

std::unique_ptr<MatrixDataWrapper>
unbalanced_rows_matrix(int size, unsigned int seed) {
    std::vector<Entry> entries;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> value_dist(-1.0, 1.0);
    
    for (int i = 0; i < size; ++i) {
        // Vary row lengths: some rows dense, some sparse
        int row_nnz;
        if (i < size / 10) {
            row_nnz = std::min(size, size / 2 + i * 2);  // Dense rows
        } else if (i < size / 2) {
            row_nnz = 3 + (i % 5);                       // Medium rows
        } else {
            row_nnz = 1 + (i % 3);                       // Sparse rows
        }
        
        // Select random column indices for this row
        std::set<int> selected_cols;
        std::uniform_int_distribution<int> col_dist(0, size - 1);
        
        while (static_cast<int>(selected_cols.size()) < row_nnz) {
            selected_cols.insert(col_dist(gen));
        }
        
        for (int col : selected_cols) {
            entries.push_back({i, col, value_dist(gen)});
        }
    }
    
    return create_from_entries(size, size, entries);
}

// ================================
// Matrix Property Analysis
// ================================

MatrixProperties analyze_matrix(const MatrixDataWrapper& matrix) {
    MatrixProperties props;
    
    props.rows = matrix->rows;
    props.cols = matrix->cols;
    props.nnz = matrix->nnz;
    props.is_square = (props.rows == props.cols);
    props.density = static_cast<double>(props.nnz) / (static_cast<double>(props.rows) * props.cols);
    props.avg_nnz_per_row = static_cast<double>(props.nnz) / props.rows;
    
    // Analyze row distribution
    std::vector<int> row_nnz(props.rows, 0);
    for (int k = 0; k < props.nnz; ++k) {
        row_nnz[matrix->entries[k].row]++;
    }
    
    props.max_nnz_per_row = *std::max_element(row_nnz.begin(), row_nnz.end());
    props.min_nnz_per_row = *std::min_element(row_nnz.begin(), row_nnz.end());
    
    // Simple symmetry check (approximate)
    props.is_symmetric = props.is_square;  // Simplified check
    if (props.is_square && props.nnz > 0) {
        // More thorough check would require sorting entries by (i,j) and (j,i)
        // For now, just mark as potentially symmetric if square
    }
    
    // Classify sparsity pattern
    if (props.density > 0.5) {
        props.sparsity_pattern = "Dense";
    } else if (props.density > 0.1) {
        props.sparsity_pattern = "Moderate";  
    } else if (props.density > 0.01) {
        props.sparsity_pattern = "Sparse";
    } else {
        props.sparsity_pattern = "Very sparse";
    }
    
    // Add pattern hints based on grid_size
    if (matrix->grid_size > 0) {
        props.sparsity_pattern += " (stencil pattern)";
    }
    
    return props;
}

void print_matrix_properties(const MatrixProperties& props, 
                            const std::string& matrix_name) {
    std::cout << "\n=== Matrix Properties: " << matrix_name << " ===" << std::endl;
    std::cout << "Dimensions: " << props.rows << " x " << props.cols;
    if (props.is_square) std::cout << " (square)";
    std::cout << std::endl;
    
    std::cout << "Non-zeros: " << props.nnz << std::endl;
    std::cout << "Density: " << std::fixed << std::setprecision(6) << props.density 
              << " (" << std::setprecision(2) << (props.density * 100) << "%)" << std::endl;
    std::cout << "Sparsity pattern: " << props.sparsity_pattern << std::endl;
    
    std::cout << "Row distribution:" << std::endl;
    std::cout << "  Average nnz per row: " << std::setprecision(2) << props.avg_nnz_per_row << std::endl;
    std::cout << "  Min nnz per row: " << props.min_nnz_per_row << std::endl;
    std::cout << "  Max nnz per row: " << props.max_nnz_per_row << std::endl;
    
    if (props.is_symmetric) {
        std::cout << "Structure: Potentially symmetric" << std::endl;
    }
    std::cout << std::endl;
}

// ================================
// Utility Functions
// ================================

std::unique_ptr<MatrixDataWrapper>
from_csr_arrays(int rows, int cols, 
               const int* row_ptr, const int* col_ind, const double* values,
               int nnz) {
    std::vector<Entry> entries;
    entries.reserve(nnz);
    
    for (int i = 0; i < rows; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            entries.push_back({i, col_ind[k], values[k]});
        }
    }
    
    return create_from_entries(rows, cols, entries);
}

bool save_to_file(const MatrixDataWrapper& matrix, const std::string& filename) {
    // This would require implementing Matrix Market writer
    // For now, just indicate success/failure
    return false;  // Not implemented
}

} // namespace MatrixFixtures