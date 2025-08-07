/**
 * @file spmv_wrapper.hpp  
 * @brief C++ wrapper for C SpMV operators to enable Google Test integration
 *
 * This wrapper provides a modern C++ interface around the existing C SpMV operators
 * while preserving the original C implementation unchanged. It enables RAII resource 
 * management and STL container integration for testing purposes.
 */

#ifndef SPMV_WRAPPER_HPP
#define SPMV_WRAPPER_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

// C headers
extern "C" {
    #include "spmv.h"
    #include "io.h"
}

/**
 * @class SpMVWrapper
 * @brief C++ wrapper for C SpMV operators with RAII and STL integration
 *
 * Usage example:
 *   SpMVWrapper spmv("SPMV_CSR");
 *   spmv.init(matrix_data); 
 *   auto result = spmv.multiply(input_vector);
 */
class SpMVWrapper {
private:
    SpmvOperator* operator_;          ///< Pointer to C operator
    bool initialized_;                ///< Initialization state
    int matrix_rows_;                 ///< Number of matrix rows
    std::string operator_name_;       ///< Operator name for debugging
    
public:
    /**
     * @brief Construct wrapper for specified SpMV operator
     * @param operator_name Name of operator ("SPMV_CSR", "SPMV_STENCIL5", "SPMV_ELLPACK")
     * @throws std::invalid_argument if operator not found
     */
    explicit SpMVWrapper(const std::string& operator_name);
    
    /**
     * @brief Destructor with automatic cleanup
     */
    ~SpMVWrapper();
    
    // Disable copy constructor and assignment (resource management)
    SpMVWrapper(const SpMVWrapper&) = delete;
    SpMVWrapper& operator=(const SpMVWrapper&) = delete;
    
    // Enable move constructor and assignment  
    SpMVWrapper(SpMVWrapper&& other) noexcept;
    SpMVWrapper& operator=(SpMVWrapper&& other) noexcept;
    
    /**
     * @brief Initialize operator with matrix data
     * @param matrix_data Pointer to MatrixData structure
     * @return true if initialization successful
     * @throws std::runtime_error if initialization fails
     */
    bool init(MatrixData* matrix_data);
    
    /**
     * @brief Execute SpMV operation: y = A * x  
     * @param x Input vector
     * @return Output vector y
     * @throws std::runtime_error if not initialized or operation fails
     */
    std::vector<double> multiply(const std::vector<double>& x);
    
    /**
     * @brief Execute SpMV with preallocated output vector
     * @param x Input vector
     * @param y Output vector (must be correctly sized)
     * @throws std::runtime_error if vectors have wrong size or operation fails
     */
    void multiply(const std::vector<double>& x, std::vector<double>& y);
    
    /**
     * @brief Check if operator is initialized
     * @return true if ready for multiply operations
     */
    bool is_initialized() const { return initialized_; }
    
    /**
     * @brief Get operator name
     * @return Operator name string
     */
    const std::string& get_name() const { return operator_name_; }
    
    /**
     * @brief Get matrix dimensions after initialization
     * @return Number of matrix rows (0 if not initialized)
     */
    int get_matrix_rows() const { return matrix_rows_; }
    
private:
    /**
     * @brief Clean up resources without throwing exceptions
     */
    void cleanup() noexcept;
};

/**
 * @class MatrixDataWrapper  
 * @brief RAII wrapper for MatrixData with automatic memory management
 */
class MatrixDataWrapper {
private:
    std::unique_ptr<MatrixData> data_;
    
public:
    /**
     * @brief Load matrix from file with automatic memory management
     * @param filename Path to Matrix Market (.mtx) file
     * @throws std::runtime_error if file loading fails
     */
    explicit MatrixDataWrapper(const std::string& filename);
    
    /**
     * @brief Create matrix from raw data
     * @param rows Number of rows
     * @param cols Number of columns  
     * @param entries Vector of matrix entries
     * @param grid_size Grid size for stencil matrices (-1 if not stencil)
     */
    MatrixDataWrapper(int rows, int cols, const std::vector<Entry>& entries, int grid_size = -1);
    
    /**
     * @brief Destructor with automatic cleanup
     */
    ~MatrixDataWrapper();
    
    // Disable copy, enable move
    MatrixDataWrapper(const MatrixDataWrapper&) = delete;
    MatrixDataWrapper& operator=(const MatrixDataWrapper&) = delete;
    MatrixDataWrapper(MatrixDataWrapper&&) = default;
    MatrixDataWrapper& operator=(MatrixDataWrapper&&) = default;
    
    /**
     * @brief Get pointer to underlying MatrixData (for C API)
     * @return Pointer to MatrixData structure
     */
    MatrixData* get() const { return data_.get(); }
    
    /**
     * @brief Access MatrixData via arrow operator
     * @return Pointer to MatrixData structure  
     */
    MatrixData* operator->() const { return data_.get(); }
    
    /**
     * @brief Access MatrixData via dereference operator
     * @return Reference to MatrixData structure
     */
    MatrixData& operator*() const { return *data_.get(); }
};

#endif // SPMV_WRAPPER_HPP