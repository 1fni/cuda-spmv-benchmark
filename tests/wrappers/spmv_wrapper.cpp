#include "spmv_wrapper.hpp"
#include <iostream>
#include <cstring>

// SpMVWrapper Implementation
SpMVWrapper::SpMVWrapper(const std::string& operator_name) 
    : operator_(nullptr), initialized_(false), matrix_rows_(0), operator_name_(operator_name) {
    
    // Get operator from C API
    operator_ = get_operator(operator_name.c_str());
    if (!operator_) {
        throw std::invalid_argument("SpMV operator '" + operator_name + "' not found");
    }
}

SpMVWrapper::~SpMVWrapper() {
    cleanup();
}

SpMVWrapper::SpMVWrapper(SpMVWrapper&& other) noexcept
    : operator_(other.operator_), initialized_(other.initialized_), 
      matrix_rows_(other.matrix_rows_), operator_name_(std::move(other.operator_name_)) {
    
    // Transfer ownership
    other.operator_ = nullptr;
    other.initialized_ = false; 
    other.matrix_rows_ = 0;
}

SpMVWrapper& SpMVWrapper::operator=(SpMVWrapper&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        operator_ = other.operator_;
        initialized_ = other.initialized_;
        matrix_rows_ = other.matrix_rows_;
        operator_name_ = std::move(other.operator_name_);
        
        other.operator_ = nullptr;
        other.initialized_ = false;
        other.matrix_rows_ = 0;
    }
    return *this;
}

bool SpMVWrapper::init(MatrixData* matrix_data) {
    if (!operator_) {
        throw std::runtime_error("No valid operator available for initialization");
    }
    
    if (!matrix_data) {
        throw std::runtime_error("Matrix data is null");
    }
    
    // Call C initialization function
    int result = operator_->init(matrix_data);
    if (result != 0) {
        throw std::runtime_error("SpMV operator initialization failed with code: " + std::to_string(result));
    }
    
    initialized_ = true;
    matrix_rows_ = matrix_data->rows;
    
    return true;
}

std::vector<double> SpMVWrapper::multiply(const std::vector<double>& x) {
    if (!initialized_) {
        throw std::runtime_error("SpMV operator not initialized");
    }
    
    if (static_cast<int>(x.size()) != matrix_rows_) {
        throw std::runtime_error("Input vector size (" + std::to_string(x.size()) + 
                               ") does not match matrix rows (" + std::to_string(matrix_rows_) + ")");
    }
    
    // Allocate output vector
    std::vector<double> y(matrix_rows_);
    
    // Call C run function
    int result = operator_->run(x.data(), y.data());
    if (result != 0) {
        throw std::runtime_error("SpMV operation failed with code: " + std::to_string(result));
    }
    
    return y;
}

void SpMVWrapper::multiply(const std::vector<double>& x, std::vector<double>& y) {
    if (!initialized_) {
        throw std::runtime_error("SpMV operator not initialized");
    }
    
    if (static_cast<int>(x.size()) != matrix_rows_) {
        throw std::runtime_error("Input vector size (" + std::to_string(x.size()) + 
                               ") does not match matrix rows (" + std::to_string(matrix_rows_) + ")");
    }
    
    if (static_cast<int>(y.size()) != matrix_rows_) {
        throw std::runtime_error("Output vector size (" + std::to_string(y.size()) + 
                               ") does not match matrix rows (" + std::to_string(matrix_rows_) + ")");
    }
    
    // Call C run function  
    int result = operator_->run(x.data(), y.data());
    if (result != 0) {
        throw std::runtime_error("SpMV operation failed with code: " + std::to_string(result));
    }
}

void SpMVWrapper::cleanup() noexcept {
    if (initialized_ && operator_ && operator_->free) {
        try {
            operator_->free();
        } catch (...) {
            // Ignore exceptions in cleanup
        }
        initialized_ = false;
    }
}

// MatrixDataWrapper Implementation
MatrixDataWrapper::MatrixDataWrapper(const std::string& filename) 
    : data_(std::make_unique<MatrixData>()) {
    
    // Initialize structure
    std::memset(data_.get(), 0, sizeof(MatrixData));
    
    // Load from file using C API
    int result = load_matrix_market(filename.c_str(), data_.get());
    if (result != 0) {
        throw std::runtime_error("Failed to load matrix from file: " + filename + 
                               " (error code: " + std::to_string(result) + ")");
    }
}

MatrixDataWrapper::MatrixDataWrapper(int rows, int cols, const std::vector<Entry>& entries, int grid_size) 
    : data_(std::make_unique<MatrixData>()) {
    
    data_->rows = rows;
    data_->cols = cols;
    data_->nnz = static_cast<int>(entries.size());
    data_->grid_size = grid_size;
    
    // Allocate and copy entries
    data_->entries = static_cast<Entry*>(malloc(entries.size() * sizeof(Entry)));
    if (!data_->entries && !entries.empty()) {
        throw std::runtime_error("Failed to allocate memory for matrix entries");
    }
    
    std::memcpy(data_->entries, entries.data(), entries.size() * sizeof(Entry));
}

MatrixDataWrapper::~MatrixDataWrapper() {
    if (data_ && data_->entries) {
        free(data_->entries);
        data_->entries = nullptr;
    }
}