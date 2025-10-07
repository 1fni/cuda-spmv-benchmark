# API Documentation

Complete reference for the CUDA SpMV Benchmark Suite API.

## Core SpMV Interface

### SpmvOperator Structure

```c
typedef struct {
    const char* name;
    int (*init)(MatrixData* mat);
    double (*run_timed)(const double* x, double* y);
    void (*free)();
} SpmvOperator;
```

**Description**: Standardized interface for all SpMV implementations.

**Fields**:
- `name`: Human-readable operator identifier
- `init`: Initialize GPU memory and transfer matrix data
- `run_timed`: Execute SpMV operation and return kernel time in microseconds
- `free`: Release all allocated GPU resources

### Operator Selection

```c
SpmvOperator* get_operator(const char* name);
```

**Parameters**: 
- `name`: Operator identifier ("csr", "stencil5", "ellpack")

**Returns**: Pointer to operator structure, or NULL if not found

**Available Operators**:
- `"csr"`: CSR format using cuSPARSE
- `"stencil5"`: 5-point stencil with custom CUDA kernels using ELLPACK storage
- `"ellpack"`: ELLPACK format using cuSPARSE

### Usage Pattern

```c
// Standard usage flow
SpmvOperator* op = get_operator("stencil5");
if (op->init(&matrix_data) != 0) {
    // Handle initialization error
}

double kernel_time = op->run_timed(x_vector, y_vector);
op->free();
```

## Matrix I/O API

### MatrixData Structure

```c
typedef struct {
    int rows, cols, nnz;
    int* row_ptr;
    int* col_ind; 
    double* values;
} MatrixData;
```

**Description**: Generic sparse matrix storage in CSR format.

### File Operations

```c
int load_matrix_market(const char* filename, MatrixData* matrix);
int save_matrix_market(const char* filename, const MatrixData* matrix);
void free_matrix_data(MatrixData* matrix);
```

**load_matrix_market**:
- Reads Matrix Market (.mtx) files
- Automatically expands symmetric matrices to general format
- Returns 0 on success, non-zero on error

**save_matrix_market**:
- Writes matrix in Matrix Market format
- Returns 0 on success, non-zero on error

**free_matrix_data**:
- Releases all allocated memory for matrix structure

## Performance Metrics API

### BenchmarkMetrics Structure

```c
typedef struct {
    double execution_time_us;
    double gflops;
    double memory_bandwidth_gb_s;
    double arithmetic_intensity;
    const char* bottleneck_analysis;
    const char* optimization_recommendation;
} BenchmarkMetrics;
```

### Metrics Calculation

```c
BenchmarkMetrics calculate_metrics(const char* operator_name, 
                                  double execution_time_us,
                                  int nnz, int rows, int cols);
```

**Parameters**:
- `operator_name`: SpMV operator used ("csr", "stencil5", "ellpack")
- `execution_time_us`: Kernel execution time in microseconds
- `nnz`, `rows`, `cols`: Matrix dimensions

**Returns**: Complete performance analysis with GFLOPS, bandwidth, and recommendations

### Output Formatting

```c
void print_metrics(const BenchmarkMetrics* metrics, const char* operator_name);
void export_metrics_json(const BenchmarkMetrics* metrics, const char* filename);
void export_metrics_csv(const BenchmarkMetrics* metrics, const char* filename);
```

**print_metrics**: Human-readable console output
**export_metrics_json**: Structured JSON export for automation
**export_metrics_csv**: CSV format for spreadsheet analysis

## Error Handling

### CUDA Error Checking

```c
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

### cuSPARSE Error Checking

```c
#define CHECK_CUSPARSE(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

### Return Codes

- `0`: Success
- `1`: File I/O error
- `2`: Memory allocation failure
- `3`: CUDA/cuSPARSE error
- `4`: Invalid parameter

## Matrix Generation API

### Stencil Matrix Generator

```c
int generate_stencil_matrix(int grid_size, const char* output_filename);
```

**Parameters**:
- `grid_size`: Grid dimension (creates grid_size² × grid_size² matrix)
- `output_filename`: Output Matrix Market file path

**Matrix Pattern**:
- Center coefficient: 4.0
- North/South/East/West neighbors: -1.0
- Boundary handling: Natural boundary conditions

**Returns**: 0 on success, non-zero on error

## Advanced Testing API

### C++ Wrapper Interface

```cpp
namespace SpMVWrappers {
    class SpmvOperatorWrapper {
    public:
        SpmvOperatorWrapper(const std::string& operator_name);
        ~SpmvOperatorWrapper();
        
        bool initialize(const MatrixDataWrapper& matrix);
        double execute(const std::vector<double>& x_input, 
                      std::vector<double>& y_output);
        void cleanup();
        
        bool is_initialized() const;
        std::string get_operator_name() const;
    };
}
```

### Matrix Fixtures

```cpp
namespace MatrixFixtures {
    // Simple analytical matrices with known results
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult> 
    identity_3x3();
    
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    tridiagonal_4x4();
    
    // Stencil pattern generators
    std::pair<std::unique_ptr<MatrixDataWrapper>, AnalyticalResult>
    stencil_5point(int grid_size);
    
    // Random matrices for stress testing
    std::unique_ptr<MatrixDataWrapper>
    random_sparse(int size, double sparsity_ratio, unsigned int seed = 42);
}
```

### Performance Benchmarking

```cpp
namespace PerformanceBenchmarks {
    struct BenchmarkResult {
        std::string operator_name;
        double kernel_time_ms;
        double effective_gflops;
        double memory_bandwidth_gb_s;
        double arithmetic_intensity;
        bool correctness_passed;
        // ... additional metrics
    };
    
    BenchmarkResult benchmark_operator(const std::string& operator_name,
                                      const MatrixDataWrapper& matrix,
                                      const BenchmarkConfig& config = BenchmarkConfig());
    
    std::map<std::string, BenchmarkResult> 
    compare_operators(const std::vector<std::string>& operator_names,
                     const MatrixDataWrapper& matrix);
}
```

## Build Integration

### Makefile Targets

```bash
make                    # Build release version
make BUILD_TYPE=debug   # Build with debug symbols
make clean             # Clean build artifacts
make run               # Quick test run
```

### CMake Integration (Testing)

```bash
cd tests && mkdir build && cd build
cmake ..
make
./test_runner
```

**Test Categories**:
- `WrapperTest*`: Basic wrapper functionality
- `MatrixFixtureTest*`: Matrix generation and properties  
- `PerformanceTest*`: Benchmarking and metrics validation

## Memory Management

### Global Storage Structures

```c
extern CSRMatrix csr_mat;           // Global CSR matrix storage
extern ELLPACKMatrix ellpack_matrix; // Global ELLPACK matrix storage
```

**Important**: These global structures are shared across operators and automatically managed by init/free functions.

### Memory Layout by Format

- **CSR**: row_ptr, col_ind, values arrays on GPU
- **ELLPACK**: indices, data arrays with row padding for cuSPARSE
- **STENCIL5**: Uses ELLPACK storage format with specialized custom kernels

## Integration Examples

### Basic Benchmarking

```c
#include "spmv.h"

int main() {
    MatrixData matrix;
    load_matrix_market("matrix.mtx", &matrix);
    
    SpmvOperator* op = get_operator("stencil5");
    op->init(&matrix);
    
    double* x = (double*)malloc(matrix.cols * sizeof(double));
    double* y = (double*)malloc(matrix.rows * sizeof(double));
    
    // Initialize input vector
    for (int i = 0; i < matrix.cols; i++) x[i] = 1.0;
    
    double kernel_time = op->run_timed(x, y);
    BenchmarkMetrics metrics = calculate_metrics("stencil5", kernel_time, 
                                                matrix.nnz, matrix.rows, matrix.cols);
    print_metrics(&metrics, "stencil5");
    
    op->free();
    free_matrix_data(&matrix);
    free(x); free(y);
    return 0;
}
```

### Automated Comparison

```bash
#!/bin/bash
# Compare all operators on same matrix
for op in csr stencil5 ellpack; do
    ./bin/spmv_bench matrix.mtx --mode=$op --output-format=json --output-file=results_$op.json
done

# Analyze results
jq -s '.[].benchmark.performance.gflops' results_*.json
```

### CI/CD Integration

```yaml
# GitHub Actions performance validation
- name: Performance regression check
  run: |
    ./bin/spmv_bench matrix/test.mtx --mode=stencil5 --output-format=json | \
    jq '.benchmark.performance.gflops' | \
    awk '{if($1 < 5.0) {print "Performance regression detected"; exit 1}}'
```

This API provides complete programmatic access to the SpMV benchmark suite for integration into larger HPC workflows, automated testing pipelines, and performance analysis systems.