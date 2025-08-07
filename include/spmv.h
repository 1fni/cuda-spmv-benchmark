/**
 * @file spmv.h
 * @brief Central header for Sparse Matrix-Vector Multiplication (SpMV) operators and shared global structures.
 *
 * @details
 * Responsibilities:
 *  - Declare global variables for CSR and ELLPACK structures used across multiple SpMV implementations.
 *  - Provide helper macros for CUDA and cuSPARSE error handling.
 *  - Define the SpmvOperator structure, which acts as a function dispatch table for different SpMV implementations.
 *
 * Key Components:
 *  - `CSRMatrix` and `ELLPACKMatrix` global instances for storing matrix formats.
 *  - Function pointer-based `SpmvOperator` for modular algorithm selection.
 *
 * Author: Bouhrour Stephane  
 * Date: 2025-07-15
 */

#ifndef SPMV_H
#define SPMV_H

#include "spmv_csr.h"
#include "spmv_ellpack.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Global CSR and ELLPACK matrix structures used by operators. */
extern CSRMatrix csr_mat;
extern ELLPACKMatrix ellpack_matrix;

#ifdef __cplusplus
}
#endif

/** @brief CUDA error checking macro. */
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

/** @brief cuSPARSE error checking macro. */
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

/**
 * @struct SpmvOperator
 * @brief Defines the interface for different SpMV implementations.
 *
 * Each operator provides:
 *  - `name`: Identifier of the method (e.g., CSR, ELLPACK, stencil).
 *  - `init`: Function pointer for initialization with a given matrix.
 *  - `run`: Function pointer to execute SpMV with input/output vectors.
 *  - `free`: Function pointer to release associated resources.
 */
typedef struct {
    const char* name;                                      ///< Operator name (e.g., "csr", "ellpack").
    int (*init)(MatrixData* mat);                          ///< Initialization function for matrix data.
    int (*run)(const double* x, double* y);               ///< SpMV computation function.
    void (*free)();                                        ///< Resource cleanup function.
} SpmvOperator;

/** @brief External operator declarations for different formats. */
extern SpmvOperator SPMV_CSR;
extern SpmvOperator SPMV_STENCIL5;
extern SpmvOperator SPMV_ELLPACK;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Retrieves the appropriate SpMV operator by name.
 * @param mode Name of the operator (e.g., "csr", "ellpack").
 * @return Pointer to the matching SpmvOperator, or NULL if not found.
 */
SpmvOperator* get_operator(const char* mode);

#ifdef __cplusplus
}
#endif

#endif // SPMV_H
