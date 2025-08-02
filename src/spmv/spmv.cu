/**
 * @file spmv.cu
 * @brief Provides global SpMV operator instances for different implementations (CSR, ELLPACK, stencil).
 *
 * @details
 * Responsibilities:
 *  - Expose SpmvOperator objects that bind names to their respective implementations.
 *  - Allow main program to select an operator based on user input (e.g., "csr", "ellpack", "stencil5").
 *
 * Example:
 *  extern SpmvOperator SPMV_CSR;
 *  extern SpmvOperator SPMV_ELL;
 *  extern SpmvOperator SPMV_STENCIL;
 *
 *  // In main:
 *  SpmvOperator* op = &SPMV_CSR;
 *  op->init(matrix_path);
 *  op->run(x, y);
 *  op->free();
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#include "spmv.h"

SpmvOperator* get_operator(const char* mode) {
    if (strcmp(mode, "csr") == 0) return &SPMV_CSR; //cusparse csr
    if (strcmp(mode, "stencil5") == 0) return &SPMV_STENCIL5; //kernel ellpack optimized for stencil
    if (strcmp(mode, "ellpack") == 0) return &SPMV_ELLPACK; //cusparse using csr struct to build ellpack
    return NULL;
}

