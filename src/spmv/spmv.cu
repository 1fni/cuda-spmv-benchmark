/**
 * @file spmv.cu
 * @brief SpMV operator dispatch
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#include "spmv.h"

SpmvOperator* get_operator(const char* mode) {
    if (strcmp(mode, "csr-cusparse") == 0) return &SPMV_CSR;
    if (strcmp(mode, "stencil5-csr") == 0) return &SPMV_STENCIL5_CSR;
#ifdef __has_include
  #if __has_include(<mpi.h>)
    if (strcmp(mode, "stencil5-halo-mgpu") == 0) return &SPMV_STENCIL_HALO_MGPU;
  #endif
#endif
    return NULL;
}

