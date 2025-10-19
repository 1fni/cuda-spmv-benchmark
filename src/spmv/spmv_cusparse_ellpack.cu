#include <stdio.h>
#include <cusparse.h>         
#include "spmv.h"

//ELLPACKMatrix ellpack_matrix;
//CSRMatrix csr_mat;

int ellpack_init(MatrixData* matrix) {
//	printf("[cusparse ELLPACK with CSR structure] Init from : %s\n", matrix_path);
//
//	int rows, cols, nnz;
//	int* row_ptr, *col_indices;
//	int nnz_general;
//	double * values;
//
//	int type = read_matrix_type(matrix_path);
//	if(type == 2) //if symmetric type
//	{
//		read_matrix_symtogen(matrix_path, &rows, &cols, &nnz, &row_ptr, &col_indices, &values, &nnz_general);
//		nnz = nnz_general;
//	}
//	else //if general type
//	{
//		read_matrix_general(matrix_path, &rows, &cols, &nnz, &row_ptr, &col_indices, &values);
//		nnz_general = nnz;
//	}
//
//	nx = rows;
//	ny = cols;
//
//	//convert csr to ellpack
//	csr_mat.nb_rows = rows;
//	csr_mat.nb_cols = cols;
//	csr_mat.nb_nonzeros = nnz;
//	csr_mat.row_ptr = row_ptr;
//	csr_mat.col_indices = col_indices;
//	csr_mat.values = values;
//
//	int max;
//	convert_csr_to_ellpack(&csr_mat, &ellpack_matrix, &max);
//	printf("Ellpack moyenne non zero per row %d, max non zero per row %d\n", ellpack_matrix.nb_nonzeros/ellpack_matrix.nb_rows, ellpack_matrix.ell_width);
//	//Leverage the structure of ELLPACK (uniform row width and aligned memory) to generate CSR inputs in a way that enables cuSPARSE 
//	//to benefit from efficient memory access patterns, essentially simulating ELLPACK alignment through supported CSR routines.
//	csr_mat.nb_rows = ellpack_matrix.nb_rows;
//	csr_mat.nb_cols = ellpack_matrix.nb_cols;
//	csr_mat.nb_nonzeros = ellpack_matrix.nb_nonzeros;
//	csr_mat.row_ptr[0] = 0;
//	for (int i = 0; i < ellpack_matrix.nb_rows; i++) {
//		csr_mat.row_ptr[i + 1] = (i+1)*max;
//	}
//	csr_mat.col_indices = ellpack_matrix.indices;
//	csr_mat.values = ellpack_matrix.values;
//	for (int i = 0; i < csr_mat.nb_rows; i++) {
//			printf("row_ptr %d\n", csr_mat.row_ptr[i]);
//		for (int j = 0; j < max; j++) {
//			printf("row %d, index ellpack %d, value %lf, col index %d", i, j, csr_mat.values[i*max + j], csr_mat.col_indices[i*max + j]);
//			puts("");
//		}
//			puts("");
//	}
	return 0;
}

int ellpack_run_timed(const double* x, double* y, double* kernel_time_ms) {
	// ELLPACK implementation currently not functional - returning placeholder timing
	*kernel_time_ms = 0.0;
	printf("[cusparse ELLPACK with CSR struct] Running SpMV\n");
	// SpMV: y = A * x using provided vectors
	double     alpha           = 1.0;
	double     beta            = 0.0;

	//device memory management
	int   *dA_csrOffsets, *dA_columns;
	double *dA_values, *dX, *dY;
	size_t tot_mem = 0;
	CUDA_CHECK( cudaMalloc((void**) &dA_csrOffsets,
				(csr_mat.nb_rows + 1) * sizeof(int)) )
		tot_mem += (csr_mat.nb_rows + 1) * sizeof(int);
	CUDA_CHECK( cudaMalloc((void**) &dA_columns, csr_mat.nb_nonzeros * sizeof(int))        )
		tot_mem += csr_mat.nb_nonzeros * sizeof(int);
	CUDA_CHECK( cudaMalloc((void**) &dA_values,  csr_mat.nb_nonzeros * sizeof(double))      )
		tot_mem += csr_mat.nb_nonzeros * sizeof(double);
	CUDA_CHECK( cudaMalloc((void**) &dX,         csr_mat.nb_cols * sizeof(double)) )
		CUDA_CHECK( cudaMalloc((void**) &dY,         csr_mat.nb_rows * sizeof(double)) )

		CUDA_CHECK( cudaMemcpy(dA_csrOffsets, csr_mat.row_ptr,
					(csr_mat.nb_rows + 1) * sizeof(int),
					cudaMemcpyHostToDevice) )
		CUDA_CHECK( cudaMemcpy(dA_columns, csr_mat.col_indices, csr_mat.nb_nonzeros * sizeof(int),
					cudaMemcpyHostToDevice) )
		CUDA_CHECK( cudaMemcpy(dA_values, csr_mat.values, csr_mat.nb_nonzeros * sizeof(double),
					cudaMemcpyHostToDevice) )
		CUDA_CHECK( cudaMemcpy(dX, x, csr_mat.nb_cols * sizeof(double),
					cudaMemcpyHostToDevice) )
		CUDA_CHECK( cudaMemcpy(dY, y, csr_mat.nb_rows * sizeof(double),
					cudaMemcpyHostToDevice) )

		fprintf(stderr, "tot memory matrix format %zu\n", tot_mem);

	//cusparse api
	cusparseHandle_t     handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	void*                dBuffer    = NULL;
	size_t               bufferSize = 0;
	CHECK_CUSPARSE( cusparseCreate(&handle) )
		//create sparse matrix A in CSR format

		CHECK_CUSPARSE( cusparseCreateCsr(&matA, csr_mat.nb_rows, csr_mat.nb_cols, csr_mat.nb_nonzeros,
					dA_csrOffsets, dA_columns, dA_values,
					CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
					CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )

		//sepcify matrix symetric (BY ME) 
		//CHECK_CUSPARSE( cusparseSetMatType(&matA, CUSPARSE_MATRIX_TYPE_SYMMETRIC) )
		//create dense vector X
		CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, csr_mat.nb_cols, dX, CUDA_R_64F) )
		//create dense vector y
		CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, csr_mat.nb_rows, dY, CUDA_R_64F) )
		// allocate an external buffer if needed
		CHECK_CUSPARSE( cusparseSpMV_bufferSize(
					handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					&alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
					CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
		CUDA_CHECK( cudaMalloc(&dBuffer, bufferSize) )

		cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//execute SpMV
	CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				&alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
				CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
		//cusparseCsrmvEx();
		cudaDeviceSynchronize();             // Assure la fin du kernel
						     // ====================================================

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Temps du SpMV : %.3f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//destroy matrix/vector descriptors
	CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
		CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
		CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
		CHECK_CUSPARSE( cusparseDestroy(handle) )

		// device result check
		CUDA_CHECK( cudaMemcpy(y, dY, csr_mat.nb_rows * sizeof(double),
					cudaMemcpyDeviceToHost) )
		double check_sum = 0.0;
	for (int i = 0; i < csr_mat.nb_rows; i++) {
		check_sum += y[i];
	}
	printf("check_sum %le\n", check_sum);

	// Free GPU memory
	CUDA_CHECK( cudaFree(dBuffer) )
		CUDA_CHECK( cudaFree(dA_csrOffsets) )
		CUDA_CHECK( cudaFree(dA_columns) )
		CUDA_CHECK( cudaFree(dA_values) )
		CUDA_CHECK( cudaFree(dX) )
		CUDA_CHECK( cudaFree(dY) )
		
	return EXIT_SUCCESS;
}

void ellpack_free() {
	printf("[ELLLPACK] Nothing to free\n");
}

SpmvOperator SPMV_ELLPACK = {
	.name = "ellpack",
	.init = ellpack_init,
	.run_timed = ellpack_run_timed,
	.run_device = NULL,
	.free = ellpack_free
};

