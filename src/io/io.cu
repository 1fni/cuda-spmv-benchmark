/**
 * @file io.cu
 * @brief Handles I/O operations for Matrix Market files and memory allocation.
 *
 * @details
 * I/O functionality for sparse matrix operations:
 * - Reading sparse matrices from Matrix Market (.mtx) format
 * - Converting matrices to CSR (Compressed Sparse Row) representation
 * - Handling symmetry expansion for symmetric matrices
 * - Writing generated matrices to Matrix Market format
 * - Managing memory allocation and deallocation for matrix data structures
 *
 * The implementation supports both general and symmetric matrix formats,
 * automatically detecting the matrix type and applying appropriate conversion
 * strategies for optimal SpMV performance.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#include "io.h"
#include "spmv.h"

/**
 * @brief Determines the matrix type from Matrix Market file header.
 * @details Parses the Matrix Market file header to determine if the matrix
 * is stored in general or symmetric format. This information is crucial
 * for choosing the appropriate reading strategy.
 * @param filename Path to the Matrix Market file
 * @return 1 if general matrix, 2 if symmetric matrix, -1 on error
 */
int read_matrix_type(const char* filename) {
	int ret = -1;
	FILE *file;

	// Determine if the matrix is general or symmetrical
	file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stderr, "Error opening file\n");
		return ret;
	}

	char line[MAX_LINE_LENGTH];
	while (fgets(line, sizeof(line), file) != NULL) {
		// Check if the line is commented
		if (!(line[0] == '%')) {
			fprintf(stderr, "Error opening file\n");
			// Return if symmetrical or general have not been found while parsing header
			return ret;
		}

		if (strstr(line, "general") != NULL) {
			return 1;
		}
		if (strstr(line, "symmetric") != NULL) {
			return 2;
		}
	}

	fclose(file);
	return ret;
}

/**
 * @brief Loads a matrix from Matrix Market format into MatrixData structure.
 * @details Main entry point for matrix loading. Automatically detects matrix type
 * (general or symmetric) and calls the appropriate reading function. For symmetric
 * matrices, performs expansion to full format for efficient SpMV operations.
 * @param matrix_path Path to the Matrix Market file
 * @param mat Pointer to MatrixData structure to fill
 * @return 0 on success, non-zero on error
 */
int load_matrix_market(const char* matrix_path, MatrixData* mat) {
    // Read the matrix, fill mat->rows, mat->cols, mat->nnz, mat->entries
    printf("Loading matrix: %s\n", matrix_path);
    int rows, cols, nnz;
    int* row_ptr, *col_indices;
    int nnz_general;
    double * values;

    int type = read_matrix_type(matrix_path);
    if(type == 2) // If symmetric type
    {
	    read_matrix_symtogen(mat, matrix_path, &rows, &cols, &nnz, &row_ptr, &col_indices, &values, &nnz_general);
	    nnz = nnz_general;
    }
    else // If general type
    {
	    read_matrix_general(mat, matrix_path, &rows, &cols, &nnz, &row_ptr, &col_indices, &values);
    }

    return 0;
}

/**
 * @brief Reads a general (non-symmetric) matrix from Matrix Market format.
 * @details Parses a Matrix Market file containing a general matrix and populates
 * the MatrixData structure with entries in coordinate format. Adjusts 1-based
 * Matrix Market indices to 0-based indexing used internally.
 * @param mat Pointer to MatrixData structure to populate
 * @param filename Path to the Matrix Market file
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @param nnz Pointer to store number of non-zero elements
 * @param csr_rowptr Pointer to CSR row pointer array (unused in this function)
 * @param csr_colind Pointer to CSR column indices array (unused in this function)
 * @param csr_val Pointer to CSR values array (unused in this function)
 */
void read_matrix_general(MatrixData* mat, const char* filename, int* rows, int* cols, int* nnz, int** csr_rowptr, int** csr_colind, double ** csr_val) {
	FILE *file;
	int i;
	Entry *entries;

	file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stderr, "Error opening file\n");
		return;
	}

	char buffer[MAX_LINE_LENGTH]; // Buffer to store each line
	int grid_size = -1; // Valeur par défaut si pas trouvé
	while (fgets(buffer, MAX_LINE_LENGTH, file) != NULL) {
		if (buffer[0] != '%') { 
			// Process the line here (or skip it)
			sscanf(buffer, "%d %d %d", rows, cols, nnz);
			break; // Stop when three digits are found
		} else {
			// Chercher le commentaire STENCIL_GRID_SIZE
			if (strstr(buffer, "STENCIL_GRID_SIZE") != NULL) {
				sscanf(buffer, "%% STENCIL_GRID_SIZE %d", &grid_size);
			}
		}
	}

	entries = (Entry *)malloc((*nnz) * sizeof(Entry));
	if (!entries) { fprintf(stderr, "Allocation failed at line %d\n", __LINE__); exit(1); }
	if (entries == NULL) {
		fprintf(stderr, "Memory allocation error\n");
		return;
	}

	mat->entries = entries;
	mat->rows = *rows;
	mat->cols= *cols;
	mat->nnz = *nnz;
	mat->grid_size = grid_size; // Stocker le grid_size extrait

	// Read entries
	for (i = 0; i < *nnz; i++) {
		int items_read = fscanf(file, "%d %d %le", &entries[i].row, &entries[i].col, &entries[i].value);
		if (items_read != 3) {
			fprintf(stderr, "Error reading matrix entry %d (expected 3 items, got %d)\n", i, items_read);
			free(entries);
			fclose(file);
			return;
		}
		// Adjust indices (if necessary, because indices start from 1 in Matrix Market format)
		entries[i].row--; // Adjust for 0-based indexing
		entries[i].col--; // Adjust for 0-based indexing
	}

	fclose(file);

	return;
}

/**
 * @brief Reads a symmetric matrix and expands it to general format with CSR conversion.
 * @details Processes a Matrix Market file containing a symmetric matrix (lower triangular
 * part only) and expands it to full general format. Simultaneously converts the data
 * to CSR format for efficient SpMV operations. Handles diagonal elements correctly
 * to avoid duplication.
 * @param mat Pointer to MatrixData structure to populate
 * @param filename Path to the Matrix Market file
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @param nnz Pointer to store original number of non-zero elements
 * @param csr_rowptr Pointer to store CSR row pointer array
 * @param csr_colind Pointer to store CSR column indices array
 * @param csr_val Pointer to store CSR values array
 * @param nnz_general Pointer to store expanded number of non-zero elements
 */
void read_matrix_symtogen(MatrixData* mat, const char* filename, int* rows, int* cols, int* nnz, int** csr_rowptr, int** csr_colind, double ** csr_val, int *nnz_general) {
	FILE *file;
	int i;
	Entry *entries;
	int *row_ptr, *col_indices;;
	double *values;

	// Open the file
	file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stderr, "Error opening file\n");
		return;
	}

	// Read the header (Matrix Market format)
	// Skip comment mtx format
	char buffer[MAX_LINE_LENGTH]; // Buffer to store each line
	while (fgets(buffer, MAX_LINE_LENGTH, file) != NULL) {
		if (buffer[0] != '%') { 
			// Process the line here (or skip it)
			sscanf(buffer, "%d %d %d", rows, cols, nnz);
			break; // Stop when three digits are found
		}
	}

	// fscanf(file, "%d %d %d", rows, cols, nnz); // Read matrix dimensions and number of non-zero entries
	fprintf(stderr, "in %s\n", filename);
	fprintf(stderr, "rows %d cols %d nnz %d \n", *rows, *cols, *nnz);

	entries = (Entry *)malloc(*nnz * sizeof(Entry)); 
	if (entries == NULL) {
		fprintf(stderr, "Memory allocation error\n");
		return;
	}

	// Read entries
	int nb_value_on_diag = 0;
	for (i = 0; i < *nnz; i++) {
		int items_read = fscanf(file, "%d %d %le", &entries[i].row, &entries[i].col, &entries[i].value);
		if (items_read != 3) {
			fprintf(stderr, "Error reading symmetric matrix entry %d (expected 3 items, got %d)\n", i, items_read);
			free(entries);
			fclose(file);
			return;
		}
		if(i < 1)
		{
			fprintf(stderr, "%d %d %le\n", entries[i].row, entries[i].col, entries[i].value);
		}
		// Store the lower part are on diag and immediately after the symmetric if not on diag
		if(entries[i].col == entries[i].row)
		{
			nb_value_on_diag++;
		}
		// Adjust indices (if necessary, because indices start from 1 in Matrix Market format)
		entries[i].row--; // Adjust for 0-based indexing
		entries[i].col--; // Adjust for 0-based indexing
	}

	// Close the file
	fclose(file);

	// Allocate memory for CSR format arrays
	*nnz_general = (2*(*nnz) - nb_value_on_diag); // (2*(*nnz) - nb_value_on_diag) for the full nnz not only diag and lower

	row_ptr = (int *)calloc(*rows + 1, sizeof(int));
	col_indices = (int *)malloc(*nnz_general * sizeof(int));
	values = (double *)malloc(*nnz_general * sizeof(double ));

	// To simplify access to pointers
	*csr_rowptr = row_ptr;
	*csr_colind = col_indices;
	*csr_val = values;

	if (row_ptr == NULL || col_indices == NULL || values == NULL) {
		fprintf(stderr, "Memory allocation error\n");
		free(entries);
		return;
	}

	// Convert to CSR format
	for (i = 0; i < *nnz; i++) {
		row_ptr[entries[i].row + 1]++; // [row +1] for accumulate row offset in tab per row on top of accumulation prec row ind
		if(entries[i].col != entries[i].row)
		{
			row_ptr[entries[i].col + 1]++; // To have a general matrix csr not only the lower and diag part
		}
	}

	for (i = 1; i < *rows+1; i++) {
		row_ptr[i] += row_ptr[i-1];
	}
	row_ptr[0] = 0;

	int *cpt_row_local_idx = (int *)calloc(*rows, sizeof(int));
	for (i = 0; i < *nnz; i++) {
		int row = entries[i].row;
		int index = row_ptr[row];
		col_indices[index + cpt_row_local_idx[row]] = entries[i].col;
		values[index + cpt_row_local_idx[row]] = entries[i].value;
		cpt_row_local_idx[row]++;
		// printf("value row %d found, value %le\n", entries[i].row, values[index + cpt_row_local_idx[row]]);
		if(entries[i].row != entries[i].col) // Then add the symmetric
		{
			int row = entries[i].col;
			int index = row_ptr[row];
			col_indices[index + cpt_row_local_idx[row]] = entries[i].row;
			values[index + cpt_row_local_idx[row]] = entries[i].value;
			// printf("value sym of row %d found, value %le\n", entries[i].row, values[index + cpt_row_local_idx[row]]);
			cpt_row_local_idx[row]++;
		}
	}

	return;
}

/**
 * @brief Generates and writes a 5-point stencil matrix to Matrix Market format.
 * @details Creates a 5-point finite difference stencil matrix for a 2D grid of size n×n.
 * The resulting matrix has dimensions (n²)×(n²) and represents discretized Laplacian
 * operator. Each interior point connects to its 4 neighbors with value -1.0 and
 * has a center value of -4.0. Boundary conditions are handled naturally.
 * @param n Grid dimension (creates n×n grid, resulting in n²×n² matrix)
 * @param filename Output file path for Matrix Market format
 * @return 0 on success, non-zero on error
 */
int write_matrix_market_stencil5 (int n, const char* filename){

	int grid_size = n * n;

	// Calculate the exact number of non-zeros
	int nnz = 0;
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			nnz++; // Center
			if (col > 0) nnz++; // Left
			if (col < n - 1) nnz++; // Right
			if (row > 0) nnz++; // Top
			if (row < n - 1) nnz++; // Bottom
		}
	}

	FILE* f = fopen(filename, "w");
	if (!f) {
		perror("fopen");
		exit(1);
	}

	// Write Matrix Market header
	fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
	fprintf(f, "%% STENCIL_GRID_SIZE %d\n", n);  // Commentaire avec n original
	fprintf(f, "%d %d %d\n", grid_size, grid_size, nnz);

	// Write values with progress indication
	long long total_points = (long long)n * n;
	long long progress_step = total_points / 100; // Print every 1%
	if (progress_step == 0) progress_step = 1; // Ensure progress for small matrices
	
	printf("Writing matrix entries: 0%%");
	fflush(stdout);
	
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			int idx = row * n + col + 1;  // 1-based index
			long long current_point = (long long)row * n + col;
			
			// Progress indicator - update every 1% or every 1000 points (whichever is smaller)
			if (current_point % progress_step == 0 || current_point % 1000 == 0) {
				int percent = (int)((current_point * 100) / total_points);
				printf("\rWriting matrix entries: %d%%", percent);
				fflush(stdout);
			}

			// Center
			fprintf(f, "%d %d -4.0\n", idx, idx);

			// Left
			if (col > 0)
				fprintf(f, "%d %d -1.0\n", idx, idx - 1);

			// Right
			if (col < n - 1)
				fprintf(f, "%d %d -1.0\n", idx, idx + 1);

			// Top
			if (row > 0)
				fprintf(f, "%d %d -1.0\n", idx, idx - n);

			// Bottom
			if (row < n - 1)
				fprintf(f, "%d %d -1.0\n", idx, idx + n);
		}
	}
	
	printf("\rWriting matrix entries: 100%%\n");
	fclose(f);
	printf("Matrix generated: %s (%dx%d, %d nnz)\n", filename, grid_size, grid_size, nnz);
	return 0;
}
