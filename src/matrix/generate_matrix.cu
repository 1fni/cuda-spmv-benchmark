/**
 * @file generate_matrix.cu
 * @brief Generates synthetic sparse matrices (e.g., 5-point stencil) in Matrix Market format.
 *
 * @details
 * Responsibilities:
 *  - Create grid-based stencil patterns (NxN)
 *  - Write output in .mtx format for use in benchmarks
 *
 * Usage:
 *  ./generate_matrix 100 matrices/example100x100.mtx
 *
 * Author: [Bouhrour Stephane]
 * Date: [2025-07-15]
 */

#include <stdio.h>
#include <stdlib.h>
#include "io.h"

/**
 * Writes a sparse matrix in Matrix Market format
 * using a 5-point stencil on a 2D grid (grid x grid).
 *
 * Each point is connected to:
 * - itself
 * - its left, right, top, and bottom neighbors (if they exist)
 */

int generate_matrix_stencil5(int grid, const char* filename) {
    // code to generate a 5-point stencil matrix in .mtx format
    return write_matrix_market_stencil5(grid, filename);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <grid_dim> <output_filename>\n", argv[0]);
        return 1;
    }

    int grid = atoi(argv[1]);                // 2D grid size
    const char* filename = argv[2];	   // output file
    return generate_matrix_stencil5(grid, filename); 
}

