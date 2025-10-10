// Generate stencil matrix in Kokkos column order
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <grid_size> <output.mtx>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    const char* filename = argv[2];
    int grid_size = n * n;

    // Calculate nnz
    int nnz = 0;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row > 0) nnz++; // Top
            if (col > 0) nnz++; // Left
            nnz++; // Center
            if (col < n - 1) nnz++; // Right
            if (row < n - 1) nnz++; // Bottom
        }
    }

    FILE* f = fopen(filename, "w");
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%% STENCIL_GRID_SIZE %d\n", n);
    fprintf(f, "%d %d %d\n", grid_size, grid_size, nnz);

    // Write in Kokkos order: Top, Left, Center, Right, Bottom
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            int idx = row * n + col + 1;  // 1-based

            // Top (-n)
            if (row > 0)
                fprintf(f, "%d %d -1.0\n", idx, idx - n);
            // Left (-1)
            if (col > 0)
                fprintf(f, "%d %d -1.0\n", idx, idx - 1);
            // Center (0)
            fprintf(f, "%d %d -4.0\n", idx, idx);
            // Right (+1)
            if (col < n - 1)
                fprintf(f, "%d %d -1.0\n", idx, idx + 1);
            // Bottom (+n)
            if (row < n - 1)
                fprintf(f, "%d %d -1.0\n", idx, idx + n);
        }
    }

    fclose(f);
    printf("Kokkos-ordered matrix: %s (%dx%d, %d nnz)\n", filename, grid_size, grid_size, nnz);
    return 0;
}
