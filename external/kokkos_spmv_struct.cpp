#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_kokkos.mtx>\n";
        return 1;
    }

    std::string matrix_file = argv[1];

    Kokkos::initialize(argc, argv);
    {
        using ordinal_type = int;
        using size_type = int;
        using scalar_type = double;
        using device_type = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
        using matrix_type = KokkosSparse::CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
        using vector_type = Kokkos::View<scalar_type*, device_type>;

        // Load Kokkos-ordered matrix
        matrix_type A = KokkosSparse::Impl::read_kokkos_crst_matrix<matrix_type>(matrix_file.c_str());
        int N = A.numRows();
        int grid_size = static_cast<int>(std::sqrt(N));

        // Create vectors
        vector_type x("x", N);
        vector_type y("y", N);
        Kokkos::deep_copy(x, 1.0);
        Kokkos::deep_copy(y, 0.0);

        // Setup structured stencil parameters (2D grid)
        Kokkos::View<ordinal_type*, Kokkos::HostSpace> structure("structure", 2);
        structure(0) = grid_size;  // X dimension
        structure(1) = grid_size;  // Y dimension

        const int stencil_type = 0; // FD = Finite Difference (5-point stencil for 2D)
        scalar_type alpha = 1.0;
        scalar_type beta = 0.0;

        Kokkos::Cuda cuda_space;

        // Warmup
        for (int i = 0; i < 10; i++) {
            KokkosSparse::Experimental::spmv_struct(cuda_space, "N", stencil_type, structure, alpha, A, x, beta, y);
        }
        Kokkos::fence();

        // Benchmark
        int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            KokkosSparse::Experimental::spmv_struct(cuda_space, "N", stencil_type, structure, alpha, A, x, beta, y);
        }
        Kokkos::fence();

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        // Compute checksum
        auto y_host = Kokkos::create_mirror_view(y);
        Kokkos::deep_copy(y_host, y);
        double checksum = 0.0;
        for (int i = 0; i < N; i++) {
            checksum += y_host(i);
        }

        // Output
        std::cout << "Kokkos Structured SpMV (5-point stencil optimized)\n";
        std::cout << "Grid: " << grid_size << "x" << grid_size << "\n";
        std::cout << "Rows: " << N << "\n";
        std::cout << "NNZ: " << A.nnz() << "\n";
        std::cout << "Time: " << time_ms << " ms\n";
        std::cout << "Checksum: " << checksum << "\n";
    }
    Kokkos::finalize();

    return 0;
}
