/**
 * @file kokkos_cg_baseline.cpp
 * @brief CG solver using Kokkos generic CRS SpMV (baseline)
 *
 * Uses Kokkos-Kernels pcgsolve() without preconditioning.
 * Provides baseline comparison for structured stencil optimization.
 */

#include "KokkosSparse_pcg.hpp"
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosKernels_Handle.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using ordinal_type = int;
using size_type = int;
using scalar_type = double;
using device_type = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
using matrix_type = KokkosSparse::CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
using vector_type = Kokkos::View<scalar_type*, device_type>;
using execution_space = Kokkos::Cuda;
using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
    size_type, ordinal_type, scalar_type,
    execution_space, Kokkos::CudaSpace, Kokkos::CudaSpace>;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_kokkos.mtx> [--tol=1e-6] [--max-iters=1000]\n";
        std::cerr << "Example: " << argv[0] << " matrix/stencil_512x512_kokkos.mtx\n";
        return 1;
    }

    std::string matrix_file = argv[1];
    double tolerance = 1e-6;
    int max_iters = 1000;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg.find("--tol=") == 0) {
            tolerance = std::stod(arg.substr(6));
        } else if (arg.find("--max-iters=") == 0) {
            max_iters = std::stoi(arg.substr(12));
        }
    }

    Kokkos::initialize(argc, argv);
    {
        std::cout << "========================================\n";
        std::cout << "Kokkos CG - Generic CRS SpMV (Baseline)\n";
        std::cout << "========================================\n";
        std::cout << "Matrix: " << matrix_file << "\n";
        std::cout << "Tolerance: " << tolerance << "\n";
        std::cout << "Max iterations: " << max_iters << "\n\n";

        // Load matrix
        matrix_type A = KokkosSparse::Impl::read_kokkos_crst_matrix<matrix_type>(matrix_file.c_str());
        int N = A.numRows();
        int grid_size = static_cast<int>(std::sqrt(N));

        std::cout << "Matrix loaded: " << N << " x " << N << ", " << A.nnz() << " nonzeros\n";
        std::cout << "Grid: " << grid_size << " x " << grid_size << "\n\n";

        // Create RHS: b = A * ones
        vector_type ones("ones", N);
        vector_type b("b", N);
        vector_type x("x", N);

        Kokkos::deep_copy(ones, 1.0);
        Kokkos::deep_copy(x, 0.0);  // x0 = 0

        KokkosSparse::spmv("N", 1.0, A, ones, 0.0, b);
        Kokkos::fence();

        std::cout << "RHS created (b = A*ones)\n";
        std::cout << "Initial guess: x0 = 0\n\n";

        // Setup kernel handle (no preconditioning)
        KernelHandle kh;

        // Solve with PCG (use_sgs = false → unpreconditioned CG)
        KokkosKernels::Experimental::Example::CGSolveResult stats;

        std::cout << "Starting CG solver (generic CRS SpMV)...\n";
        std::cout << "========================================\n";

        KokkosKernels::Experimental::Example::pcgsolve(
            kh, A, b, x,
            max_iters,
            tolerance,
            &stats,
            false  // use_sgs = false → unpreconditioned CG
        );

        std::cout << "========================================\n\n";

        // Verify solution
        auto x_host = Kokkos::create_mirror_view(x);
        Kokkos::deep_copy(x_host, x);

        double error = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = x_host(i) - 1.0;
            error += diff * diff;
        }
        error = std::sqrt(error / N);

        std::cout << "Solution error (RMS vs exact=1): " << std::scientific << error << "\n\n";

        // Summary
        std::cout << "========================================\n";
        std::cout << "Summary\n";
        std::cout << "========================================\n";
        std::cout << "Converged: " << (stats.norm_res < tolerance ? "YES" : "NO") << "\n";
        std::cout << "Iterations: " << stats.iteration << "\n";
        std::cout << "Final residual: " << std::scientific << stats.norm_res << "\n";
        std::cout << "Time-to-solution: " << std::fixed << std::setprecision(3)
                  << stats.iter_time * 1000.0 << " ms\n\n";

        std::cout << "Breakdown:\n";
        std::cout << "  SpMV:       " << stats.matvec_time * 1000.0 << " ms ("
                  << std::setprecision(1) << 100.0 * stats.matvec_time / stats.iter_time << "%)\n";
        std::cout << "  Other:      " << (stats.iter_time - stats.matvec_time) * 1000.0 << " ms ("
                  << 100.0 * (stats.iter_time - stats.matvec_time) / stats.iter_time << "%)\n";
        std::cout << "========================================\n";
    }
    Kokkos::finalize();

    return 0;
}
