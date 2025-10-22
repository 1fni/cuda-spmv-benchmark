/**
 * @file kokkos_cg_struct.cpp
 * @brief CG solver using Kokkos structured stencil SpMV (optimized)
 *
 * Adapted from Kokkos-Kernels pcgsolve() to use spmv_struct() for
 * 5-point stencil operations.
 */

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosBlas.hpp>
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

struct CGStats {
    size_t iterations;
    double total_time_s;
    double spmv_time_s;
    double norm_res;
    bool converged;
};

/**
 * CG solver using structured stencil SpMV
 */
CGStats cg_solve_struct(
    const matrix_type& A,
    const vector_type& b,
    vector_type& x,
    int grid_size,
    size_t max_iters = 1000,
    double tolerance = 1e-6)
{
    const int N = b.extent(0);
    CGStats stats;

    // Allocate CG vectors
    vector_type pAll("cg::p", N);
    vector_type p = Kokkos::subview(pAll, std::pair<size_t, size_t>(0, N));
    vector_type r("cg::r", N);
    vector_type Ap("cg::Ap", N);

    // Setup structured stencil parameters
    Kokkos::View<ordinal_type*, Kokkos::HostSpace> structure("structure", 2);
    structure(0) = grid_size;
    structure(1) = grid_size;
    const int stencil_type = 0;  // FD (5-point stencil)

    execution_space cuda_space;

    Kokkos::Timer total_timer;
    Kokkos::Timer spmv_timer;

    double spmv_time = 0.0;

    // r = b - A*x (with x0 = 0, so r = b)
    Kokkos::deep_copy(p, x);

    // Ap = A * p
    spmv_timer.reset();
    KokkosSparse::Experimental::spmv_struct(cuda_space, "N", stencil_type, structure, 1.0, A, pAll, 0.0, Ap);
    execution_space().fence();
    spmv_time += spmv_timer.seconds();

    // r = b - Ap
    Kokkos::deep_copy(r, Ap);
    KokkosBlas::axpby(1.0, b, -1.0, r);

    // p = r
    Kokkos::deep_copy(p, r);

    double old_rdot = KokkosBlas::dot(r, r);
    double norm_res = std::sqrt(old_rdot);

    // CG iterations
    size_t iter;
    for (iter = 0; iter < max_iters; iter++) {
        std::cout << "Running CG iteration " << iter << ", current resnorm = " << norm_res << '\n';

        // Ap = A * p (structured SpMV)
        spmv_timer.reset();
        KokkosSparse::Experimental::spmv_struct(cuda_space, "N", stencil_type, structure, 1.0, A, pAll, 0.0, Ap);
        execution_space().fence();
        spmv_time += spmv_timer.seconds();

        // pAp = dot(p, Ap)
        const double pAp_dot = KokkosBlas::dot(p, Ap);
        const double alpha = old_rdot / pAp_dot;

        // x += alpha * p
        KokkosBlas::axpby(alpha, p, 1.0, x);

        // r += -alpha * Ap
        KokkosBlas::axpby(-alpha, Ap, 1.0, r);

        const double r_dot = KokkosBlas::dot(r, r);
        norm_res = std::sqrt(r_dot);

        // Check convergence
        if (norm_res < tolerance) {
            iter++;
            break;
        }

        const double beta = r_dot / old_rdot;

        // p = r + beta * p
        KokkosBlas::axpby(1.0, r, beta, p);

        old_rdot = r_dot;
    }

    execution_space().fence();
    double total_time = total_timer.seconds();

    // Fill statistics
    stats.iterations = iter;
    stats.total_time_s = total_time;
    stats.spmv_time_s = spmv_time;
    stats.norm_res = norm_res;
    stats.converged = (norm_res < tolerance);

    return stats;
}

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
        std::cout << "Kokkos CG - Structured Stencil SpMV (Optimized)\n";
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

        // Compute b using structured SpMV
        Kokkos::View<ordinal_type*, Kokkos::HostSpace> structure("structure", 2);
        structure(0) = grid_size;
        structure(1) = grid_size;
        const int stencil_type = 0;

        execution_space cuda_space;
        KokkosSparse::Experimental::spmv_struct(cuda_space, "N", stencil_type, structure, 1.0, A, ones, 0.0, b);
        Kokkos::fence();

        std::cout << "RHS created (b = A*ones)\n";
        std::cout << "Initial guess: x0 = 0\n\n";

        // Solve
        std::cout << "Starting CG solver (structured stencil SpMV)...\n";
        std::cout << "========================================\n";

        CGStats stats = cg_solve_struct(A, b, x, grid_size, max_iters, tolerance);

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
        std::cout << "Converged: " << (stats.converged ? "YES" : "NO") << "\n";
        std::cout << "Iterations: " << stats.iterations << "\n";
        std::cout << "Final residual: " << std::scientific << stats.norm_res << "\n";
        std::cout << "Time-to-solution: " << std::fixed << std::setprecision(3)
                  << stats.total_time_s * 1000.0 << " ms\n\n";

        std::cout << "Breakdown:\n";
        std::cout << "  SpMV:       " << stats.spmv_time_s * 1000.0 << " ms ("
                  << std::setprecision(1) << 100.0 * stats.spmv_time_s / stats.total_time_s << "%)\n";
        std::cout << "  Other:      " << (stats.total_time_s - stats.spmv_time_s) * 1000.0 << " ms ("
                  << 100.0 * (stats.total_time_s - stats.spmv_time_s) / stats.total_time_s << "%)\n";
        std::cout << "========================================\n";
    }
    Kokkos::finalize();

    return 0;
}
