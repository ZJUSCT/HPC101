#include <cstdlib>
#include <iostream>

#include "judger.h"

extern "C" int bicgstab(int N, double* A, double* b, double* x, int max_iter, double tol);

int main(int argc, char* argv[]) {
    int world_size, world_rank = 0;
    // When using MPI, please remember to initialize here

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_data>" << std::endl;
        return -1;
    }
    // Read data from file
    std::string filename = argv[1];

    // N: size of matrix A (N x N)
    // A: matrix A
    // b: vector b
    // x: initial guess of solution
    int N;
    double *A = nullptr, *b = nullptr, *x = nullptr;

    // Read data from file
    if (world_rank == 0) {
        read_data(filename, &N, &A, &b, &x);
    }

    // Call BiCGSTAB function
    auto start = std::chrono::high_resolution_clock::now();
    int iter   = bicgstab(N, A, b, x, MAX_ITER, TOL);
    auto end   = std::chrono::high_resolution_clock::now();

    // Check the result
    if (world_rank == 0) {
        auto duration = end - start;
        judge(iter, duration, N, A, b, x);
    }

    // Free allocated memory
    free(A);
    free(b);
    free(x);

    // When using MPI, please remember to finalize here
    return 0;
}
