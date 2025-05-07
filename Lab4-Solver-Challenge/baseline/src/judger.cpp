#include "judger.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

void read_data(const std::string filename, int* N, double** A, double** b, double** x) {
    std::ifstream ifs(filename, std::ios::binary);

    if (!ifs) {
        std::cerr << "Failed to read data file at " << filename << std::endl;
        exit(-1);
    }

    ifs.read(reinterpret_cast<char*>(N), sizeof(int));
    *A = (double*)calloc((*N) * (*N), sizeof(double));
    *b = (double*)calloc(*N, sizeof(double));
    *x = (double*)calloc(*N, sizeof(double));  // x = 0 as initial guess

    // Read A, b, x as binary
    int size = (*N) * (*N);
    ifs.read(reinterpret_cast<char*>(*A), size * sizeof(double));
    ifs.read(reinterpret_cast<char*>(*b), *N * sizeof(double));

    ifs.close();
}

static int check_answer(int N, double* A, double* x, double* b) {
    // b_computed = A * x
    double* b_computed = (double*)calloc(N, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b_computed[i] += A[i * N + j] * x[j];
        }
    }

    // ||r||2 / ||b||2
    double res_l2norm = 0.0;
    double b_l2norm   = 0.0;
    for (int i = 0; i < N; i++) {
        res_l2norm += (b[i] - b_computed[i]) * (b[i] - b_computed[i]);
        b_l2norm += b[i] * b[i];
    }
    res_l2norm = sqrt(res_l2norm);
    b_l2norm   = sqrt(b_l2norm);
    double rr  = res_l2norm / b_l2norm;
    printf(" Check: Relative residual = %e\n", rr);

    // Judge if l2norm meets tolerance
    if (!std::isnan(rr) && rr < TOL) {
        printf(" Result: \033[1;32mAccepted, great job! :)\033[0m\n");
        return 1;
    }

    printf(" Result: \033[1;31mWrong Answer, please check carefully~ :)\033[0m\n");
    return 0;
}

void judge(int iter, Duration duration, int N, double* A, double* b, double* x) {
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

    std::cout << "====================== Summary ======================" << std::endl;
    std::cout << " Elapsed time: " << time / 1e6 << " s" << std::endl;
    if (iter == -1) {
        std::cout << " Status: Failed to converge after " << MAX_ITER << " iterations."
                  << std::endl;
    } else {
        std::cout << " Status: Converged after " << iter << " iterations." << std::endl;
    }

    // Check the result
    int result = check_answer(N, A, x, b);

    std::cout << "=====================================================" << std::endl;
}
