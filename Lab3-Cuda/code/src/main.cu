#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cute/tensor.hpp>

#include <chrono>
#include <cmath>
#include <random>

#include "check.cuh"
#include "common.cuh"
#include "gemm.cuh"

constexpr uint32_t M = 8192;
constexpr uint32_t N = 8192;
constexpr uint32_t K = 8192;

#ifdef FP32
using Element = float;
#elif defined FP16
using Element = cute::half_t;
#else
#error "Not Implemented"
#endif

template <typename T>
int run() {
    // Generate random data

    std::cout << "Generating random data" << std::endl;

    std::random_device rd;
    thrust::default_random_engine rng(rd());
    thrust::normal_distribution<float> dist(0.0f, 1.0f);

    thrust::host_vector<float> data_A(M * K);
    thrust::host_vector<float> data_B(K * N);

    thrust::generate(data_A.begin(), data_A.end(), [&]() { return dist(rng); });
    thrust::generate(data_B.begin(), data_B.end(), [&]() { return dist(rng); });

    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(K * N);

    for (size_t i = 0; i < M * K; i++) {
        h_A[i] = static_cast<T>(data_A[i]);
    }
    for (size_t i = 0; i < K * N; i++) {
        h_B[i] = static_cast<T>(data_B[i]);
    }

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C(M * N);
    thrust::device_vector<T> d_C_cublas(M * N);

    // Run the kernel

    std::cout << "Running kernel" << std::endl;

    run_cublas<T>(d_A, d_B, d_C_cublas, M, N, K);
    run_custom<T>(d_A, d_B, d_C, M, N, K);

    thrust::host_vector<T> h_C_cublas = d_C_cublas;
    thrust::host_vector<T> h_C = d_C;

    if (check(h_C, h_C_cublas, (double)K)) {
        std::cout << "\033[1;32mSuccess\033[0m" << std::endl;
        return 0;
    } else {
        std::cout << "\033[1;31mFailure\033[0m" << std::endl;
        return 1;
    }
}

int main() {
    int result = run<Element>();
    if (result != 0) {
        return result;
    }
    return 0;
}