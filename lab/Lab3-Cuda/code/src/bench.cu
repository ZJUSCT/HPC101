#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cute/tensor.hpp>
#include <chrono>
#include <cmath>
#include <random>
#include <iostream>
#include <nvbench/nvbench.cuh>
#include "common.cuh"
#include "gemm.cuh"


template <typename T, bool is_cublas>
void bench(nvbench::state& state) {
    size_t M = state.get_int64("M");
    size_t N = state.get_int64("N");
    size_t K = state.get_int64("K");

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

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
        if constexpr (is_cublas) {
            run_cublas<T>(d_A, d_B, d_C, M, N, K);
        } else {
            run_custom<T>(d_A, d_B, d_C, M, N, K);
        }
    });
}

#ifdef FP32
auto bench_fp32_cublas = bench<float, true>;
auto bench_fp32_custom = bench<float, false>;
NVBENCH_BENCH(bench_fp32_cublas)
    .set_name("bench_fp32_cublas")
    .add_int64_axis("M", { 8192 })
    .add_int64_axis("N", { 8192 })
    .add_int64_axis("K", { 8192 });
NVBENCH_BENCH(bench_fp32_custom)
    .set_name("bench_fp32_custom")
    .add_int64_axis("M", {8192})
    .add_int64_axis("N", {8192})
    .add_int64_axis("K", {8192});
#elif defined FP16
auto bench_fp16_cublas = bench<cute::half_t, true>;
auto bench_fp16_custom = bench<cute::half_t, false>;
NVBENCH_BENCH(bench_fp16_cublas)
    .set_name("bench_fp16_cublas")
    .add_int64_axis("M", {8192})
    .add_int64_axis("N", {8192})
    .add_int64_axis("K", {8192});
NVBENCH_BENCH(bench_fp16_custom)
    .set_name("bench_fp16_custom")
    .add_int64_axis("M", {8192})
    .add_int64_axis("N", {8192})
    .add_int64_axis("K", {8192});
#else
#error "Not Implemented"
#endif