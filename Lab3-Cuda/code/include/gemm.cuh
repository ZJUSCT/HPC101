#pragma once
#include <thrust/device_vector.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

template <typename T>
void run_cublas(thrust::device_vector<T> &d_A, thrust::device_vector<T> &d_B,
                thrust::device_vector<T> &d_C, const int m,
                const int n, const int k);

template <typename T>
void run_custom(thrust::device_vector<T> &d_A, thrust::device_vector<T> &d_B,
                thrust::device_vector<T> &d_C, const int m,
                const int n, const int k);
