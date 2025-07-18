#pragma once

#include "common.cuh"

using half_t = cutlass::half_t;

template <typename Element, typename ElementCompute>
__global__ void conv2d_cuda_kernel(const Element *__restrict__ a, const Element *__restrict__ w,
                                   Element *__restrict__ b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                ElementCompute conv = 0;
                for (int c = 0; c < C; ++c) {
                    int x = i - R / 2, y = j - S / 2;
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            if (!(x < 0 || x >= size || y < 0 || y >= size)) {
                                conv += static_cast<ElementCompute>(a(n, x, y, c)) *
                                        static_cast<ElementCompute>(w(k, r, s, c));
                            }
                            y++;
                        }
                        x++;
                        y -= S;
                    }
                }
                b(n, i, j, k) = static_cast<half_t>(conv);
            }
        }
    }
}

template <>
__global__ void conv2d_cuda_kernel<int8_t, int>(const int8_t *__restrict__ a, const int8_t *__restrict__ w,
                                                int8_t *__restrict__ b);
template <>
__global__ void conv2d_cuda_kernel<half_t, float>(const half_t *__restrict__ a, const half_t *__restrict__ w,
                                                  half_t *__restrict__ b);



struct KernelConfig {
    dim3 grid;
    dim3 block;
    int shared_memory_size;  // 0 for default
};

template <typename T>
KernelConfig get_kernel_config();

template <>
KernelConfig get_kernel_config<int8_t>();

template <>
KernelConfig get_kernel_config<half_t>();


