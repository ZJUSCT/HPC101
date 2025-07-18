#include "conv.cuh"

#define a(_n, _x, _y, _c) a[(_n) * H * W * C + (_x) * W * C + (_y) * C + (_c)]
#define w(_k, _x, _y, _c) w[(_k) * R * S * C + (_x) * S * C + (_y) * C + (_c)]
#define b(_n, _x, _y, _k) b[(_n) * H * W * K + (_x) * W * K + (_y) * K + (_k)]

static constexpr int BLOCK = 16;

// INT8
template <>
KernelConfig get_kernel_config<int8_t>() {
    KernelConfig config;
    config.grid = dim3((H + BLOCK - 1) / BLOCK, (W + BLOCK - 1) / BLOCK);
    config.block = dim3(BLOCK, BLOCK);
    config.shared_memory_size = 0;  // Use default shared memory size
    return config;
}

template <>
__global__ void conv2d_cuda_kernel<int8_t, int>(const int8_t *__restrict__ a, const int8_t *__restrict__ w,
                                                int8_t *__restrict__ b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                int result = 0;
                for (int c = 0; c < C; ++c) {
                    int x = i - R / 2, y = j - S / 2;
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            if (!(x < 0 || x >= size || y < 0 || y >= size)) {
                                result += static_cast<int>(a(n, x, y, c)) * static_cast<int>(w(k, r, s, c));
                            }
                            y++;
                        }
                        x++;
                        y -= S;
                    }
                }
                b(n, i, j, k) = static_cast<int8_t>(result);
            }
        }
    }
}

// HALF
template <>
KernelConfig get_kernel_config<half_t>() {
    KernelConfig config;
    config.grid = dim3((H + BLOCK - 1) / BLOCK, (W + BLOCK - 1) / BLOCK);
    config.block = dim3(BLOCK, BLOCK);
    config.shared_memory_size = 0;  // Use default shared memory size
    return config;
}

template <>
__global__ void conv2d_cuda_kernel<half_t, float>(const half_t *__restrict__ a, const half_t *__restrict__ w,
                                                  half_t *__restrict__ b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                float result = 0;
                for (int c = 0; c < C; ++c) {
                    int x = i - R / 2, y = j - S / 2;
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            if (!(x < 0 || x >= size || y < 0 || y >= size)) {
                                result += static_cast<int>(a(n, x, y, c)) * static_cast<int>(w(k, r, s, c));
                            }
                            y++;
                        }
                        x++;
                        y -= S;
                    }
                }
                b(n, i, j, k) = static_cast<half_t>(result);
            }
        }
    }
}
