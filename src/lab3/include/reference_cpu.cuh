#pragma once
#include <chrono>
#include "cuda.h"
#include "common.cuh"

template <typename Element, typename ElementCompute>
void conv2d_cpu_ref(const Element *a, const Element *w, Element *b) {
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        size_t output_bytes = size_t(K) * sizeof(Element);
        Element *packedB = static_cast<Element *>(malloc(output_bytes));

        size_t input_bytes = size_t(R) * S * C * sizeof(Element);
        Element *packedA = static_cast<Element *>(malloc(input_bytes));

        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                // Collected needed input data,
                // A[n, i - R/2: i + R/2, j - S/2: j + S/2, :]
                int x = i - R / 2;
                int y = j - S / 2;
                int input_index = n * H * W * C + x * W * C + y * C;
                memset(packedA, 0, input_bytes);
                int A_buffer_index = 0;
                for (int r = 0; r < R; ++r) {
                    for (int s = 0; s < S; ++s) {
                        if (!(x < 0 || x >= H || y < 0 || y >= W)) {
                            memcpy(packedA + A_buffer_index, a + input_index, C * sizeof(Element));
                        } else {
                            memset(packedA + A_buffer_index, 0, C * sizeof(Element));
                        }
                        y++;
                        A_buffer_index += C;
                        input_index += C;
                    }
                    x++;
                    y -= S;
                    input_index = input_index - S * C + W * C;
                }

                // B[n, i, j, :]
                int output_index = n * H * W * K + i * W * K + j * K;
                memset(packedB, 0, output_bytes);

                int kernel_index = 0;

                // B[n, i, j, k] = \sum_{r,s,c} A[n, f(i, r), g(j, s), c] * W[k, r, s, c]
                for (int k = 0; k < K; ++k) {
                    ElementCompute result = 0;
                    A_buffer_index = 0;
                    for (int rs = 0; rs < R * S; ++rs) {
                        for (int c = 0; c < C; ++c) {
                            result += ElementCompute(packedA[A_buffer_index++]) * ElementCompute(w[kernel_index++]);
                        }
                    }
                    packedB[k] = static_cast<Element>(result);
                }
                memcpy(b + output_index, packedB, sizeof(Element) * K);
            }
        free(packedA);
        free(packedB);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CPU Conv2D: \n" << duration << " ms\n";
    std::cout << std::fixed << std::setprecision(4) << getTFLOPS(duration) << " TFLOPS\n";
}