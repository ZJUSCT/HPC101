// Driver for the bonus task. This file is replaced with the original version
// when grading — put your code in student/lora_opt.cpp.
#include "lora.h"

#include <chrono>
#include <cstring>
#include <iostream>

#define N_ITER 10

int main() {
    float* x = new float[NUM_TOKENS * D_MODEL];
    float* y = new float[NUM_TOKENS * D_MODEL];
    float* y_ref = new float[NUM_TOKENS * D_MODEL];
    float* g = new float[NUM_TOKENS * D_MODEL];

    MoEWeights w;
    w.w_router = new float[NUM_EXPERTS * D_MODEL];
    w.bias = new float[NUM_EXPERTS];
    w.w_gate = new int8_t[(size_t)NUM_EXPERTS * D_FF * D_MODEL];
    w.w_up = new int8_t[(size_t)NUM_EXPERTS * D_FF * D_MODEL];
    w.w_down = new int8_t[(size_t)NUM_EXPERTS * D_MODEL * D_FF];
    w.s_gate = new float[NUM_EXPERTS];
    w.s_up = new float[NUM_EXPERTS];
    w.s_down = new float[NUM_EXPERTS];
    w.sh_gate = new int8_t[(size_t)D_FF * D_MODEL];
    w.sh_up = new int8_t[(size_t)D_FF * D_MODEL];
    w.sh_down = new int8_t[(size_t)D_MODEL * D_FF];

    LoRAWeights lora;
    lora.a = new float[(size_t)NUM_EXPERTS * LORA_RANK * D_FF];
    lora.b = new float[(size_t)NUM_EXPERTS * D_MODEL * LORA_RANK];

    float* grad_a = new float[(size_t)NUM_EXPERTS * LORA_RANK * D_FF];
    float* grad_b = new float[(size_t)NUM_EXPERTS * D_MODEL * LORA_RANK];

    init_data(x, w, 42);
    init_lora_data(lora, g, 1234);

    // Warm-up (untimed) so the baseline is not penalized for cold caches
    // or a not-yet-boosted CPU
    for (int iter = 0; iter < 5; iter++) {
        moe_lora_forward_ref(x, w, lora, y_ref);
    }

    // Forward: reference vs optimized (student/lora_opt.cpp)
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < N_ITER; iter++) {
        moe_lora_forward_ref(x, w, lora, y_ref);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ref = end_time - start_time;
    std::cout << "Forward baseline time:  " << duration_ref.count() << " s"
              << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < N_ITER; iter++) {
        moe_lora_forward_optimized(x, w, lora, y);
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_opt = end_time - start_time;
    std::cout << "Forward optimized time: " << duration_opt.count() << " s"
              << std::endl;

    check_result(y, y_ref, NUM_TOKENS, D_MODEL);
    std::cout << "Forward speedup: "
              << duration_ref.count() / duration_opt.count() << std::endl;

    // Backward: verify against finite differences, then time it
    size_t size_a = (size_t)NUM_EXPERTS * LORA_RANK * D_FF;
    size_t size_b = (size_t)NUM_EXPERTS * D_MODEL * LORA_RANK;
    memset(grad_a, 0, size_a * sizeof(float));
    memset(grad_b, 0, size_b * sizeof(float));
    moe_lora_backward(x, w, lora, g, grad_a, grad_b);

    if (!check_gradients(x, w, lora, g, grad_a, grad_b)) {
        return 1;
    }

    start_time = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < N_ITER; iter++) {
        memset(grad_a, 0, size_a * sizeof(float));
        memset(grad_b, 0, size_b * sizeof(float));
        moe_lora_backward(x, w, lora, g, grad_a, grad_b);
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_bwd = end_time - start_time;
    std::cout << "Backward time: " << duration_bwd.count() << " s" << std::endl;

    return 0;
}
