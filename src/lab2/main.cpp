// Driver for the main task. This file is replaced with the original version
// when grading — put your code in student/moe_opt.cpp.
#include "moe.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define N_ITER_WARMUP 10

<<<<<<< HEAD
int main(int argc, char** argv) {
    // --benchmark (optional, last argument): skip the baseline loop so a
    // profiler sees almost nothing but the optimized implementation.
    // Correctness is still checked; no Speedup is reported.
    bool benchmark = false;
    if (argc >= 2 && strcmp(argv[argc - 1], "--benchmark") == 0) {
        benchmark = true;
        argc--;
    }
    // Every dimension of the problem is a runtime parameter.
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: " << argv[0]
                  << " <num_tokens> <d_model> <d_ff> <num_experts> <top_k>"
                  << " [n_iter] [--benchmark]" << std::endl;
        return 1;
    }
    const int num_tokens = atoi(argv[1]);
    const int d_model = atoi(argv[2]);
    const int d_ff = atoi(argv[3]);
    const int num_experts = atoi(argv[4]);
    const int top_k = atoi(argv[5]);
    const int n_iter = argc == 7 ? atoi(argv[6]) : 1000;
    if (num_tokens < 1 || num_tokens > MAX_NUM_TOKENS || d_model < 64 ||
        d_model > MAX_D_MODEL || d_model % 64 != 0 || d_ff < 64 ||
        d_ff > MAX_D_FF || d_ff % 64 != 0 || num_experts < 1 ||
        num_experts > MAX_NUM_EXPERTS || top_k < 1 || top_k > MAX_TOP_K ||
        top_k > num_experts || n_iter < 1) {
        std::cerr << "invalid problem size" << std::endl;
        return 1;
    }

    std::cout << "problem size: num_tokens=" << num_tokens
              << " d_model=" << d_model << " d_ff=" << d_ff
              << " num_experts=" << num_experts << " top_k=" << top_k << " ("
              << n_iter << " iterations)" << std::endl;

    MoEWeights w;
    w.d_model = d_model;
    w.d_ff = d_ff;
    w.num_experts = num_experts;
    w.top_k = top_k;
=======
int main() {
    float* x = new float[NUM_TOKENS * D_MODEL];
    float* y = new float[NUM_TOKENS * D_MODEL];
    float* y_ref = new float[NUM_TOKENS * D_MODEL];

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
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)

    // Rotate several input batches through the timed loop so a cached
    // result cannot be reused across iterations.
    const int pool = n_iter < 16 ? n_iter : 16;
    float** x_pool = new float*[pool];
    for (int r = 0; r < pool; r++) {
        x_pool[r] = new float[(size_t)num_tokens * d_model];
    }
    float* y = new float[(size_t)num_tokens * d_model];
    float* y_ref = new float[(size_t)num_tokens * d_model];

    w.w_router = new float[(size_t)num_experts * d_model];
    w.bias = new float[num_experts];
    w.w_gate = new int8_t[(size_t)num_experts * d_ff * d_model];
    w.w_up = new int8_t[(size_t)num_experts * d_ff * d_model];
    w.w_down = new int8_t[(size_t)num_experts * d_model * d_ff];
    w.s_gate = new float[num_experts];
    w.s_up = new float[num_experts];
    w.s_down = new float[num_experts];
    w.sh_gate = new int8_t[(size_t)d_ff * d_model];
    w.sh_up = new int8_t[(size_t)d_ff * d_model];
    w.sh_down = new int8_t[(size_t)d_model * d_ff];

    init_data(x_pool[0], w, num_tokens, 42);
    for (int r = 1; r < pool; r++) {
        init_tokens(x_pool[r], num_tokens, d_model, 42ull + r);
    }

    std::chrono::duration<double> duration_ref(0.0);
    if (!benchmark) {
        // Warm-up (untimed): cold caches, not-yet-boosted CPU.
        for (int iter = 0; iter < N_ITER_WARMUP; iter++) {
            moe_forward_ref(x_pool[iter % pool], w, y_ref, num_tokens);
        }

        // Reference implementation (baseline)
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < n_iter; iter++) {
            moe_forward_ref(x_pool[iter % pool], w, y_ref, num_tokens);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        duration_ref = end_time - start_time;
        std::cout << "Baseline time:  " << duration_ref.count() << " s"
                  << std::endl;
    }

    // Optimized implementation (student/moe_opt.cpp)
    preprocess(w);
    if (benchmark) {
        // The baseline loop normally doubles as machine warm-up; warm up
        // here instead.
        for (int iter = 0; iter < N_ITER_WARMUP; iter++) {
            moe_forward_optimized(x_pool[iter % pool], w, y, num_tokens);
        }
    }

<<<<<<< HEAD
=======
    // Warm-up (untimed) so the baseline is not penalized for cold caches
    // or a not-yet-boosted CPU
    for (int iter = 0; iter < 5; iter++) {
        moe_forward_ref(x, w, y_ref);
    }

    // Reference implementation (baseline)
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < n_iter; iter++) {
        moe_forward_optimized(x_pool[iter % pool], w, y, num_tokens);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
<<<<<<< HEAD
=======
    std::chrono::duration<double> duration_ref = end_time - start_time;
    std::cout << "Baseline time:  " << duration_ref.count() << " s" << std::endl;

    // Optimized implementation (student/moe_opt.cpp)
    preprocess(w);

    start_time = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < N_ITER; iter++) {
        moe_forward_optimized(x, w, y);
    }
    end_time = std::chrono::high_resolution_clock::now();
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
    std::chrono::duration<double> duration_opt = end_time - start_time;
    std::cout << "Optimized time: " << duration_opt.count() << " s" << std::endl;

    // Verify on a fresh batch never used during timing: a cached result
    // has no entry for it and fails the check.
    float* x_verify = x_pool[0];
    init_tokens(x_verify, num_tokens, d_model, 0x5eed5eedull);
    moe_forward_optimized(x_verify, w, y, num_tokens);
    moe_forward_ref(x_verify, w, y_ref, num_tokens);
    check_result(y, y_ref, num_tokens, d_model);
    if (!benchmark) {
        std::cout << "Speedup: " << duration_ref.count() / duration_opt.count()
                  << std::endl;
    }

    return 0;
}
