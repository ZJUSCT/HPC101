#include "moe.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

// Unit-variance normal clamped to [-4, 4]: the spread in per-token max|x|
// makes the per-token quantization scale s_x = max|x| / 127 genuinely vary.
void init_tokens(float* x, int num_tokens, int d_model, uint64_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist_x(0.0f, 1.0f);
    for (size_t i = 0; i < (size_t)num_tokens * d_model; i++) {
        float v = dist_x(gen);
        x[i] = v < -4.0f ? -4.0f : (v > 4.0f ? 4.0f : v);
    }
}

void init_data(float* x, MoEWeights& w, int num_tokens, uint64_t seed) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;

    std::mt19937 gen(seed);
    // Small router weights keep sigmoid affinities off saturation; the
    // 1/sqrt(d_model) factor keeps the logit w_router . x near unit variance.
    const float router_range = 0.115f * sqrtf(256.0f / (float)d_model);
    std::uniform_real_distribution<float> dist_router(-router_range,
                                                      router_range);
    std::uniform_real_distribution<float> dist_bias(-0.1f, 0.1f);
    std::uniform_int_distribution<int> dist_i8(-127, 127);
    std::uniform_real_distribution<float> dist_scale(0.002f, 0.008f);

    // Draw weights before tokens so the model is identical for every
    // num_tokens; only the input batch changes.
    for (size_t i = 0; i < (size_t)num_experts * d_model; i++) {
        w.w_router[i] = dist_router(gen);
    }
    for (int e = 0; e < num_experts; e++) {
        w.bias[e] = dist_bias(gen);
    }
    for (size_t i = 0; i < (size_t)num_experts * d_ff * d_model; i++) {
        w.w_gate[i] = (int8_t)dist_i8(gen);
        w.w_up[i] = (int8_t)dist_i8(gen);
    }
    for (size_t i = 0; i < (size_t)num_experts * d_model * d_ff; i++) {
        w.w_down[i] = (int8_t)dist_i8(gen);
    }
    for (int e = 0; e < num_experts; e++) {
        w.s_gate[e] = dist_scale(gen);
        w.s_up[e] = dist_scale(gen);
        w.s_down[e] = dist_scale(gen);
    }
    for (size_t i = 0; i < (size_t)d_ff * d_model; i++) {
        w.sh_gate[i] = (int8_t)dist_i8(gen);
        w.sh_up[i] = (int8_t)dist_i8(gen);
        w.sh_down[i] = (int8_t)dist_i8(gen);
    }
    w.sh_s_gate = dist_scale(gen);
    w.sh_s_up = dist_scale(gen);
    w.sh_s_down = dist_scale(gen);

    // First batch, from a seed distinct from the weight seed.
    init_tokens(x, num_tokens, d_model, seed * 2654435761u + 1u);
}

// The int8 matmuls are exact; only the surrounding fp32 stages may reorder.
// A 1-ulp shift in the hidden activation can flip one round(h / s_h) by +-1
// and smear it across an output row, so a per-element tolerance is too
// fragile. We compare by per-token relative L2 error and global relative
// RMSE, which absorb isolated rounding flips but not a mis-routed token.
void check_result(const float* y, const float* y_ref, int rows, int cols) {
    const float token_tol = 2e-2f;  // per-token relative L2 error
    const float rmse_tol = 2e-3f;   // global relative RMSE
    double gsq_diff = 0.0, gsq_ref = 0.0;
    float worst_tok_rel = 0.0f;
    int worst_tok = -1;
    int bad_tokens = 0;
    for (int i = 0; i < rows; i++) {
        double sq_diff = 0.0, sq_ref = 0.0;
        for (int j = 0; j < cols; j++) {
            double ref = y_ref[(size_t)i * cols + j];
            double diff = (double)y[(size_t)i * cols + j] - ref;
            sq_diff += diff * diff;
            sq_ref += ref * ref;
        }
        float tok_rel = (float)sqrt(sq_diff / (sq_ref + 1e-12));
        if (tok_rel > worst_tok_rel) {
            worst_tok_rel = tok_rel;
            worst_tok = i;
        }
        if (tok_rel > token_tol) {
            if (bad_tokens < 10) {
                std::cerr << "Token " << i << ": relative L2 error " << tok_rel
                          << " (e.g. y[0] = " << y[(size_t)i * cols]
                          << ", expected " << y_ref[(size_t)i * cols] << ")"
                          << std::endl;
            }
            bad_tokens++;
        }
        gsq_diff += sq_diff;
        gsq_ref += sq_ref;
    }
    float rel_rmse = (float)sqrt(gsq_diff / (gsq_ref + 1e-12));
    if (bad_tokens > 0 || rel_rmse > rmse_tol) {
        std::cerr << "Result is wrong: " << bad_tokens << " / " << rows
                  << " tokens over tolerance, relative RMSE " << rel_rmse
                  << std::endl;
        exit(1);
    }
    std::cout << "Result is correct! (rel RMSE: " << rel_rmse
              << ", worst token: " << worst_tok_rel << " at " << worst_tok
              << ")" << std::endl;
}
