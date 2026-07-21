#include "lora.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>

// Central finite differences: perturbing one adapter coordinate only moves
// the fp32 LoRA bypass (routing, quantization and the int8 base path do not
// depend on A/B), so L(A, B) is smooth and the check is reliable. L is in
// fact linear in each individual coordinate, so a large step has no
// truncation error and drowns out fp32 rounding noise.
#define FD_SAMPLES 16  // sampled coordinates per adapter matrix
#define FD_EPS 1e-1f

namespace {

double loss(const float* x, const MoEWeights& w, const LoRAWeights& lora,
            const float* g, float* y_scratch) {
    moe_lora_forward_ref(x, w, lora, y_scratch);
    double l = 0.0;
    for (size_t i = 0; i < (size_t)NUM_TOKENS * D_MODEL; i++) {
        l += (double)g[i] * (double)y_scratch[i];
    }
    return l;
}

}  // namespace

bool check_gradients(const float* x, const MoEWeights& w, LoRAWeights& lora,
                     const float* g, const float* grad_a, const float* grad_b) {
    float* y_scratch = new float[NUM_TOKENS * D_MODEL];
    std::mt19937 gen(7);

    struct Target {
        const char* name;
        float* param;
        const float* grad;
        size_t size;
    };
    Target targets[2] = {
        {"grad_a", lora.a, grad_a, (size_t)NUM_EXPERTS * LORA_RANK * D_FF},
        {"grad_b", lora.b, grad_b, (size_t)NUM_EXPERTS * D_MODEL * LORA_RANK},
    };

    int failures = 0;
    double max_err = 0.0;
    for (const Target& tgt : targets) {
        std::uniform_int_distribution<size_t> dist(0, tgt.size - 1);
        for (int i = 0; i < FD_SAMPLES; i++) {
            size_t idx = dist(gen);
            float saved = tgt.param[idx];
            tgt.param[idx] = saved + FD_EPS;
            double l_plus = loss(x, w, lora, g, y_scratch);
            tgt.param[idx] = saved - FD_EPS;
            double l_minus = loss(x, w, lora, g, y_scratch);
            tgt.param[idx] = saved;

            double fd = (l_plus - l_minus) / (2.0 * FD_EPS);
            double got = tgt.grad[idx];
            double err = fabs(fd - got) / std::max({fabs(fd), fabs(got), 1e-3});
            if (err > max_err) max_err = err;
            if (err > 2e-2) {
                if (failures < 10) {
                    std::cerr << "Gradient error in " << tgt.name << "[" << idx
                              << "]: expected " << fd << " (finite difference)"
                              << ", got " << got << std::endl;
                }
                failures++;
            }
        }
    }

    delete[] y_scratch;

    if (failures > 0) {
        std::cerr << "Gradient check FAILED: " << failures << " / "
                  << 2 * FD_SAMPLES << " sampled coordinates mismatched"
                  << std::endl;
        return false;
    }
    std::cout << "Gradient check passed! (" << 2 * FD_SAMPLES
              << " sampled coordinates, max error: " << max_err << ")"
              << std::endl;
    return true;
}
