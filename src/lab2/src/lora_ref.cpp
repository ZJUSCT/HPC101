#include "lora.h"

#include <cmath>
#include <cstddef>
#include <random>

void init_lora_data(LoRAWeights& lora, float* g, uint64_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist_adapter(-0.1f, 0.1f);
    std::uniform_real_distribution<float> dist_g(-1.0f, 1.0f);

    for (size_t i = 0; i < (size_t)NUM_EXPERTS * LORA_RANK * D_FF; i++) {
        lora.a[i] = dist_adapter(gen);
    }
    for (size_t i = 0; i < (size_t)NUM_EXPERTS * D_MODEL * LORA_RANK; i++) {
        lora.b[i] = dist_adapter(gen);
    }
    for (size_t i = 0; i < (size_t)NUM_TOKENS * D_MODEL; i++) {
        g[i] = dist_g(gen);
    }
}

// Identical to moe_forward_ref (see src/moe_ref.cpp for the stage-by-stage
// walkthrough), plus the [LoRA] additions. The adapter pointers are null for
// the shared expert, which has no adapter.
static void expert_ffn_lora(const int8_t* w_gate, const int8_t* w_up,
                            const int8_t* w_down, float s_gate, float s_up,
                            float s_down, const float* a, const float* b,
                            const int8_t* xq, float s_x, float* out) {
    // Gate / up projections + SwiGLU activation
    float h[D_FF];
    float h_amax = 0.0f;
    for (int f = 0; f < D_FF; f++) {
        int32_t acc_g = 0;
        int32_t acc_u = 0;
        for (int d = 0; d < D_MODEL; d++) {
            acc_g += (int32_t)w_gate[f * D_MODEL + d] * (int32_t)xq[d];
            acc_u += (int32_t)w_up[f * D_MODEL + d] * (int32_t)xq[d];
        }
        float vg = (float)acc_g * (s_x * s_gate);
        float vu = (float)acc_u * (s_x * s_up);
        float silu = vg / (1.0f + expf(-vg));
        h[f] = silu * vu;
        float v = fabsf(h[f]);
        if (v > h_amax) h_amax = v;
    }

    // [LoRA] low-rank bypass reads the fp32 hidden activation:
    //   lora_mid = A @ h
    float lora_mid[LORA_RANK];
    if (a != nullptr) {
        for (int r = 0; r < LORA_RANK; r++) {
            float acc = 0.0f;
            for (int f = 0; f < D_FF; f++) {
                acc += a[r * D_FF + f] * h[f];
            }
            lora_mid[r] = acc;
        }
    }

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    int8_t hq[D_FF];
    for (int f = 0; f < D_FF; f++) {
        hq[f] = (int8_t)lrintf(h[f] / s_h);
    }

    // Down projection + [LoRA] bypass:
    //   out = (W_down @ h_q) * s_h * s_down + B @ lora_mid
    for (int d = 0; d < D_MODEL; d++) {
        int32_t acc = 0;
        for (int f = 0; f < D_FF; f++) {
            acc += (int32_t)w_down[d * D_FF + f] * (int32_t)hq[f];
        }
        out[d] = (float)acc * (s_h * s_down);
        if (b != nullptr) {
            for (int r = 0; r < LORA_RANK; r++) {
                out[d] += b[d * LORA_RANK + r] * lora_mid[r];
            }
        }
    }
}

void moe_lora_forward_ref(const float* x, const MoEWeights& w,
                          const LoRAWeights& lora, float* y) {
    for (int t = 0; t < NUM_TOKENS; t++) {
        const float* xt = x + t * D_MODEL;
        float* yt = y + t * D_MODEL;

        // 1. Affinity scores
        float s[NUM_EXPERTS];
        for (int e = 0; e < NUM_EXPERTS; e++) {
            float acc = 0.0f;
            for (int d = 0; d < D_MODEL; d++) {
                acc += w.w_router[e * D_MODEL + d] * xt[d];
            }
            s[e] = 1.0f / (1.0f + expf(-acc));
        }

        // 2. Top-K selection by biased score
        int topk_idx[TOP_K];
        bool used[NUM_EXPERTS] = {};
        for (int k = 0; k < TOP_K; k++) {
            int best = -1;
            for (int e = 0; e < NUM_EXPERTS; e++) {
                if (used[e]) continue;
                if (best < 0 || s[e] + w.bias[e] > s[best] + w.bias[best]) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
        }

        // 3. Gate values
        float gate_sum = 0.0f;
        for (int k = 0; k < TOP_K; k++) gate_sum += s[topk_idx[k]];

        // 4. Quantize the token to int8
        float x_amax = 0.0f;
        for (int d = 0; d < D_MODEL; d++) {
            float v = fabsf(xt[d]);
            if (v > x_amax) x_amax = v;
        }
        float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
        int8_t xq[D_MODEL];
        for (int d = 0; d < D_MODEL; d++) {
            xq[d] = (int8_t)lrintf(xt[d] / s_x);
        }

        // 5+6. Shared expert (no adapter), then routed experts with their
        //      LoRA adapters, combined on top of the residual connection
        float o[D_MODEL];
        expert_ffn_lora(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate, w.sh_s_up,
                        w.sh_s_down, nullptr, nullptr, xq, s_x, o);
        for (int d = 0; d < D_MODEL; d++) {
            yt[d] = xt[d] + o[d];
        }

        for (int k = 0; k < TOP_K; k++) {
            int e = topk_idx[k];
            float gate = s[e] / gate_sum;
            expert_ffn_lora(w.w_gate + (size_t)e * D_FF * D_MODEL,
                            w.w_up + (size_t)e * D_FF * D_MODEL,
                            w.w_down + (size_t)e * D_MODEL * D_FF, w.s_gate[e],
                            w.s_up[e], w.s_down[e],
                            lora.a + (size_t)e * LORA_RANK * D_FF,
                            lora.b + (size_t)e * D_MODEL * LORA_RANK, xq, s_x,
                            o);
            for (int d = 0; d < D_MODEL; d++) {
                yt[d] += gate * o[d];
            }
        }
    }
}
