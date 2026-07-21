#include "moe.h"

#include <cmath>
#include <cstddef>

<<<<<<< HEAD
// Reference scalar MoE forward pass (DeepSeek-V3 style)
=======
// Reference scalar MoE forward pass (DeepSeek-V3 style), one token at a time.
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
//
// Per-token pipeline:
//   1. Affinity:  s_e = sigmoid(w_router[e] . x_t)                  (fp32)
//   2. Top-K:     select by (s_e + bias_e); the bias balances load
//                 and is used for SELECTION ONLY                    (fp32)
//   3. Gates:     g_e = s_e / sum of selected s (bias NOT included) (fp32)
//   4. Quantize:  x_q = round(x_t / s_x), s_x = max|x_t| / 127      (fp32 -> int8)
//   5. Expert FFN (shared expert + each selected routed expert), SwiGLU:
//        hg  = (W_gate @ x_q) * s_x * s_gate                        (int8 dot -> int32 -> fp32)
//        hu  = (W_up   @ x_q) * s_x * s_up                          (int8 dot -> int32 -> fp32)
//        h   = SiLU(hg) * hu,  SiLU(v) = v / (1 + exp(-v))          (fp32)
//        h_q = round(h / s_h), s_h = max|h| / 127                   (fp32 -> int8)
//        o   = (W_down @ h_q) * s_h * s_down                        (int8 dot -> int32 -> fp32)
//   6. Combine:   y_t = x_t + o_shared + sum of g_e * o_e           (fp32)
<<<<<<< HEAD

// One SwiGLU expert on one already-quantized token
static void expert_ffn(const int8_t* w_gate, const int8_t* w_up,
                       const int8_t* w_down, float s_gate, float s_up,
                       float s_down, const int8_t* xq, float s_x, float* out,
                       int d_model, int d_ff) {
    // Gate / up projections + SwiGLU activation
    float h[MAX_D_FF];
    float h_amax = 0.0f;
    for (int f = 0; f < d_ff; f++) {
        int32_t acc_g = 0;
        int32_t acc_u = 0;
        for (int d = 0; d < d_model; d++) {
            acc_g += (int32_t)w_gate[f * d_model + d] * (int32_t)xq[d];
            acc_u += (int32_t)w_up[f * d_model + d] * (int32_t)xq[d];
        }
        float vg = (float)acc_g * (s_x * s_gate);
        float vu = (float)acc_u * (s_x * s_up);
        float silu = vg / (1.0f + expf(-vg));
        h[f] = silu * vu;
        float a = fabsf(h[f]);
        if (a > h_amax) h_amax = a;
    }

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    int8_t hq[MAX_D_FF];
    for (int f = 0; f < d_ff; f++) {
        hq[f] = (int8_t)lrintf(h[f] / s_h);
    }

    // Down projection
    for (int d = 0; d < d_model; d++) {
        int32_t acc = 0;
        for (int f = 0; f < d_ff; f++) {
            acc += (int32_t)w_down[d * d_ff + f] * (int32_t)hq[f];
        }
        out[d] = (float)acc * (s_h * s_down);
    }
}

void moe_forward_ref(const float* x, const MoEWeights& w, float* y,
                     int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    for (int t = 0; t < num_tokens; t++) {
        const float* xt = x + (size_t)t * d_model;
        float* yt = y + (size_t)t * d_model;

        // 1. Affinity scores
        float s[MAX_NUM_EXPERTS];
        for (int e = 0; e < num_experts; e++) {
=======

// One SwiGLU expert on one already-quantized token
static void expert_ffn(const int8_t* w_gate, const int8_t* w_up,
                       const int8_t* w_down, float s_gate, float s_up,
                       float s_down, const int8_t* xq, float s_x, float* out) {
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
        float a = fabsf(h[f]);
        if (a > h_amax) h_amax = a;
    }

    // Requantize hidden activation to int8
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    int8_t hq[D_FF];
    for (int f = 0; f < D_FF; f++) {
        hq[f] = (int8_t)lrintf(h[f] / s_h);
    }

    // Down projection
    for (int d = 0; d < D_MODEL; d++) {
        int32_t acc = 0;
        for (int f = 0; f < D_FF; f++) {
            acc += (int32_t)w_down[d * D_FF + f] * (int32_t)hq[f];
        }
        out[d] = (float)acc * (s_h * s_down);
    }
}

void moe_forward_ref(const float* x, const MoEWeights& w, float* y) {
    for (int t = 0; t < NUM_TOKENS; t++) {
        const float* xt = x + t * D_MODEL;
        float* yt = y + t * D_MODEL;

        // 1. Affinity scores
        float s[NUM_EXPERTS];
        for (int e = 0; e < NUM_EXPERTS; e++) {
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
            float acc = 0.0f;
            for (int d = 0; d < d_model; d++) {
                acc += w.w_router[(size_t)e * d_model + d] * xt[d];
            }
            s[e] = 1.0f / (1.0f + expf(-acc));
        }

        // 2. Top-K selection by biased score (ties broken by smaller index)
<<<<<<< HEAD
        int topk_idx[MAX_TOP_K];
        bool used[MAX_NUM_EXPERTS] = {};
        for (int k = 0; k < top_k; k++) {
            int best = -1;
            for (int e = 0; e < num_experts; e++) {
=======
        int topk_idx[TOP_K];
        bool used[NUM_EXPERTS] = {};
        for (int k = 0; k < TOP_K; k++) {
            int best = -1;
            for (int e = 0; e < NUM_EXPERTS; e++) {
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
                if (used[e]) continue;
                if (best < 0 || s[e] + w.bias[e] > s[best] + w.bias[best]) {
                    best = e;
                }
            }
            used[best] = true;
            topk_idx[k] = best;
        }

        // 3. Gate values: normalize the ORIGINAL affinities of the selected
        //    experts (the bias never enters the gate values)
        float gate_sum = 0.0f;
<<<<<<< HEAD
        for (int k = 0; k < top_k; k++) gate_sum += s[topk_idx[k]];
=======
        for (int k = 0; k < TOP_K; k++) gate_sum += s[topk_idx[k]];
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)

        // 4. Quantize the token to int8 (symmetric, per-token scale)
        float x_amax = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float a = fabsf(xt[d]);
            if (a > x_amax) x_amax = a;
        }
        float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
        int8_t xq[MAX_D_MODEL];
        for (int d = 0; d < d_model; d++) {
            xq[d] = (int8_t)lrintf(xt[d] / s_x);
        }

        // 5+6. Shared expert (always on), then selected routed experts,
        //      combined on top of the residual connection
<<<<<<< HEAD
        float o[MAX_D_MODEL];
        expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate, w.sh_s_up,
                   w.sh_s_down, xq, s_x, o, d_model, d_ff);
        for (int d = 0; d < d_model; d++) {
            yt[d] = xt[d] + o[d];
        }

        for (int k = 0; k < top_k; k++) {
            int e = topk_idx[k];
            float gate = s[e] / gate_sum;
            expert_ffn(w.w_gate + (size_t)e * d_ff * d_model,
                       w.w_up + (size_t)e * d_ff * d_model,
                       w.w_down + (size_t)e * d_model * d_ff, w.s_gate[e],
                       w.s_up[e], w.s_down[e], xq, s_x, o, d_model, d_ff);
            for (int d = 0; d < d_model; d++) {
=======
        float o[D_MODEL];
        expert_ffn(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate, w.sh_s_up,
                   w.sh_s_down, xq, s_x, o);
        for (int d = 0; d < D_MODEL; d++) {
            yt[d] = xt[d] + o[d];
        }

        for (int k = 0; k < TOP_K; k++) {
            int e = topk_idx[k];
            float gate = s[e] / gate_sum;
            expert_ffn(w.w_gate + (size_t)e * D_FF * D_MODEL,
                       w.w_up + (size_t)e * D_FF * D_MODEL,
                       w.w_down + (size_t)e * D_MODEL * D_FF, w.s_gate[e],
                       w.s_up[e], w.s_down[e], xq, s_x, o);
            for (int d = 0; d < D_MODEL; d++) {
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
                yt[d] += gate * o[d];
            }
        }
    }
}
