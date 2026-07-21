#ifndef _LORA_H_
#define _LORA_H_

#include "moe.h"

#define LORA_RANK 8

// Bonus: QLoRA-style fine-tuning of the frozen int8 MoE layer.
//
// Each ROUTED expert's down projection gets a rank-LORA_RANK fp32 adapter
// that bypasses the quantized path, reading the fp32 hidden activation h
// (the SwiGLU product, before requantization):
//
//   o_e = (W_down[e] @ h_q) * s_h * s_down[e]  +  B[e] @ (A[e] @ h)
//
// The base weights stay frozen; only A and B are trained. The shared expert
// has no adapter.
struct LoRAWeights {
    float* a;  // [NUM_EXPERTS][LORA_RANK][D_FF]
    float* b;  // [NUM_EXPERTS][D_MODEL][LORA_RANK]
};

// --------------------------------------------------------------------------
// Framework (replaced with the original version when grading)
// --------------------------------------------------------------------------

// Fill adapters and the upstream gradient g with reproducible random data
void init_lora_data(LoRAWeights& lora, float* g, uint64_t seed);

// Reference scalar forward with the LoRA bypass (baseline; also the exact
// specification your backward pass must match)
void moe_lora_forward_ref(const float* x, const MoEWeights& w,
                          const LoRAWeights& lora, float* y);

// Verify grad_a / grad_b against central finite differences of the loss
//   L(A, B) = sum(g * y),  y = moe_lora_forward_ref(x, w, {A, B})
// on a sample of coordinates. Returns true if all sampled coordinates match.
// (Perturbs the adapters during the check and restores them afterwards.)
bool check_gradients(const float* x, const MoEWeights& w, LoRAWeights& lora,
                     const float* g, const float* grad_a, const float* grad_b);

// --------------------------------------------------------------------------
// Your code (student/lora_opt.cpp)
// --------------------------------------------------------------------------

// Your optimized forward pass with the LoRA bypass
void moe_lora_forward_optimized(const float* x, const MoEWeights& w,
                                const LoRAWeights& lora, float* y);

// Backward pass: given the upstream gradient g = dL/dy (same shape as y),
// accumulate the gradients of the adapters. The base weights are frozen —
// you do NOT need dW / dx.
//   grad_a : dL/dA, layout [NUM_EXPERTS][LORA_RANK][D_FF]
//   grad_b : dL/dB, layout [NUM_EXPERTS][D_MODEL][LORA_RANK]
// Both are zero-initialized by the caller.
void moe_lora_backward(const float* x, const MoEWeights& w,
                       const LoRAWeights& lora, const float* g,
                       float* grad_a, float* grad_b);

#endif
