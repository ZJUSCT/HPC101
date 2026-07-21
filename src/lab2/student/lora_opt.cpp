// Bonus: QLoRA-style fine-tuning of the frozen int8 MoE layer.
//
// This is YOUR file — it is the only file submitted for the bonus, and
// everything outside student/ is replaced with the original version when
// grading. You may add helper functions, includes, global buffers, etc.,
// as long as this file implements the two functions declared in lora.h.
#include "lora.h"

#include <cmath>
#include <cstddef>

// Your optimized forward pass with the LoRA bypass. Replace the reference
// call with your own implementation.
void moe_lora_forward_optimized(const float* x, const MoEWeights& w,
                                const LoRAWeights& lora, float* y) {
    moe_lora_forward_ref(x, w, lora, y);
}

// Backward pass: given the upstream gradient g = dL/dy (same shape as y),
// accumulate the gradients of the adapters. The base weights are frozen —
// you do NOT need dW / dx.
//   grad_a : dL/dA, layout [NUM_EXPERTS][LORA_RANK][D_FF]
//   grad_b : dL/dB, layout [NUM_EXPERTS][D_MODEL][LORA_RANK]
// Both are zero-initialized by the caller. You will need intermediates of
// the forward pass (routing, h, A @ h) — recompute them here.
void moe_lora_backward(const float* x, const MoEWeights& w,
                       const LoRAWeights& lora, const float* g, float* grad_a,
                       float* grad_b) {
    // TODO: implement me
}
