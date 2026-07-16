// Main task: optimize the MoE forward pass.

#include "moe.h"

void preprocess(MoEWeights& w) {}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    moe_forward_ref(x, w, y, num_tokens);
}
