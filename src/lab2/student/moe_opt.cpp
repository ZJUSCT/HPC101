// Main task: optimize the MoE forward pass.
<<<<<<< HEAD

#include "moe.h"

void preprocess(MoEWeights& w) {}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    moe_forward_ref(x, w, y, num_tokens);
=======
//
// This is YOUR file — it is the only file submitted for the main task, and
// everything outside student/ is replaced with the original version when
// grading. You may add helper functions, includes, global buffers, etc.,
// as long as this file implements the two functions declared in moe.h.
#include "moe.h"

// Called ONCE before timing starts. You may repack or reorder the weights
// here (real inference engines pre-pack weights offline). The input tokens
// are NOT visible here — do not precompute any part of the answer.
void preprocess(MoEWeights& w) {}

// Your optimized forward pass. Replace the reference call with your own
// implementation.
void moe_forward_optimized(const float* x, const MoEWeights& w, float* y) {
    moe_forward_ref(x, w, y);
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
}
