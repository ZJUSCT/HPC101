#ifndef _MOE_H_
#define _MOE_H_

#include <cstdint>

#define MAX_NUM_TOKENS 1024
#define MAX_D_MODEL 1024
#define MAX_D_FF 512
#define MAX_NUM_EXPERTS 512
#define MAX_TOP_K 4

// Mixed-precision weights:
//   Router runs in fp32; all expert FFNs use int8 weights with a fp32
//   dequantization scale per matrix (weight_fp32 = weight_int8 * scale)
struct MoEWeights {
    // Model shape
    int d_model;
    int d_ff;
    int num_experts;
    int top_k;
    // Router
    float* w_router;  // [num_experts][d_model]
    float* bias;      // [num_experts], load-balancing bias: added to the
                      // affinity score for Top-K SELECTION ONLY, never used
                      // in the gate values (DeepSeek-V3 auxiliary-loss-free
                      // load balancing)
    // Routed experts (SwiGLU)
    int8_t* w_gate;   // [num_experts][d_ff][d_model]
    int8_t* w_up;     // [num_experts][d_ff][d_model]
    int8_t* w_down;   // [num_experts][d_model][d_ff]
    float* s_gate;    // [num_experts]
    float* s_up;      // [num_experts]
    float* s_down;    // [num_experts]
    // Shared expert (same shape as one routed expert)
    int8_t* sh_gate;  // [d_ff][d_model]
    int8_t* sh_up;    // [d_ff][d_model]
    int8_t* sh_down;  // [d_model][d_ff]
    float sh_s_gate;
    float sh_s_up;
    float sh_s_down;
};

// --------------------------------------------------------------------------
// Framework
// --------------------------------------------------------------------------

// Fill weights (and one token batch) with reproducible random data (w's
// shape fields must be set by the caller beforehand)
void init_data(float* x, MoEWeights& w, int num_tokens, uint64_t seed);

void init_tokens(float* x, int num_tokens, int d_model, uint64_t seed);

// Reference scalar implementation (baseline)
void moe_forward_ref(const float* x, const MoEWeights& w, float* y,
                     int num_tokens);

// Compare y against y_ref element-wise with fp tolerance; exits on mismatch
void check_result(const float* y, const float* y_ref, int rows, int cols);

// --------------------------------------------------------------------------
// Your code (student/moe_opt.cpp)
// --------------------------------------------------------------------------

// Called ONCE before timing starts. You may repack or reorder the weights here.
void preprocess(MoEWeights& w);

// Your optimized forward pass.
void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens);

#endif
