#ifndef _MOE_H_
#define _MOE_H_

#include <cstdint>

<<<<<<< HEAD
#define MAX_NUM_TOKENS 1024
#define MAX_D_MODEL 1024
#define MAX_D_FF 512
#define MAX_NUM_EXPERTS 512
#define MAX_TOP_K 4
=======
// Problem size
// A scaled-down DeepSeek-V3 style MoE layer processing one batch of tokens:
//   NUM_TOKENS  tokens, each a D_MODEL-dim fp32 hidden vector
//   NUM_EXPERTS fine-grained routed experts, each a SwiGLU FFN
//               (D_MODEL -> D_FF -> D_MODEL), plus ONE shared expert of the
//               same shape that every token passes through
//   Each token is routed to its TOP_K routed experts
#define NUM_TOKENS 128
#define D_MODEL 256
#define D_FF 128
#define NUM_EXPERTS 16
#define TOP_K 4
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)

// Mixed-precision weights:
//   Router runs in fp32; all expert FFNs use int8 weights with a fp32
//   dequantization scale per matrix (weight_fp32 = weight_int8 * scale)
struct MoEWeights {
<<<<<<< HEAD
    // Model shape
    int d_model;
    int d_ff;
    int num_experts;
    int top_k;
    // Router
    float* w_router;  // [num_experts][d_model]
    float* bias;      // [num_experts], load-balancing bias: added to the
=======
    // Router
    float* w_router;  // [NUM_EXPERTS][D_MODEL]
    float* bias;      // [NUM_EXPERTS], load-balancing bias: added to the
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
                      // affinity score for Top-K SELECTION ONLY, never used
                      // in the gate values (DeepSeek-V3 auxiliary-loss-free
                      // load balancing)
    // Routed experts (SwiGLU)
<<<<<<< HEAD
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
=======
    int8_t* w_gate;   // [NUM_EXPERTS][D_FF][D_MODEL]
    int8_t* w_up;     // [NUM_EXPERTS][D_FF][D_MODEL]
    int8_t* w_down;   // [NUM_EXPERTS][D_MODEL][D_FF]
    float* s_gate;    // [NUM_EXPERTS]
    float* s_up;      // [NUM_EXPERTS]
    float* s_down;    // [NUM_EXPERTS]
    // Shared expert (same shape as one routed expert)
    int8_t* sh_gate;  // [D_FF][D_MODEL]
    int8_t* sh_up;    // [D_FF][D_MODEL]
    int8_t* sh_down;  // [D_MODEL][D_FF]
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)
    float sh_s_gate;
    float sh_s_up;
    float sh_s_down;
};

// --------------------------------------------------------------------------
<<<<<<< HEAD
// Framework
// --------------------------------------------------------------------------

// Fill weights (and one token batch) with reproducible random data (w's
// shape fields must be set by the caller beforehand)
void init_data(float* x, MoEWeights& w, int num_tokens, uint64_t seed);

void init_tokens(float* x, int num_tokens, int d_model, uint64_t seed);
=======
// Framework (replaced with the original version when grading)
// --------------------------------------------------------------------------

// Fill tokens and weights with reproducible random data
void init_data(float* x, MoEWeights& w, uint64_t seed);
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)

// Reference scalar implementation (baseline)
void moe_forward_ref(const float* x, const MoEWeights& w, float* y,
                     int num_tokens);

// Compare y against y_ref element-wise with fp tolerance; exits on mismatch
void check_result(const float* y, const float* y_ref, int rows, int cols);

// --------------------------------------------------------------------------
// Your code (student/moe_opt.cpp)
// --------------------------------------------------------------------------

<<<<<<< HEAD
// Called ONCE before timing starts. You may repack or reorder the weights here.
void preprocess(MoEWeights& w);

// Your optimized forward pass.
void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens);
=======
// Called ONCE before timing starts. You may repack or reorder the weights
// here (real inference engines pre-pack weights offline). The input tokens
// are NOT visible here — do not precompute any part of the answer.
void preprocess(MoEWeights& w);

// Your optimized forward pass
void moe_forward_optimized(const float* x, const MoEWeights& w, float* y);
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)

#endif
