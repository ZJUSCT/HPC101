#pragma once

#include "cutlass/layout/tensor.h"

#define CUDA_CALL(func)                                                                             \
    {                                                                                               \
        cudaError_t e = (func);                                                                     \
        if (!(e == cudaSuccess || e == cudaErrorCudartUnloading)) {                                 \
            fprintf(stdout, "CUDA: %s:%d: error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            abort();                                                                                \
        }                                                                                           \
    }

#define CUTLASS_CALL(status)                                                                                   \
    {                                                                                                          \
        cutlass::Status error = status;                                                                        \
        if (error != cutlass::Status::kSuccess) {                                                              \
            fprintf(stdout, "CUTLASS: %s:%d: error: %s\n", __FILE__, __LINE__, cutlassGetStatusString(error)); \
            abort();                                                                                           \
        }                                                                                                      \
    }

static constexpr int alignment = 32;  // 32 byte alignment
static constexpr int size = INPUT_SIZE;
static constexpr int kernel = KERNEL_SIZE;
static constexpr int batch_size = BATCH_SIZE;
static constexpr int in_channel = IN_CHANNEL;
static constexpr int out_channel = OUT_CHANNEL;

static constexpr int N = batch_size;
static constexpr int H = size;
static constexpr int W = size;
static constexpr int C = in_channel;
static constexpr int K = out_channel;
static constexpr int R = kernel;
static constexpr int S = kernel;

static constexpr double FLOP = 2.0f * N * H * W * C * K * R * S;  // FLOP for conv2d

double getTFLOPS(double ms);

#ifdef USE_INT8
using Element = int8_t;
using ElementCompute = int;
#elif USE_HALF
using Element = cutlass::half_t;
using ElementCompute = float;
#else
#error "Unsupported data type. Define USE_INT8 or USE_HALF."
#endif

// For convenience, we assume input and output channels are aligned.
static_assert(in_channel * sizeof(Element) % alignment == 0, "Input channel must be aligned to 32 bytes");
static_assert(out_channel * sizeof(Element) % alignment == 0, "Output channel must be aligned to 32 bytes");
static_assert(kernel % 2 == 1, "Kernel size must be odd");