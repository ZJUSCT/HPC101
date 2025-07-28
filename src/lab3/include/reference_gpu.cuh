#pragma once
#include <chrono>
#include "cuda.h"
#include "common.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"

using LayoutA = cutlass::layout::TensorNHWC;  // NHWC input tensor
using LayoutW = cutlass::layout::TensorNHWC;  // KRSC kernel tensor
using LayoutB = cutlass::layout::TensorNHWC;  // NHWK output tensor

template <typename Element, typename ElementCompute>
struct CutlassConv2d {
    static_assert("Element must be int8_t or half.");
};

// Some bugs in CUTLASS prevent us using int8_t

// template <>
// struct CutlassConv2d<int8_t, int> {
//     using Element = int8_t;
//     using ElementCompute = int;
//     using MMAOp = cutlass::arch::OpClassSimt;
//     using SmArch = cutlass::arch::Sm70;
//     using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
//     using WarpShape = cutlass::gemm::GemmShape<32, 64, 128>;
//     using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
//     using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
//     static constexpr int NumStages = 2;
//     using MathOperatorTag = cutlass::arch::OpMultiplyAdd;
//     using EpilogueOp = cutlass::epilogue::thread::LinearCombination<Element, 1, ElementCompute, ElementCompute>;
//     using Conv2dFprop =
//         cutlass::conv::kernel::DefaultConv2dFprop<Element, LayoutA, Element, LayoutW, Element, LayoutB, ElementCompute,
//                                                   MMAOp, SmArch, ThreadblockShape, WarpShape, InstructionShape,
//                                                   EpilogueOp, SwizzleThreadBlock, NumStages, MathOperatorTag>;
//     using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFprop::Kernel>;
// };

template <>
struct CutlassConv2d<cutlass::half_t, float> {
    using Element = cutlass::half_t;
    using ElementCompute = float;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm70;
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    static constexpr int NumStages = 2;
    using MathOperatorTag = cutlass::arch::OpMultiplyAdd;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<Element, 8, ElementCompute, ElementCompute>;
    using Conv2dFprop =
        cutlass::conv::kernel::DefaultConv2dFprop<Element, LayoutA, Element, LayoutW, Element, LayoutB, ElementCompute,
                                                  MMAOp, SmArch, ThreadblockShape, WarpShape, InstructionShape,
                                                  EpilogueOp, SwizzleThreadBlock, NumStages, MathOperatorTag>;
    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFprop::Kernel>;
};

template <typename Element, typename ElementCompute>
void conv2d_gpu_ref(const Element *a, const Element *w, Element *b) {
    Element *d_a, *d_w, *d_b;
    CUDA_CALL(cudaMalloc(&d_a, N * H * W * C * sizeof(Element)));
    CUDA_CALL(cudaMemcpy(d_a, a, N * H * W * C * sizeof(Element), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_w, K * R * S * C * sizeof(Element)));
    CUDA_CALL(cudaMemcpy(d_w, w, K * R * S * C * sizeof(Element), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_b, N * H * W * K * sizeof(Element)));

    cutlass::Tensor4DCoord input_size(N, H, W, C);
    cutlass::Tensor4DCoord kernel_size(K, R, S, C);
    cutlass::Tensor4DCoord output_size(N, H, W, K);

    auto layout_a = cutlass::layout::TensorNHWC::packed(input_size);
    auto layout_w = cutlass::layout::TensorNHWC::packed(kernel_size);
    auto layout_b = cutlass::layout::TensorNHWC::packed(output_size);

    auto tensor_a = cutlass::make_TensorView(d_a, layout_a, input_size);
    auto tensor_w = cutlass::make_TensorView(d_w, layout_w, kernel_size);
    auto tensor_b = cutlass::make_TensorView(d_b, layout_b, output_size);

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    cutlass::conv::Conv2dProblemSize problem_size(N, H, W, C, H, W, K, R, S, mode);

    using Conv2d = typename CutlassConv2d<Element, ElementCompute>::ImplicitGemm;

    typename Conv2d::Arguments args{
        problem_size,
        tensor_a,
        tensor_w,
        tensor_b,
        tensor_b,
        {ElementCompute(1), ElementCompute(0)}  // alpha, beta  
    };

    Conv2d conv2d_op;

    size_t workspace_size = conv2d_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CALL(conv2d_op.can_implement(args));
    CUTLASS_CALL(conv2d_op.initialize(args, workspace.get()));
    
    cudaEvent_t start_e, stop_e;
    cudaEventCreate(&start_e);
    cudaEventCreate(&stop_e);
    
    cudaEventRecord(start_e);
    CUTLASS_CALL(conv2d_op());
    CUDA_CALL(cudaGetLastError());
    // Stop Timer
    cudaEventRecord(stop_e);
    cudaEventSynchronize(stop_e);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_e, stop_e);
    std::cout << "CUTLASS GPU Conv2D: \n" << milliseconds << " ms\n";
    std::cout << std::fixed << std::setprecision(4) << getTFLOPS(milliseconds) << " TFLOPS\n";

    CUDA_CALL(cudaMemcpy(b, d_b, N * H * W * K * sizeof(Element), cudaMemcpyDeviceToHost));
    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_b);
}
