/*
 *  cublas_emulated.cu — cuBLAS 自带的 FP64-via-INT8 模拟（Ozaki scheme）
 *  调用 cublasGemmEx with CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT。这是
 *  NVIDIA 库自己实现的"FP64 模拟"，是端到端性能对比的基线。
 */
#include "gemm_api.h"
#include "utils.h"

int gemm_cublas_emulated(int M, int N, int K,
                         const double* dA, const double* dB, double* dC,
                         int splits, cublasHandle_t handle, void* workspace,
                         size_t workspace_bytes, cudaStream_t stream)
{
    if (splits < 1) splits = 1;
    if (workspace && workspace_bytes > 0) {
        CUBLAS_CHECK(cublasSetWorkspace(handle, workspace, workspace_bytes));
    }
    cublasSetStream(handle, stream);
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
    CUBLAS_CHECK(cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
    /* splits 控制模拟精度：每个 split 约贡献 8 bit，封顶 55（cuBLAS 的上限，对应 7 个 int8 切片）*/
    int max_bits = 8 * splits;
    if (max_bits > 55) max_bits = 55;
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(handle, max_bits));

    const double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K,
                              &alpha,
                              dA, CUDA_R_64F, M,
                              dB, CUDA_R_64F, K,
                              &beta,
                              dC, CUDA_R_64F, M,
                              CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
                              CUBLAS_GEMM_DEFAULT));
    return 0;
}
