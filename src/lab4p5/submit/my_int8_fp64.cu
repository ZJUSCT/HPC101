#include "gemm_api.h"
#include "utils.h"
#include <cmath>

/* ====================================================================== */
/*  TODO (1): 量化一个 split                                              */
/*  用你的实现替换下面的占位代码                                          */
/* ====================================================================== */
static __global__ void quantize_split_kernel(double* residual, int8_t* q,
                                      double scale, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    /* TODO(student):
     *   double x  = residual[i];
     *   double qf = rint(x / scale);
     *   if (qf >  127.0) qf =  127.0;
     *   if (qf < -127.0) qf = -127.0;
     *   int8_t qi = (int8_t)qf;
     *   q[i]         = qi;
     *   residual[i]  = x - (double)qi * scale;
     */

    /* --- 占位代码:避免编译器报错。完成后请删除。 --- */
    q[i]        = 0;
    residual[i] = residual[i];
    (void)scale;
}

/* ====================================================================== */
/*  TODO (2): 将一次 split-pair 的 INT32 结果累加到 FP64 C               */
/*  用你的实现替换下面的占位代码                                          */
/* ====================================================================== */
static __global__ void recombine_add_kernel(const int32_t* temp, double* C,
                                     double scale, size_t mn)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mn) return;

    /* TODO(student):
     *   C[i] += scale * (double)temp[i];
     */

    /* --- 占位代码: 避免编译器报错。完成后请删除。 --- */
    (void)temp; (void)C; (void)scale;
}

/* ====================================================================== */
/*  Host侧代码: 完整实现，无需修改配合上面两个kernel即可获得正确的结果。            */
/*  可以自由修改以获得更高的性能。                                              */
/* ====================================================================== */
int gemm_my_int8_fp64(int M, int N, int K,
                      const double* dA, const double* dB, double* dC,
                      int splits, cublasHandle_t handle, cudaStream_t stream)
{
    if (splits < 1) splits = 1;
    const size_t E_A = (size_t)M * K;
    const size_t E_B = (size_t)K * N;
    const size_t E_C = (size_t)M * N;

    /* 1. 计算比例尺: sA0 = max|dA|/127, sB0 = max|dB|/127 */
    double maxA = device_maxabs_fp64(dA, E_A);
    double maxB = device_maxabs_fp64(dB, E_B);
    if (maxA == 0.0) maxA = 1.0;
    if (maxB == 0.0) maxB = 1.0;
    const double sA0 = maxA / 127.0;
    const double sB0 = maxB / 127.0;

    /* 2. 分配临时缓冲 */
    int8_t** dAq = (int8_t**)malloc(splits * sizeof(int8_t*));
    int8_t** dBq = (int8_t**)malloc(splits * sizeof(int8_t*));
    for (int s = 0; s < splits; ++s) {
        dAq[s] = (int8_t*)device_alloc(E_A);
        dBq[s] = (int8_t*)device_alloc(E_B);
    }
    double* residA = (double*)device_alloc(E_A * sizeof(double));
    double* residB = (double*)device_alloc(E_B * sizeof(double));
    int32_t* temp  = (int32_t*)device_alloc(E_C * sizeof(int32_t));

    /* 3. 拷贝 A,B 到残差缓冲（量化会原地修改）, 清零 C */
    CUDA_CHECK(cudaMemcpyAsync(residA, dA, E_A * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(residB, dB, E_B * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(dC, 0, E_C * sizeof(double), stream));

    const int block = 256;

    /* 4. 量化 A 的 splits 个 split */
    {
        int grid = (int)((E_A + block - 1) / block);
        double sA = sA0;
        for (int i = 0; i < splits; ++i) {
            quantize_split_kernel<<<grid, block, 0, stream>>>(
                residA, dAq[i], sA, E_A);
            sA = sA / 254.0;
        }
    }
    /* 量化 B 的 splits 个 split */
    {
        int grid = (int)((E_B + block - 1) / block);
        double sB = sB0;
        for (int j = 0; j < splits; ++j) {
            quantize_split_kernel<<<grid, block, 0, stream>>>(
                residB, dBq[j], sB, E_B);
            sB = sB / 254.0;
        }
    }

    /* 5. K² 次 INT8 GEMM + 重组累加 */
    cublasSetStream(handle, stream);
    const int32_t alpha_i = 1, beta_i = 0;
    double sA = sA0;
    for (int i = 0; i < splits; ++i) {
        double sB = sB0;
        for (int j = 0; j < splits; ++j) {
            /* temp = Aq[i] @ Bq[j]  (INT8 张量核, INT32 累加) */
            CUBLAS_CHECK(cublasGemmEx(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      M, N, K,
                                      &alpha_i,
                                      dAq[i], CUDA_R_8I, M,
                                      dBq[j], CUDA_R_8I, K,
                                      &beta_i,
                                      temp,  CUDA_R_32I, M,
                                      CUBLAS_COMPUTE_32I,
                                      CUBLAS_GEMM_DEFAULT));
            /* C += sA*sB * temp  (FP64 累加) */
            double scale = sA * sB;
            int grid = (int)((E_C + block - 1) / block);
            recombine_add_kernel<<<grid, block, 0, stream>>>(
                temp, dC, scale, E_C);
            sB = sB / 254.0;
        }
        sA = sA / 254.0;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* 6. 释放临时缓冲 */
    for (int s = 0; s < splits; ++s) { cudaFree(dAq[s]); cudaFree(dBq[s]); }
    free(dAq); free(dBq);
    cudaFree(residA); cudaFree(residB); cudaFree(temp);
    return 0;
}
