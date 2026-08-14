#ifndef GEMM_API_H
#define GEMM_API_H

/*
 * HPC101 Lab 4.5 — 用 INT8 张量核模拟 FP64 GEMM
 * 统一 API 头文件
 *
 * 所有实现计算列主序 GEMM:
 *   C[M×N] = A[M×K] × B[K×N]
 *
 * 内存布局（列主序）:
 *   A[k*M + m]   B[n*K + k]   C[n*M + m]
 *
 * 所有函数接收 device 指针，scratch memory 由各实现内部管理。
 * 返回值: 0 成功，非 0 为 CUDA/cuBLAS 错误码。
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- FP64 基准 ---- */

/* cuBLAS FP64 GEMM (cublasDgemm)，精度与性能参考 */
int gemm_fp64_cublas(int M, int N, int K,
                     const double* dA, const double* dB, double* dC,
                     cublasHandle_t handle);

/* 朴素实现：逐 split 量化 + cuBLAS INT8 GEMM + 逐次重组，未优化 */
int gemm_int8_cublas_baseline(int M, int N, int K,
                              const double* dA, const double* dB, double* dC,
                              int splits, cublasHandle_t handle,
                              cudaStream_t stream);

/* 学生实现 */
int gemm_my_int8_fp64(int M, int N, int K,
                      const double* dA, const double* dB, double* dC,
                      int splits, cublasHandle_t handle, cudaStream_t stream);

/* cuBLAS 自带 FP64 模拟 (CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT)，基线 */
int gemm_cublas_emulated(int M, int N, int K,
                         const double* dA, const double* dB, double* dC,
                         int splits, cublasHandle_t handle, void* workspace,
                         size_t workspace_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_API_H */
