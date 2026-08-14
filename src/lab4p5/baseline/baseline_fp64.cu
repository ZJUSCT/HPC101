/*
 * ============================================================================
 *  baseline_fp64.cu — FP64 GEMM baselines
 * ----------------------------------------------------------------------------
 *  gemm_fp64_cublas : cublasDgemm, the "vendor native FP64 tensor core"
 *                     path. This is BOTH the accuracy reference and the
 *                     performance reference for native FP64.

 * ============================================================================
 */
#include "gemm_api.h"
#include "utils.h"

/* -------------------------------------------------------------------------
 *  cuBLAS FP64 GEMM
 *  C[M,N] = A[M,K] @ B[K,N]   (all column-major, no transpose)
 *      m=M, n=N, k=K, lda=M, ldb=K, ldc=M
 * ------------------------------------------------------------------------- */
int gemm_fp64_cublas(int M, int N, int K,
                     const double* dA, const double* dB, double* dC,
                     cublasHandle_t handle)
{
    /* reset math mode: cublas_emulated may have left it on FP64_EMULATED */
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    const double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             dA, M,
                             dB, K,
                             &beta,
                             dC, M));
    return 0;
}

