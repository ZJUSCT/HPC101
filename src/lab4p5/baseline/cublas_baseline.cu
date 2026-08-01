/*
 * ============================================================================
 *  cublas_baseline.cu — INT8 emulation of FP64 via cuBLAS INT8 GEMM
 * ----------------------------------------------------------------------------
 *  Reference "easy path" implementation. Deliberately UNOPTIMIZED so that
 *  students have something concrete to beat:
 *
 *    * one separate quantize kernel launch per split (K launches, each reads
 *      & writes the whole residual buffer);
 *    * INT8 -> INT32 cuBLAS GEMM (CUBLAS_COMPUTE_32I), then a separate
 *      FP64 recombine-add kernel per split pair (K*K launches);
 *    * a single reusable INT32 scratch buffer (small memory footprint, at
 *      the cost of K*K extra global-memory passes over C).
 *
 *  Algorithm (successive-approximation / "Ozaki-style" split):
 *    A ~= sum_i  sA[i] * Aq[i]      (Aq[i] : int8,  sA[i] : double)
 *    B ~= sum_j  sB[j] * Bq[j]
 *    C   = sum_{i,j} sA[i]*sB[j] * (Aq[i] @ Bq[j])
 *
 *  With a geometric scale sA[i] = sA[0]/254^i (and sA[0] = max|A|/127), every
 *  int8 component is guaranteed to lie in [-127,127] with no per-split
 *  reduction. Each split recovers ~7 bits of mantissa.
 *
 *  NOTE: cuBLAS INT8 GEMM requires leading dimensions (here M and K) to be
 *  multiples of 4 and 16-byte aligned pointers. The benchmark uses
 *  matrix dimensions that are multiples of 16.
 * ============================================================================
 */
#include "gemm_api.h"
#include "utils.h"
#include <cmath>

/* round-to-nearest-even via rint(); clamp to int8 range [-127,127] */
__device__ __forceinline__ int8_t quantize_to_int8(double x, double scale)
{
    double q = rint(x / scale);
    if (q > 127.0) q = 127.0;
    if (q < -127.0) q = -127.0;
    return (int8_t)q;
}

/* -------------------------------------------------------------------------
 *  Naive per-split quantize kernel.
 *  Reads residual[e], writes int8 q[e], and updates residual[e] IN-PLACE
 *  to residual[e] - q[e]*scale  (preparing the next split).
 * ------------------------------------------------------------------------- */
static __global__ void quantize_one_split_kernel(double* residual, int8_t* q,
                                          double scale, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = residual[i];
    int8_t qi = quantize_to_int8(x, scale);
    q[i] = qi;
    residual[i] = x - (double)qi * scale;
}

/* -------------------------------------------------------------------------
 *  Naive recombine-add kernel.
 *  C[m,n] += scale * (double)temp[m,n]    (temp is the INT8@INT8 INT32 output)
 * ------------------------------------------------------------------------- */
static __global__ void recombine_add_kernel(const int32_t* temp, double* C,
                                      double scale, size_t mn)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mn) return;
    C[i] += scale * (double)temp[i];
}

int gemm_int8_cublas_baseline(int M, int N, int K,
                              const double* dA, const double* dB, double* dC,
                              int splits, cublasHandle_t handle,
                              cudaStream_t stream)
{
    if (splits < 1) splits = 1;
    const size_t E_A = (size_t)M * K;
    const size_t E_B = (size_t)K * N;
    const size_t E_C = (size_t)M * N;

    /* --- 1. per-matrix max abs -> first scale (geometric sequence follows) - */
    double maxA = device_maxabs_fp64(dA, E_A);
    double maxB = device_maxabs_fp64(dB, E_B);
    if (maxA == 0.0) maxA = 1.0;
    if (maxB == 0.0) maxB = 1.0;
    const double sA0 = maxA / 127.0;
    const double sB0 = maxB / 127.0;

    /* --- 2. allocate scratch ------------------------------------------------ */
    /* splits 个 int8 buffers for A and B, plus FP64 residual buffers, plus one INT32
     * scratch for the GEMM output that is reused across split pairs. */
    int8_t** dAq = (int8_t**)malloc(splits * sizeof(int8_t*));
    int8_t** dBq = (int8_t**)malloc(splits * sizeof(int8_t*));
    for (int s = 0; s < splits; ++s) {
        dAq[s] = (int8_t*)device_alloc(E_A);
        dBq[s] = (int8_t*)device_alloc(E_B);
    }
    double* residA = (double*)device_alloc(E_A * sizeof(double));
    double* residB = (double*)device_alloc(E_B * sizeof(double));
    int32_t* temp  = (int32_t*)device_alloc(E_C * sizeof(int32_t));
    if (!residA || !residB || !temp) { return 1; }

    /* copy A,B into residuals (we modify them in place) */
    CUDA_CHECK(cudaMemcpyAsync(residA, dA, E_A * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(residB, dB, E_B * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));

    /* zero the output C (we accumulate into it) */
    CUDA_CHECK(cudaMemsetAsync(dC, 0, E_C * sizeof(double), stream));

    /* --- 3. quantize each split (sequential dependency on residual) ------- */
    const int block = 256;
    {
        size_t n = E_A;
        int grid = (int)((n + block - 1) / block);
        double sA = sA0;
        for (int i = 0; i < splits; ++i) {
            quantize_one_split_kernel<<<grid, block, 0, stream>>>(
                residA, dAq[i], sA, n);
            sA = sA / 254.0;
        }
    }
    {
        size_t n = E_B;
        int grid = (int)((n + block - 1) / block);
        double sB = sB0;
        for (int j = 0; j < splits; ++j) {
            quantize_one_split_kernel<<<grid, block, 0, stream>>>(
                residB, dBq[j], sB, n);
            sB = sB / 254.0;
        }
    }

    /* --- 4. K*K INT8 GEMMs + recombine ------------------------------------- */
    cublasSetStream(handle, stream);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    const int32_t alpha_i = 1;
    const int32_t beta_i  = 0;

    double sA = sA0;
    for (int i = 0; i < splits; ++i) {
        double sB = sB0;
        for (int j = 0; j < splits; ++j) {
            /* temp = Aq[i] @ Bq[j]   (INT8 tensor cores, INT32 accumulate) */
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
            /* C += sA*sB * temp   (FP64 accumulation) */
            double scale = sA * sB;
            size_t mn = E_C;
            int grid = (int)((mn + block - 1) / block);
            recombine_add_kernel<<<grid, block, 0, stream>>>(
                temp, dC, scale, mn);
            sB = sB / 254.0;
        }
        sA = sA / 254.0;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* --- 5. cleanup -------------------------------------------------------- */
    for (int s = 0; s < splits; ++s) { cudaFree(dAq[s]); cudaFree(dBq[s]); }
    free(dAq); free(dBq);
    cudaFree(residA); cudaFree(residB); cudaFree(temp);
    return 0;
}
