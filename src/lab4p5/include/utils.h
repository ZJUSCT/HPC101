#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdint>

/* ----------------------------------------------------------------------- */
/*  Error-checking macros                                                  */
/* ----------------------------------------------------------------------- */

#define CUDA_CHECK(call)                                                     \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s (%s)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorName(_e),               \
                    cudaGetErrorString(_e));                                \
            return (int)_e;                                                 \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _s = (call);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",              \
                    __FILE__, __LINE__, (int)_s);                           \
            return (int)(0x10000 + (int)_s);                                 \
        }                                                                   \
    } while (0)

/* ----------------------------------------------------------------------- */
/*  Timing                                                                 */
/* ----------------------------------------------------------------------- */

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    void end(cudaStream_t s = 0)   { cudaEventRecord(stop, s); }
    /* returns milliseconds (blocks until stop recorded) */
    float elapsed_ms() {
        float ms = 0.f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

/* ----------------------------------------------------------------------- */
/*  Accuracy metrics                                                       */
/* ----------------------------------------------------------------------- */

struct AccuracyReport {
    double max_abs_err;   /* max |C_ref - C_test|                          */
    double max_rel_err;   /* max |C_ref - C_test| / |C_ref| over non-tiny  */
    double l2_rel_err;    /* ||C_ref - C_test||_2 / ||C_ref||_2            */
};

/* Compare a test (device, FP64) GEMM output against a reference (device, FP64)
 * output. Both are M*N column-major. */
void accuracy_compare(const double* d_ref, const double* d_test,
                      int M, int N, AccuracyReport* rep);

/* ----------------------------------------------------------------------- */
/*  Matrix helpers                                                         */
/* ----------------------------------------------------------------------- */

/* Fill a host buffer with a deterministic, reproducible FP64 matrix in a sane
 * range (values uniformly in [-1, 1] with a fixed seed). */
void fill_matrix(double* h, size_t n, uint64_t seed);

/* Allocate a device buffer of `bytes` bytes and return the device pointer.
 * Returns nullptr on failure. */
void* device_alloc(size_t bytes);
void  device_free(void* p);

/* Compute max |x| over a device FP64 buffer of `n` elements. Synchronizes.
 * Returns 0.0 for an empty buffer. */
double device_maxabs_fp64(const double* dx, size_t n);

#endif /* UTILS_H */
