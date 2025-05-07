#pragma once
#define CUDA_CALL(func)                                                     \
    {                                                                       \
        cudaError_t e = (func);                                             \
        if (!(e == cudaSuccess || e == cudaErrorCudartUnloading)) {         \
            fprintf(stderr, "CUDA: %s:%d: error: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                                 \
            abort();                                                        \
        }                                                                   \
    }
#define CUBLAS_CALL(func)                                                     \
    {                                                                         \
        cublasStatus_t e = (func);                                            \
        if (!(e == CUBLAS_STATUS_SUCCESS)) {                                  \
            fprintf(stderr, "CUBLAS: %s:%d: error: %d\n", __FILE__, __LINE__, \
                    e);                                                       \
            abort();                                                          \
        }                                                                     \
    }
