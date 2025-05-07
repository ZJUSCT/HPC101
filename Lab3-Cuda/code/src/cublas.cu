#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include "common.cuh"
#include "gemm.cuh"
#include <cute/tensor.hpp>

template <>
void run_cublas<float>(thrust::device_vector<float>& d_A,
                       thrust::device_vector<float>& d_B,
                       thrust::device_vector<float>& d_C, const int m,
                       const int n, const int k) {

    cublasLtHandle_t ltHandle = NULL;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    const int lda = k;
    const int ldb = k;
    const int ldc = m;

    thrust::device_vector<float> d_workspace(1024 * 1024 * 4);
    float* workspace = d_workspace.data().get();
    size_t workspaceSize = d_workspace.size() * sizeof(float);
    float* A = d_A.data().get();
    float* B = d_B.data().get();
    float* C = d_C.data().get();

    CUBLAS_CALL(cublasLtCreate(&ltHandle));


    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    CUBLAS_CALL(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CALL(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CALL(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    CUBLAS_CALL(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    CUBLAS_CALL(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    CUBLAS_CALL(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    CUBLAS_CALL(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CALL(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    CUBLAS_CALL(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        CUBLAS_CALL(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    CUBLAS_CALL(cublasLtMatmul(ltHandle,
                               operationDesc,
                               &alpha,
                               A,
                               Adesc,
                               B,
                               Bdesc,
                               &beta,
                               C,
                               Cdesc,
                               C,
                               Cdesc,
                               &heuristicResult.algo,
                               workspace,
                               workspaceSize,
                               0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (ltHandle) CUBLAS_CALL(cublasLtDestroy(ltHandle));
    if (preference) CUBLAS_CALL(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) CUBLAS_CALL(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CALL(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CALL(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) CUBLAS_CALL(cublasLtMatmulDescDestroy(operationDesc));
}

template <>
void run_cublas<cute::half_t>(thrust::device_vector<cute::half_t>& d_A,
                       thrust::device_vector<cute::half_t>& d_B,
                       thrust::device_vector<cute::half_t>& d_C, const int m,
                       const int n, const int k) {

    cublasLtHandle_t ltHandle = NULL;
    const cute::half_t alpha(1.0f);
    const cute::half_t beta(0.0f);
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    const int lda = k;
    const int ldb = k;
    const int ldc = m;

    thrust::device_vector<int> d_workspace(1024 * 1024);
    void* workspace = d_workspace.data().get();
    size_t workspaceSize = d_workspace.size() * sizeof(int);
    cute::half_t* A = d_A.data().get();
    cute::half_t* B = d_B.data().get();
    cute::half_t* C = d_C.data().get();

    CUBLAS_CALL(cublasLtCreate(&ltHandle));


    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    CUBLAS_CALL(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    CUBLAS_CALL(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CALL(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    CUBLAS_CALL(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    CUBLAS_CALL(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    CUBLAS_CALL(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    CUBLAS_CALL(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CALL(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    CUBLAS_CALL(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        CUBLAS_CALL(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    CUBLAS_CALL(cublasLtMatmul(ltHandle,
                               operationDesc,
                               &alpha,
                               A,
                               Adesc,
                               B,
                               Bdesc,
                               &beta,
                               C,
                               Cdesc,
                               C,
                               Cdesc,
                               &heuristicResult.algo,
                               workspace,
                               workspaceSize,
                               0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (ltHandle) CUBLAS_CALL(cublasLtDestroy(ltHandle));
    if (preference) CUBLAS_CALL(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) CUBLAS_CALL(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CALL(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CALL(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) CUBLAS_CALL(cublasLtMatmulDescDestroy(operationDesc));
}