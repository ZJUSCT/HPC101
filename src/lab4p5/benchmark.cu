/*
 * benchmark.cu — 统一性能与精度测评程序
 *
 * 对每个矩阵规模和 splits 组合运行所有 GEMM 实现，
 * 输出吞吐量（GFLOPS）和相对 FP64 真值的 L2 相对误差。
 *
 * 用法:
 *   benchmark [sizes] [splits] [iters] [--csv]
 *
 *   sizes  : 逗号分隔的矩阵维度（方阵 M=N=K），默认 "1024,2048,4096"
 *   splits : 逗号分隔的 split 数，默认 "2,4"
 *   iters  : 计时迭代次数（取最小值），默认 5
 *
 * 示例:
 *   benchmark 1024,2048,4096,8192 2,4,8 10 --csv > results.csv
 */
#include "gemm_api.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

struct Size { int M, N, K; };

static std::vector<int> parse_ints(const char* s) {
    std::vector<int> v; const char* p = s;
    while (*p) { int x = strtol(p, (char**)&p, 10); v.push_back(x); if (*p==',') ++p; }
    return v;
}

/* 计时：取 iters 次的最小值 */
template <class Fn>
static double time_min_ms(Fn fn, int iters, cudaStream_t stream) {
    fn();
    cudaStreamSynchronize(stream);
    CudaTimer t;
    t.begin(stream);
    for (int i = 0; i < iters; ++i) fn();
    t.end(stream);
    return t.elapsed_ms() / iters;
}

int main(int argc, char** argv) {
    std::vector<int> sizes  = parse_ints("1024,2048,4096");
    std::vector<int> splits = parse_ints("2,4,6,8");
    int iters = 5;
    bool csv = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv") csv = true;
        else if (i == 1) sizes  = parse_ints(argv[i]);
        else if (i == 2) splits = parse_ints(argv[i]);
        else if (i == 3) iters  = atoi(argv[i]);
    }

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    cublasHandle_t handle; cublasCreate(&handle);
    cudaStream_t stream; cudaStreamCreate(&stream);

    /* cuBLAS emulated 需要的 workspace */
    size_t ws_bytes = (size_t)2 * 1024 * 1024 * 1024;
    void* emu_workspace = nullptr;
    cudaMalloc(&emu_workspace, ws_bytes);

    if (!csv) {
        printf("==========================================================================\n");
        printf(" HPC101 Lab 4.5  INT8 模拟 FP64 GEMM 测评\n");
        printf(" GPU: %s (CC %d.%d), %.0f MiB global memory\n",
               prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0*1024.0));
        printf(" iters=%d (取最小值), 存储格式: 列主序\n", iters);
        printf("==========================================================================\n");
    } else {
        printf("size_M,size_N,size_K,method,splits,time_ms,gflops,max_abs_err,l2_rel_err\n");
    }

    for (int S : sizes) {
        Size sz{ S, S, S };
        int M = sz.M, N = sz.N, K = sz.K;
        size_t EA = (size_t)M*K, EB = (size_t)K*N, EC = (size_t)M*N;

        std::vector<double> hA(EA), hB(EB);
        fill_matrix(hA.data(), EA, 0x1234ULL + 7*S);
        fill_matrix(hB.data(), EB, 0x5678ULL + 7*S);

        double *dA, *dB, *dCref, *dCtest;
        cudaMalloc(&dA, EA*sizeof(double));
        cudaMalloc(&dB, EB*sizeof(double));
        cudaMalloc(&dCref, EC*sizeof(double));
        cudaMalloc(&dCtest, EC*sizeof(double));
        cudaMemcpy(dA, hA.data(), EA*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), EB*sizeof(double), cudaMemcpyHostToDevice);

        double flops = 2.0 * (double)M * (double)N * (double)K;

        /* FP64 cuBLAS 参考基准（精度参考 + 性能参考） */
        double ms_ref = time_min_ms([&](){
            gemm_fp64_cublas(M,N,K,dA,dB,dCref,handle); }, iters, stream);
        double gf_ref = flops / (ms_ref*1e-3) / 1e9;
        AccuracyReport acc_self{};
        accuracy_compare(dCref, dCref, M, N, &acc_self);

        if (!csv) {
            printf("\n[M=%d N=%d K=%d]  (2*MNK FLOPs = %.3g)\n", M,N,K, flops);
            printf("  %-26s %6s %10s %10s %12s %12s\n",
                   "method","splits","time(ms)","GFLOPS","maxAbsErr","L2RelErr");
            printf("  %-26s %6s %10.3f %10.1f %12.3e %12.3e\n",
                   "fp64_cublas (REF)","-", ms_ref, gf_ref, 0.0, 0.0);
        } else {
            printf("%d,%d,%d,fp64_cublas,0,%.4f,%.2f,0,0\n", M,N,K, ms_ref, gf_ref);
        }

        /* INT8 模拟: baseline / student / cublas_emulated */
        const char* names[3] = { "int8_cublas_baseline",
                                 "my_int8_fp64",
                                 "cublas_emulated" };
        for (int sp : splits) {
            for (int m = 0; m < 3; ++m) {
                double ms = 0;
                switch (m) {
                case 0: gemm_int8_cublas_baseline(M,N,K,dA,dB,dCtest,sp,handle,stream); break;
                case 1: gemm_my_int8_fp64          (M,N,K,dA,dB,dCtest,sp,handle,stream); break;
                case 2: gemm_cublas_emulated       (M,N,K,dA,dB,dCtest,sp,handle,emu_workspace,ws_bytes,stream); break;
                }
                cudaStreamSynchronize(stream);
                CudaTimer t; t.begin(stream);
                for (int it=0; it<iters; ++it) {
                    switch (m) {
                    case 0: gemm_int8_cublas_baseline(M,N,K,dA,dB,dCtest,sp,handle,stream); break;
                    case 1: gemm_my_int8_fp64          (M,N,K,dA,dB,dCtest,sp,handle,stream); break;
                    case 2: gemm_cublas_emulated       (M,N,K,dA,dB,dCtest,sp,handle,emu_workspace,ws_bytes,stream); break;
                    }
                }
                t.end(stream); ms = t.elapsed_ms()/iters;
                accuracy_compare(dCref, dCtest, M, N, &acc_self);
                double gf = flops/(ms*1e-3)/1e9;
                if (!csv) {
                    printf("  %-26s %6d %10.3f %10.1f %12.3e %12.3e\n",
                           names[m], sp, ms, gf, acc_self.max_abs_err, acc_self.l2_rel_err);
                } else {
                    printf("%d,%d,%d,%s,%d,%.4f,%.2f,%.3e,%.3e\n",
                           M,N,K,names[m],sp,ms,gf,
                           acc_self.max_abs_err, acc_self.l2_rel_err);
                }
            }
        }

        cudaFree(dA); cudaFree(dB); cudaFree(dCref); cudaFree(dCtest);
    }

    if (!csv) printf("\nDone.\n");
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    if (emu_workspace) cudaFree(emu_workspace);
    return 0;
}
