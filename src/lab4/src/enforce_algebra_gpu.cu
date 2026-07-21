#include "enforce_algebra.h"

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_manager.h"

__global__ void enforce_ga_kernel(
    int n,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz
) {
    constexpr double ONE = 1.0;
    constexpr double F1o3 = 1.0/3.0;
    constexpr double TWO = 2.0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    double gxx = dxx[idx] + ONE;
    double gyy = dyy[idx] + ONE;
    double gzz = dzz[idx] + ONE;
    double gxyv = gxy[idx];
    double gxzv = gxz[idx];
    double gyzv = gyz[idx];

    double detg = gxx * gyy * gzz + gxyv * gyzv * gxzv + gxzv * gxyv * gyzv
                - gxzv * gyy * gxzv - gxyv * gxyv * gzz - gxx * gyzv * gyzv;

    double scale = ONE / pow(detg, F1o3);

    gxx *= scale; gxyv *= scale; gxzv *= scale;
    gyy *= scale; gyzv *= scale; gzz *= scale;

    dxx[idx] = gxx - ONE;
    dyy[idx] = gyy - ONE;
    dzz[idx] = gzz - ONE;
    gxy[idx] = gxyv;
    gxz[idx] = gxzv;
    gyz[idx] = gyzv;

    double gupxx =   ( gyy * gzz - gyzv * gyzv );
    double gupxy = - ( gxyv * gzz - gyzv * gxzv );
    double gupxz =   ( gxyv * gyzv - gyy * gxzv );
    double gupyy =   ( gxx * gzz - gxzv * gxzv );
    double gupyz = - ( gxx * gyzv - gxyv * gxzv );
    double gupzz =   ( gxx * gyy - gxyv * gxyv );

    double trA = gupxx * Axx[idx] + gupyy * Ayy[idx] + gupzz * Azz[idx]
                + TWO * (gupxy * Axy[idx] + gupxz * Axz[idx] + gupyz * Ayz[idx]);

    Axx[idx] -= F1o3 * gxx * trA;
    Axy[idx] -= F1o3 * gxyv * trA;
    Axz[idx] -= F1o3 * gxzv * trA;
    Ayy[idx] -= F1o3 * gyy * trA;
    Ayz[idx] -= F1o3 * gyzv * trA;
    Azz[idx] -= F1o3 * gzz * trA;
}

void gpu_enforce_ga_launch(
    cudaStream_t &stream,
    int ex[3],
    double* d_dxx, double* d_gxy, double* d_gxz,
    double* d_dyy, double* d_gyz, double* d_dzz,
    double* d_Axx, double* d_Axy, double* d_Axz,
    double* d_Ayy, double* d_Ayz, double* d_Azz
) {
    int n = ex[0] * ex[1] * ex[2];

    int block = 256;
    int grid = (n + block - 1) / block;

    enforce_ga_kernel<<<grid, block, 0, stream>>>(
        n,
        d_dxx, d_gxy, d_gxz,
        d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz, 
        d_Ayy, d_Ayz, d_Azz
    );
}