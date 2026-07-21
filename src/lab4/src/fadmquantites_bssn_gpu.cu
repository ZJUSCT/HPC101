#include "fadmquantites_bssn.h"

#include <cuda_runtime.h>
#include <math.h>

// 假设 fmisc.h 中声明了 d_fderivs_point
#include "fmisc.h" 
#include "gpu_manager.h"
#include "derivatives.h"


__global__ void admmass_bssn_kernel(
    int ex0, int ex1, int ex2,
    const double* __restrict__ d_X, const double* __restrict__ d_Y, const double* __restrict__ d_Z,
    const double* __restrict__ chi, const double* __restrict__ trK,
    const double* __restrict__ dxx, const double* __restrict__ gxy, const double* __restrict__ gxz, 
    const double* __restrict__ dyy, const double* __restrict__ gyz, const double* __restrict__ dzz,
    const double* __restrict__ Axx, const double* __restrict__ Axy, const double* __restrict__ Axz, 
    const double* __restrict__ Ayy, const double* __restrict__ Ayz, const double* __restrict__ Azz, 
    const double* __restrict__ Gamx, const double* __restrict__ Gamy, const double* __restrict__ Gamz,
    double* __restrict__ massx, double* __restrict__ massy, double* __restrict__ massz,
    int symmetry
) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    int k0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i0 >= ex0 || j0 >= ex1 || k0 >= ex2) return;

    int idx = (k0 * ex1 + j0) * ex0 + i0;

    double F1o2pi = 1.0 / (2.0 * M_PI);
    double F1o8 = 1.0 / 8.0;

    // =========================================================
    // 1. 调用已有的 device 函数计算 chi 的偏导数
    // =========================================================
    // 转换为 Fortran 的 1-based 索引
    int i = i0 + 1;
    int j = j0 + 1;
    int k = k0 + 1;
    int ext[3] = {ex0, ex1, ex2};
    
    double chix = 0.0, chiy = 0.0, chiz = 0.0;
    
    // Fortran 原代码为: call fderivs(ex,chi,chix,chiy,chiz,X,Y,Z,SYM,SYM,SYM,Symmetry,0)
    // 其中 SYM = 1.D0
    d_fderivs_point(
        ext, chi, 
        &chix, &chiy, &chiz, 
        d_X, d_Y, d_Z, 
        1.0, 1.0, 1.0, 
        symmetry, 0, 
        i0, j0, k0
    );

    // =========================================================
    // 2. 构建度规并求逆
    // =========================================================
    double gxx_v = dxx[idx] + 1.0;
    double gyy_v = dyy[idx] + 1.0;
    double gzz_v = dzz[idx] + 1.0;
    double gxy_v = gxy[idx];
    double gxz_v = gxz[idx];
    double gyz_v = gyz[idx];

    // 行列式
    double det = gxx_v * gyy_v * gzz_v + gxy_v * gyz_v * gxz_v + gxz_v * gxy_v * gyz_v
               - gxz_v * gyy_v * gxz_v - gxy_v * gxy_v * gzz_v - gxx_v * gyz_v * gyz_v;

    // 逆度规 (gup_ij)
    double gupxx = (gyy_v * gzz_v - gyz_v * gyz_v) / det;
    double gupxy = -(gxy_v * gzz_v - gyz_v * gxz_v) / det;
    double gupxz = (gxy_v * gyz_v - gyy_v * gxz_v) / det;
    double gupyy = (gxx_v * gzz_v - gxz_v * gxz_v) / det;
    double gupyz = -(gxx_v * gyz_v - gxy_v * gxz_v) / det;
    double gupzz = (gxx_v * gyy_v - gxy_v * gxy_v) / det;

    // =========================================================
    // 3. 计算 ADM 质量密度
    // =========================================================
    // f = 1.0 / 4.0 / (chi + 1.0)^1.25
    double f = 0.25 * pow(chi[idx] + 1.0, -1.25);

    // mass_i = (Gam_i / 8 + gup_ij * chix_j * f) / (2 * Pi)
    massx[idx] = (F1o8 * Gamx[idx] + f * (gupxx * chix + gupxy * chiy + gupxz * chiz)) * F1o2pi;
    massy[idx] = (F1o8 * Gamy[idx] + f * (gupxy * chix + gupyy * chiy + gupyz * chiz)) * F1o2pi;
    massz[idx] = (F1o8 * Gamz[idx] + f * (gupxz * chix + gupyz * chiy + gupzz * chiz)) * F1o2pi;
}

void gpu_admmass_bssn_launch(
    cudaStream_t stream, const int ext[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    const double* chi, const double* trK,
    const double* dxx, const double* gxy, const double* gxz, 
    const double* dyy, const double* gyz, const double* dzz,
    const double* Axx, const double* Axy, const double* Axz, 
    const double* Ayy, const double* Ayz, const double* Azz, 
    const double* Gamx, const double* Gamy, const double* Gamz,
    double* massx, double* massy, double* massz,
    int symmetry
) {
    dim3 block(8, 8, 8);
    dim3 grid((ext[0] + block.x - 1) / block.x, 
              (ext[1] + block.y - 1) / block.y, 
              (ext[2] + block.z - 1) / block.z);

    admmass_bssn_kernel<<<grid, block, 0, stream>>>(
        ext[0], ext[1], ext[2], 
        d_X, d_Y, d_Z,
        chi, trK, 
        dxx, gxy, gxz, dyy, gyz, dzz,
        Axx, Axy, Axz, Ayy, Ayz, Azz,
        Gamx, Gamy, Gamz, 
        massx, massy, massz, 
        symmetry
    );
}