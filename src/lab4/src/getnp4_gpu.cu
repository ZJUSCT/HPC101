#include "getnp4.h"

#include <cmath>
#include <cuda_runtime.h>
#include "derivatives.h"
#include "macrodef.h"

// ---------------------------------------------------------
// CUDA Kernel
// ---------------------------------------------------------
__global__ void getnp4_kernel(
    int ex0, int ex1, int ex2, 
    const double* X, const double* Y, const double* Z,
    const double* chi, const double* trK,
    const double* dxx, const double* gxy, const double* gxz,
    const double* dyy, const double* gyz, const double* dzz,
    const double* Axx, const double* Axy, const double* Axz,
    const double* Ayy, const double* Ayz, const double* Azz,
    const double* Gamxxx, const double* Gamxxy, const double* Gamxxz,
    const double* Gamxyy, const double* Gamxyz, const double* Gamxzz,
    const double* Gamyxx, const double* Gamyxy, const double* Gamyxz,
    const double* Gamyyy, const double* Gamyyz, const double* Gamyzz,
    const double* Gamzxx, const double* Gamzxy, const double* Gamzxz,
    const double* Gamzyy, const double* Gamzyz, const double* Gamzzz,
    const double* Rxx, const double* Rxy, const double* Rxz,
    const double* Ryy, const double* Ryz, const double* Rzz,
    double* Rpsi4, double* Ipsi4,
    int symmetry
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    // 适配 Fortran 的 Column-Major (列优先) 内存布局
    size_t idx = i + j * ex0 + k * ex0 * ex1;

    // 构造局部 ex 数组以匹配 d_fderivs_point 的签名
    const int ex[3] = {ex0, ex1, ex2};

    const double ZEO = 0.0;
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double F1o3 = 1.0 / 3.0;
    const double SYM = 1.0;
    const double ANTI = -1.0;
    const double TINYRR = 1e-14;

    double cx = X[i];
    double cy = Y[j];
    double cz = Z[k];

    // 加载当前点度规分量并计算物理度规 (+1)
    double l_gxx = dxx[idx] + ONE;
    double l_gyy = dyy[idx] + ONE;
    double l_gzz = dzz[idx] + ONE;
    double l_gxy = gxy[idx];
    double l_gxz = gxz[idx];
    double l_gyz = gyz[idx];

    double l_chi = chi[idx];
    double chipn1 = l_chi + ONE;
    double f = ONE / chipn1;

    // 求解逆度规
    double detG = l_gxx * l_gyy * l_gzz + l_gxy * l_gyz * l_gxz + l_gxz * l_gxy * l_gyz -
                  l_gxz * l_gyy * l_gxz - l_gxy * l_gxy * l_gzz - l_gxx * l_gyz * l_gyz;
    
    double l_gupxx =  (l_gyy * l_gzz - l_gyz * l_gyz) / detG;
    double l_gupxy = -(l_gxy * l_gzz - l_gyz * l_gxz) / detG;
    double l_gupxz =  (l_gxy * l_gyz - l_gyy * l_gxz) / detG;
    double l_gupyy =  (l_gxx * l_gzz - l_gxz * l_gxz) / detG;
    double l_gupyz = -(l_gxx * l_gyz - l_gxy * l_gxz) / detG;
    double l_gupzz =  (l_gxx * l_gyy - l_gxy * l_gxy) / detG;

    // 初始化 U, V, W 向量
    double vx, vy, vz, ux, uy, uz, wx, wy, wz;
    
    if (fabs(cx) < TINYRR && fabs(cy) < TINYRR && fabs(cz) < TINYRR) {
        vx = TINYRR; vy = TINYRR; vz = TINYRR;
    } else {
        vx = cx; vy = cy; vz = cz;
    }
    
    if (fabs(cx) < TINYRR && fabs(cy) < TINYRR) {
        ux = -TINYRR; uy = TINYRR; uz = ZEO;
        wx = TINYRR * cz; wy = TINYRR * cz; wz = -TWO * TINYRR * TINYRR;
    } else {
        ux = -cy; uy = cx; uz = ZEO;
        wx = cx * cz; wy = cy * cz; wz = -(cx * cx + cy * cy);
    }

    double fxt = vx, fyt = vy, fzt = vz;
    vx = l_gupxx * fxt + l_gupxy * fyt + l_gupxz * fzt;
    vy = l_gupxy * fxt + l_gupyy * fyt + l_gupyz * fzt;
    vz = l_gupxz * fxt + l_gupyz * fyt + l_gupzz * fzt;

    double temp;

    // Gram-Schmidt 正交化过程
    temp = l_gxx*vx*vx + l_gyy*vy*vy + l_gzz*vz*vz + (l_gxy*vx*vy + l_gxz*vx*vz + l_gyz*vy*vz)*TWO;
    temp = sqrt(temp*f);
    vx /= temp; vy /= temp; vz /= temp;

    temp = l_gxx*vx*ux + l_gxy*vx*uy + l_gxz*vx*uz +
           l_gxy*vy*ux + l_gyy*vy*uy + l_gyz*vy*uz +
           l_gxz*vz*ux + l_gyz*vz*uy + l_gzz*vz*uz;
    temp *= f;
    ux -= temp*vx; uy -= temp*vy; uz -= temp*vz;
    
    temp = l_gxx*ux*ux + l_gyy*uy*uy + l_gzz*uz*uz + (l_gxy*ux*uy + l_gxz*ux*uz + l_gyz*uy*uz)*TWO;
    temp = sqrt(temp*f);
    ux /= temp; uy /= temp; uz /= temp;

    temp = l_gxx*vx*wx + l_gxy*vx*wy + l_gxz*vx*wz +
           l_gxy*vy*wx + l_gyy*vy*wy + l_gyz*vy*wz +
           l_gxz*vz*wx + l_gyz*vz*wy + l_gzz*vz*wz;
    temp *= f;
    wx -= temp*vx; wy -= temp*vy; wz -= temp*vz;
    
    temp = l_gxx*ux*wx + l_gxy*ux*wy + l_gxz*ux*wz +
           l_gxy*uy*wx + l_gyy*uy*wy + l_gyz*uy*wz +
           l_gxz*uz*wx + l_gyz*uz*wy + l_gzz*uz*wz;
    temp *= f;
    wx -= temp*ux; wy -= temp*uy; wz -= temp*uz;
    
    temp = l_gxx*wx*wx + l_gyy*wy*wy + l_gzz*wz*wz + (l_gxy*wx*wy + l_gxz*wx*wz + l_gyz*wy*wz)*TWO;
    temp = sqrt(temp*f);
    wx /= temp; wy /= temp; wz /= temp;

    // 计算导数
    double Axxx, Axxy, Axxz;
    d_fderivs_point(ex, Axx, &Axxx, &Axxy, &Axxz, X, Y, Z, SYM, SYM, SYM, symmetry, 0, i, j, k);
    
    double Axyx, Axyy, Axyz;
    d_fderivs_point(ex, Axy, &Axyx, &Axyy, &Axyz, X, Y, Z, ANTI, ANTI, SYM, symmetry, 0, i, j, k);
    
    double Axzx, Axzy, Axzz;
    d_fderivs_point(ex, Axz, &Axzx, &Axzy, &Axzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, 0, i, j, k);
    
    double Ayyx, Ayyy, Ayyz;
    d_fderivs_point(ex, Ayy, &Ayyx, &Ayyy, &Ayyz, X, Y, Z, SYM, SYM, SYM, symmetry, 0, i, j, k);
    
    double Ayzx, Ayzy, Ayzz;
    d_fderivs_point(ex, Ayz, &Ayzx, &Ayzy, &Ayzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, 0, i, j, k);
    
    double Azzx, Azzy, Azzz;
    d_fderivs_point(ex, Azz, &Azzx, &Azzy, &Azzz, X, Y, Z, SYM, SYM, SYM, symmetry, 0, i, j, k);
    
    double chix, chiy, chiz;
    d_fderivs_point(ex, chi, &chix, &chiy, &chiz, X, Y, Z, SYM, SYM, SYM, symmetry, 0, i, j, k);
    
    double fx_trk, fy_trk, fz_trk;
    d_fderivs_point(ex, trK, &fx_trk, &fy_trk, &fz_trk, X, Y, Z, SYM, SYM, SYM, symmetry, 0, i, j, k);

    // 加载当前点的无迹外曲率分量
    double l_Axx = Axx[idx];
    double l_Axy = Axy[idx];
    double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx];
    double l_Ayz = Ayz[idx];
    double l_Azz = Azz[idx];

    // 计算 D_k K_ij
    Axxx -= (Gamxxx[idx]*l_Axx + Gamyxx[idx]*l_Axy + Gamzxx[idx]*l_Axz)*TWO + chix/chipn1*l_Axx - F1o3*l_gxx*fx_trk;
    Axxy -= (Gamxxy[idx]*l_Axx + Gamyxy[idx]*l_Axy + Gamzxy[idx]*l_Axz)*TWO + chiy/chipn1*l_Axx - F1o3*l_gxx*fy_trk;
    Axxz -= (Gamxxz[idx]*l_Axx + Gamyxz[idx]*l_Axy + Gamzxz[idx]*l_Axz)*TWO + chiz/chipn1*l_Axx - F1o3*l_gxx*fz_trk;
    
    Ayyx -= (Gamxxy[idx]*l_Axy + Gamyxy[idx]*l_Ayy + Gamzxy[idx]*l_Ayz)*TWO + chix/chipn1*l_Ayy - F1o3*l_gyy*fx_trk;
    Ayyy -= (Gamxyy[idx]*l_Axy + Gamyyy[idx]*l_Ayy + Gamzyy[idx]*l_Ayz)*TWO + chiy/chipn1*l_Ayy - F1o3*l_gyy*fy_trk;
    Ayyz -= (Gamxyz[idx]*l_Axy + Gamyyz[idx]*l_Ayy + Gamzyz[idx]*l_Ayz)*TWO + chiz/chipn1*l_Ayy - F1o3*l_gyy*fz_trk;
    
    Azzx -= (Gamxxz[idx]*l_Axz + Gamyxz[idx]*l_Ayz + Gamzxz[idx]*l_Azz)*TWO + chix/chipn1*l_Azz - F1o3*l_gzz*fx_trk;
    Azzy -= (Gamxyz[idx]*l_Axz + Gamyyz[idx]*l_Ayz + Gamzyz[idx]*l_Azz)*TWO + chiy/chipn1*l_Azz - F1o3*l_gzz*fy_trk;
    Azzz -= (Gamxzz[idx]*l_Axz + Gamyzz[idx]*l_Ayz + Gamzzz[idx]*l_Azz)*TWO + chiz/chipn1*l_Azz - F1o3*l_gzz*fz_trk;
    
    Axyx -= (Gamxxy[idx]*l_Axx + Gamyxy[idx]*l_Axy + Gamzxy[idx]*l_Axz + Gamxxx[idx]*l_Axy + Gamyxx[idx]*l_Ayy + Gamzxx[idx]*l_Ayz) + chix/chipn1*l_Axy - F1o3*l_gxy*fx_trk;
    Axyy -= (Gamxyy[idx]*l_Axx + Gamyyy[idx]*l_Axy + Gamzyy[idx]*l_Axz + Gamxxy[idx]*l_Axy + Gamyxy[idx]*l_Ayy + Gamzxy[idx]*l_Ayz) + chiy/chipn1*l_Axy - F1o3*l_gxy*fy_trk;
    Axyz -= (Gamxyz[idx]*l_Axx + Gamyyz[idx]*l_Axy + Gamzyz[idx]*l_Axz + Gamxxz[idx]*l_Axy + Gamyxz[idx]*l_Ayy + Gamzxz[idx]*l_Ayz) + chiz/chipn1*l_Axy - F1o3*l_gxy*fz_trk;
    
    Axzx -= (Gamxxz[idx]*l_Axx + Gamyxz[idx]*l_Axy + Gamzxz[idx]*l_Axz + Gamxxx[idx]*l_Axz + Gamyxx[idx]*l_Ayz + Gamzxx[idx]*l_Azz) + chix/chipn1*l_Axz - F1o3*l_gxz*fx_trk;
    Axzy -= (Gamxyz[idx]*l_Axx + Gamyyz[idx]*l_Axy + Gamzyz[idx]*l_Axz + Gamxxy[idx]*l_Axz + Gamyxy[idx]*l_Ayz + Gamzxy[idx]*l_Azz) + chiy/chipn1*l_Axz - F1o3*l_gxz*fy_trk;
    Axzz -= (Gamxzz[idx]*l_Axx + Gamyzz[idx]*l_Axy + Gamzzz[idx]*l_Axz + Gamxxz[idx]*l_Axz + Gamyxz[idx]*l_Ayz + Gamzxz[idx]*l_Azz) + chiz/chipn1*l_Axz - F1o3*l_gxz*fz_trk;
    
    Ayzx -= (Gamxxz[idx]*l_Axy + Gamyxz[idx]*l_Ayy + Gamzxz[idx]*l_Ayz + Gamxxy[idx]*l_Axz + Gamyxy[idx]*l_Ayz + Gamzxy[idx]*l_Azz) + chix/chipn1*l_Ayz - F1o3*l_gyz*fx_trk;
    Ayzy -= (Gamxyz[idx]*l_Axy + Gamyyz[idx]*l_Ayy + Gamzyz[idx]*l_Ayz + Gamxyy[idx]*l_Axz + Gamyyy[idx]*l_Ayz + Gamzyy[idx]*l_Azz) + chiy/chipn1*l_Ayz - F1o3*l_gyz*fy_trk;
    Ayzz -= (Gamxzz[idx]*l_Axy + Gamyzz[idx]*l_Ayy + Gamzzz[idx]*l_Ayz + Gamxyz[idx]*l_Axz + Gamyyz[idx]*l_Ayz + Gamzyz[idx]*l_Azz) + chiz/chipn1*l_Ayz - F1o3*l_gyz*fz_trk;

    // Symmetrize B_ij
    double Bxx = (vy*(Axxy - Axyx) + vz*(Axxz - Axzx)) * f;
    double Byy = (vx*(Ayyx - Axyy) + vz*(Ayyz - Ayzy)) * f;
    double Bzz = (vx*(Azzx - Axzz) + vy*(Azzy - Ayzz)) * f;
    double Bxy = (vx*(Axyx - (Axxy+Axyx)/TWO) + vy*(Axyy-Ayyx)/TWO + vz*(Axyz - (Axzy+Ayzx)/TWO)) * f;
    double Bxz = (vx*(Axzx - (Axxz+Axzx)/TWO) + vy*(Axzy - (Axyz+Ayzx)/TWO) + vz*(Axzz-Azzx)/TWO) * f;
    double Byz = (vx*(Ayzx - (Axyz+Axzy)/TWO) + vy*(Ayzy - (Ayyz+Ayzy)/TWO) + vz*(Ayzz-Azzy)/TWO) * f;

    double l_trK = trK[idx];

    // 物理 K_ij
    double Kxx_c = l_Axx + F1o3 * l_trK * l_gxx;
    double Kxy_c = l_Axy + F1o3 * l_trK * l_gxy;
    double Kxz_c = l_Axz + F1o3 * l_trK * l_gxz;
    double Kyy_c = l_Ayy + F1o3 * l_trK * l_gyy;
    double Kyz_c = l_Ayz + F1o3 * l_trK * l_gyz;
    double Kzz_c = l_Azz + F1o3 * l_trK * l_gzz;

    // 计算 E_ij
    double Exx = l_gupxx*Kxx_c*Kxx_c + l_gupyy*Kxy_c*Kxy_c + l_gupzz*Kxz_c*Kxz_c +
                 TWO*(l_gupxy*Kxx_c*Kxy_c + l_gupxz*Kxx_c*Kxz_c + l_gupyz*Kxy_c*Kxz_c);
    double Eyy = l_gupxx*Kxy_c*Kxy_c + l_gupyy*Kyy_c*Kyy_c + l_gupzz*Kyz_c*Kyz_c +
                 TWO*(l_gupxy*Kxy_c*Kyy_c + l_gupxz*Kxy_c*Kyz_c + l_gupyz*Kyy_c*Kyz_c);
    double Ezz = l_gupxx*Kxz_c*Kxz_c + l_gupyy*Kyz_c*Kyz_c + l_gupzz*Kzz_c*Kzz_c +
                 TWO*(l_gupxy*Kxz_c*Kyz_c + l_gupxz*Kxz_c*Kzz_c + l_gupyz*Kyz_c*Kzz_c);
    
    double Exy = l_gupxx*Kxx_c*Kxy_c + l_gupyy*Kxy_c*Kyy_c + l_gupzz*Kxz_c*Kyz_c +
                 l_gupxy*(Kxx_c*Kyy_c + Kxy_c*Kxy_c) +
                 l_gupxz*(Kxx_c*Kyz_c + Kxz_c*Kxy_c) +
                 l_gupyz*(Kxy_c*Kyz_c + Kxz_c*Kyy_c);
    double Exz = l_gupxx*Kxx_c*Kxz_c + l_gupyy*Kxy_c*Kyz_c + l_gupzz*Kxz_c*Kzz_c +
                 l_gupxy*(Kxx_c*Kyz_c + Kxy_c*Kxz_c) +
                 l_gupxz*(Kxx_c*Kzz_c + Kxz_c*Kxz_c) +
                 l_gupyz*(Kxy_c*Kzz_c + Kxz_c*Kyz_c);
    double Eyz = l_gupxx*Kxy_c*Kxz_c + l_gupyy*Kyy_c*Kyz_c + l_gupzz*Kyz_c*Kzz_c +
                 l_gupxy*(Kxy_c*Kyz_c + Kyy_c*Kxz_c) +
                 l_gupxz*(Kxy_c*Kzz_c + Kyz_c*Kxz_c) +
                 l_gupyz*(Kyy_c*Kzz_c + Kyz_c*Kyz_c);

    Exx = Rxx[idx] - (Exx - Kxx_c*l_trK)*f - Bxx;
    Exy = Rxy[idx] - (Exy - Kxy_c*l_trK)*f - Bxy;
    Exz = Rxz[idx] - (Exz - Kxz_c*l_trK)*f - Bxz;
    Eyy = Ryy[idx] - (Eyy - Kyy_c*l_trK)*f - Byy;
    Eyz = Ryz[idx] - (Eyz - Kyz_c*l_trK)*f - Byz;
    Ezz = Rzz[idx] - (Ezz - Kzz_c*l_trK)*f - Bzz;

    // uuww / uw 投影张量
    double uuwwxx = ux * ux - wx * wx;
    double uuwwxy = ux * uy - wx * wy;
    double uuwwxz = ux * uz - wx * wz;
    double uuwwyy = uy * uy - wy * wy;
    double uuwwyz = uy * uz - wy * wz;
    double uuwwzz = uz * uz - wz * wz;

    double uwxx = ux * wx + wx * ux;
    double uwxy = ux * wy + wx * uy;
    double uwxz = ux * wz + wx * uz;
    double uwyy = uy * wy + wy * uy;
    double uwyz = uy * wz + wy * uz;
    double uwzz = uz * wz + wz * uz;

    // 组合 Psi4 结果
    double l_Rpsi4 = Exx*uuwwxx + Eyy*uuwwyy + Ezz*uuwwzz + (Exy*uuwwxy + Exz*uuwwxz + Eyz*uuwwyz)*TWO;
    double l_Ipsi4 = Exx*uwxx + Eyy*uwyy + Ezz*uwzz + (Exy*uwxy + Exz*uwxz + Eyz*uwyz)*TWO;

    Rpsi4[idx] = -l_Rpsi4 / TWO;
    Ipsi4[idx] = -l_Ipsi4 / TWO;
}

void gpu_getnp4_launch(
    cudaStream_t stream,
    int ex[3], 
    const double* d_X, const double* d_Y, const double* d_Z,
    const double* d_chi, const double* d_trK,
    const double* d_dxx, const double* d_gxy, const double* d_gxz,
    const double* d_dyy, const double* d_gyz, const double* d_dzz,
    const double* d_Axx, const double* d_Axy, const double* d_Axz,
    const double* d_Ayy, const double* d_Ayz, const double* d_Azz,
    const double* d_Gamxxx, const double* d_Gamxxy, const double* d_Gamxxz,
    const double* d_Gamxyy, const double* d_Gamxyz, const double* d_Gamxzz,
    const double* d_Gamyxx, const double* d_Gamyxy, const double* d_Gamyxz,
    const double* d_Gamyyy, const double* d_Gamyyz, const double* d_Gamyzz,
    const double* d_Gamzxx, const double* d_Gamzxy, const double* d_Gamzxz,
    const double* d_Gamzyy, const double* d_Gamzyz, const double* d_Gamzzz,
    const double* d_Rxx, const double* d_Rxy, const double* d_Rxz,
    const double* d_Ryy, const double* d_Ryz, const double* d_Rzz,
    double* d_Rpsi4, double* d_Ipsi4,
    int symmetry
) {
    dim3 blockDim(8, 8, 4);
    dim3 gridDim(
        (ex[0] + blockDim.x - 1) / blockDim.x,
        (ex[1] + blockDim.y - 1) / blockDim.y,
        (ex[2] + blockDim.z - 1) / blockDim.z
    );

    getnp4_kernel<<<gridDim, blockDim, 0, stream>>>(
        ex[0], ex[1], ex[2],
        d_X, d_Y, d_Z,
        d_chi, d_trK,
        d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
        d_Gamxxx, d_Gamxxy, d_Gamxxz,
        d_Gamxyy, d_Gamxyz, d_Gamxzz,
        d_Gamyxx, d_Gamyxy, d_Gamyxz,
        d_Gamyyy, d_Gamyyz, d_Gamyzz,
        d_Gamzxx, d_Gamzxy, d_Gamzxz,
        d_Gamzyy, d_Gamzyz, d_Gamzzz,
        d_Rxx, d_Rxy, d_Rxz, d_Ryy, d_Ryz, d_Rzz,
        d_Rpsi4, d_Ipsi4,
        symmetry
    );
}