#include "sommerfeld_rout.h"
#include <cuda_runtime.h>
#include <math.h>

#include "gpu_manager.h"
#include "fmisc.h"

constexpr int ORDN = 6;
constexpr double ZEO = 0.0;
constexpr double ONE = 1.0;
constexpr double TWO = 2.0;
constexpr int NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2, CORRECTSTEP = 1;

__device__ bool is_sommerfeld_boundary(
    int i, int j, int k,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    int Symmetry
) {
    double dX = X[1] - X[0];
    double dY = Y[1] - Y[0];
    double dZ = Z[1] - Z[0];

    // Upper boundaries
    if (i == ex0 && fabs(X[ex0-1] - xmax) < dX) return true;
    if (j == ex1 && fabs(Y[ex1-1] - ymax) < dY) return true;
    if (k == ex2 && fabs(Z[ex2-1] - zmax) < dZ) return true;

    // Lower boundaries (excluding symmetry planes)
    if (i == 1 && fabs(X[0] - xmin) < dX && !(Symmetry == OCTANT && fabs(xmin) < dX/2.0)) return true;
    if (j == 1 && fabs(Y[0] - ymin) < dY && !(Symmetry == OCTANT && fabs(ymin) < dY/2.0)) return true;
    if (k == 1 && fabs(Z[0] - zmin) < dZ && !(Symmetry > NO_SYMM && fabs(zmin) < dZ/2.0)) return true;

    return false;
}

__global__ void sommerfeld_rout_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT,
    const double* chi0, const double* Lap0,
    const double* f0, double* f,
    const double SYM1, const double SYM2, const double SYM3, 
    int Symmetry,
    int precor
) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    int k0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i0 >= ex0 || j0 >= ex1 || k0 >= ex2) return;

    // 转换为 Fortran 的 1-based 索引
    int i = i0 + 1;
    int j = j0 + 1;
    int k = k0 + 1;

    // 仅当属于边界时才继续执行计算
    if (!is_sommerfeld_boundary(i, j, k, ex0, ex1, ex2, X, Y, Z, xmin, ymin, zmin, xmax, ymax, zmax, Symmetry)) {
        return;
    }

    int ex_idx = k0 * ex1 * ex0 + j0 * ex0 + i0;
    
    // 局部组装 ext 数组，安全传递给底层的 device 函数，避免 Pointer Decay
    int ext[3] = {ex0, ex1, ex2};
    const double SoA[3] = {SYM1, SYM2, SYM3};

    if (precor == CORRECTSTEP) {
        f[ex_idx] = f0[ex_idx];
    } else {
        double dX = X[1] - X[0];
        double dY = Y[1] - Y[0];
        double dZ = Z[1] - Z[0];

        double r = (Lap0[ex_idx] + ONE) * sqrt(ONE + chi0[ex_idx]) * dT / 
                   sqrt(X[i0] * X[i0] + Y[j0] * Y[j0] + Z[k0] * Z[k0]);
        double fac = ONE - r;

        double cx[3];
        cx[0] = r * X[i0] / dX;
        cx[1] = r * Y[j0] / dY;
        cx[2] = r * Z[k0] / dZ;

        int cxB[3];
        cxB[0] = (cx[0] > ZEO) ? i - (int)trunc(cx[0]) - ORDN/2 : i - (int)trunc(cx[0]) - ORDN/2 + 1;
        cxB[1] = (cx[1] > ZEO) ? j - (int)trunc(cx[1]) - ORDN/2 : j - (int)trunc(cx[1]) - ORDN/2 + 1;
        cxB[2] = (cx[2] > ZEO) ? k - (int)trunc(cx[2]) - ORDN/2 : k - (int)trunc(cx[2]) - ORDN/2 + 1;

        for (int m = 0; m < 3; ++m) {
            if (cx[m] > ZEO) {
                cx[m] = trunc(cx[m]) - cx[m] + ORDN/2;
            } else {
                cx[m] = trunc(cx[m]) - cx[m] + ORDN/2 - 1;
            }
        }

        int cxT[3];
        for (int m = 0; m < 3; ++m) {
            cxT[m] = cxB[m] + ORDN - 1;
        }

        if (Symmetry == NO_SYMM && cxB[2] < 1) {
            cx[2] += (cxB[2] - 1);
            cxT[2] -= (cxB[2] - 1);
            cxB[2] = 1;
        }
        if (Symmetry < OCTANT && cxB[1] < 1) {
            cx[1] += (cxB[1] - 1);
            cxT[1] -= (cxB[1] - 1);
            cxB[1] = 1;
        }
        if (Symmetry < OCTANT && cxB[0] < 1) {
            cx[0] += (cxB[0] - 1);
            cxT[0] -= (cxB[0] - 1);
            cxB[0] = 1;
        }

        for (int m = 0; m < 3; ++m) {
            if (cxT[m] > ext[m]) {
                cx[m] += (cxT[m] - ext[m]);
                cxB[m] -= (cxT[m] - ext[m]);
                cxT[m] = ext[m];
            }
        }

        double ya[ORDN * ORDN * ORDN];
        d_decide3d(ext, f0, f0, cxB, cxT, SoA, ya, ORDN, Symmetry);
        double ddy;
        double r_interp;
        double xa[ORDN];
        for(int m = 0; m < ORDN; ++ m) xa[m] = (double)m;
        
        d_polin3_1b(xa, xa, xa, ya, cx[0], cx[1], cx[2], r_interp, ddy, ORDN);
        f[ex_idx] = r_interp * fac;
    }
}

__global__ void sommerfeld_routbam_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double* f_rhs,
    const double* f0,
    double velocity,
    const double SYM1, const double SYM2, const double SYM3,
    int Symmetry
) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    int k0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i0 >= ex0 || j0 >= ex1 || k0 >= ex2) return;

    int i = i0 + 1;
    int j = j0 + 1;
    int k = k0 + 1;

    if (!is_sommerfeld_boundary(i, j, k, ex0, ex1, ex2, X, Y, Z, xmin, ymin, zmin, xmax, ymax, zmax, Symmetry)) {
        return;
    }

    int ex_idx = k0 * ex1 * ex0 + j0 * ex0 + i0;
    int ext[3] = {ex0, ex1, ex2};
    const double SoA[3] = {SYM1, SYM2, SYM3};

    double dX = X[1] - X[0];
    double dY = Y[1] - Y[0];
    double dZ = Z[1] - Z[0];

    double d2dx = ONE / (TWO * dX);
    double d2dy = ONE / (TWO * dY);
    double d2dz = ONE / (TWO * dZ);

    int imin = 1, jmin = 1, kmin = 1;
    if (Symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = 0;
    if (Symmetry > EQ_SYMM && fabs(X[0]) < dX) imin = 0;
    if (Symmetry > EQ_SYMM && fabs(Y[0]) < dY) jmin = 0;

    int imax = ex0;
    int jmax = ex1;
    int kmax = ex2;

    double R = sqrt(X[i0] * X[i0] + Y[j0] * Y[j0] + Z[k0] * Z[k0]);
    double wx = velocity * X[i0] / R;
    double wy = velocity * Y[j0] / R;
    double wz = velocity * Z[k0] / R;

    double fx = 0.0, fy = 0.0, fz = 0.0;

    // x 方向偏导计算
    if (wx > 0) {
        if (i - 2 >= imin) {
            fx = d2dx * (3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA) -
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i-1, j, k, SoA) +
                             d_symmetry_bd_1b(1, ext, (double*)f0, i-2, j, k, SoA));
        } else if (i - 1 >= imin) {
            fx = d2dx * (-d_symmetry_bd_1b(1, ext, (double*)f0, i-1, j, k, SoA) +
                          d_symmetry_bd_1b(1, ext, (double*)f0, i+1, j, k, SoA));
        } else {
            fx = d2dx * (-d_symmetry_bd_1b(1, ext, (double*)f0, i+2, j, k, SoA) +
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i+1, j, k, SoA) -
                         3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA));
        }
    } else if (wx < 0) {
        if (i + 2 <= imax) {
            fx = d2dx * (-d_symmetry_bd_1b(1, ext, (double*)f0, i+2, j, k, SoA) +
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i+1, j, k, SoA) -
                         3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA));
        } else if (i + 1 <= imax) {
            fx = d2dx * (-d_symmetry_bd_1b(1, ext, (double*)f0, i-1, j, k, SoA) +
                          d_symmetry_bd_1b(1, ext, (double*)f0, i+1, j, k, SoA));
        } else {
            fx = d2dx * (3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA) -
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i-1, j, k, SoA) +
                             d_symmetry_bd_1b(1, ext, (double*)f0, i-2, j, k, SoA));
        }
    }

    // y 方向偏导计算
    if (wy > 0) {
        if (j - 2 >= jmin) {
            fy = d2dy * (3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA) -
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j-1, k, SoA) +
                             d_symmetry_bd_1b(1, ext, (double*)f0, i, j-2, k, SoA));
        } else if (j - 1 >= jmin) {
            fy = d2dy * (-d_symmetry_bd_1b(1, ext, (double*)f0, i, j-1, k, SoA) +
                          d_symmetry_bd_1b(1, ext, (double*)f0, i, j+1, k, SoA));
        } else {
            fy = d2dy * (-d_symmetry_bd_1b(1, ext, (double*)f0, i, j+2, k, SoA) +
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j+1, k, SoA) -
                         3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA));
        }
    } else if (wy < 0) {
        if (j + 2 <= jmax) {
            fy = d2dy * (-d_symmetry_bd_1b(1, ext, (double*)f0, i, j+2, k, SoA) +
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j+1, k, SoA) -
                         3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA));
        } else if (j + 1 <= jmax) {
            fy = d2dy * (-d_symmetry_bd_1b(1, ext, (double*)f0, i, j-1, k, SoA) +
                          d_symmetry_bd_1b(1, ext, (double*)f0, i, j+1, k, SoA));
        } else {
            fy = d2dy * (3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA) -
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j-1, k, SoA) +
                             d_symmetry_bd_1b(1, ext, (double*)f0, i, j-2, k, SoA));
        }
    }

    // z 方向偏导计算
    if (wz > 0) {
        if (k - 2 >= kmin) {
            fz = d2dz * (3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA) -
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k-1, SoA) +
                             d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k-2, SoA));
        } else if (k - 1 >= kmin) {
            fz = d2dz * (-d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k-1, SoA) +
                          d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k+1, SoA));
        } else {
            fz = d2dz * (-d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k+2, SoA) +
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k+1, SoA) -
                         3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA));
        }
    } else if (wz < 0) {
        if (k + 2 <= kmax) {
            fz = d2dz * (-d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k+2, SoA) +
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k+1, SoA) -
                         3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA));
        } else if (k + 1 <= kmax) {
            fz = d2dz * (-d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k-1, SoA) +
                          d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k+1, SoA));
        } else {
            fz = d2dz * (3 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k, SoA) -
                         4 * d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k-1, SoA) +
                             d_symmetry_bd_1b(1, ext, (double*)f0, i, j, k-2, SoA));
        }
    }

    f_rhs[ex_idx] = -velocity * (fx * X[i0] + fy * Y[j0] + fz * Z[k0] + f0[ex_idx]) / R;
}

void gpu_sommerfeld_rout_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor
) {
    dim3 block(8, 8, 4);
    dim3 grid((ex[0] + block.x - 1) / block.x, 
              (ex[1] + block.y - 1) / block.y, 
              (ex[2] + block.z - 1) / block.z);

    sommerfeld_rout_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z, xmin, ymin, zmin, xmax, ymax, zmax,
        dT, d_chi0, d_Lap0, d_f0, d_f, SoA[0], SoA[1], SoA[2], Symmetry, precor
    );
}

void gpu_sommerfeld_routbam_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double* d_f_rhs, const double* d_f0,
    double velocity, const double SoA[3], int Symmetry
) {
    dim3 block(8, 8, 4);
    dim3 grid((ex[0] + block.x - 1) / block.x, 
              (ex[1] + block.y - 1) / block.y, 
              (ex[2] + block.z - 1) / block.z);

    sommerfeld_routbam_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z, xmin, ymin, zmin, xmax, ymax, zmax,
        d_f_rhs, d_f0, velocity, SoA[0], SoA[1], SoA[2], Symmetry
    );
}