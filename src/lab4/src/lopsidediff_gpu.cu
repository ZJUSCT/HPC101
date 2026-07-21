#include "lopsidediff.h"

#include "fmisc.h"

#include "macrodef.fh"
#include <cmath>

__device__ __forceinline__ int d_idx3d(int i, int j, int k, const int ex[3]) {
    return i + ex[0] * (j + ex[1] * k);
}

__device__ double d_lopsided_point(
    const int ex[3], const double* f,
    const double* f_rhs, const double* Sfx, const double* Sfy, const double* Sfz,
    const double* X, const double* Y, const double* Z,
    int symmetry, double SYM1, double SYM2, double SYM3,
    int i, int j, int k // i, j, k is 0-based
) {
    const double ZEO = 0.0;
    const double ONE = 1.0;
    const double F3 = 3.0;
    const double F6 = 6.0;
    const double F18 = 18.0;
    const double F12 = 12.0;
    const double F10 = 10.0;
    const double EIT = 8.0;
    const int NO_SYMM = 0, EQ_SYMM = 1;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const double d12dx = ONE / F12 / dX;
    const double d12dy = ONE / F12 / dY;
    const double d12dz = ONE / F12 / dZ;

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    if (i >= imax || j >= jmax || k >= kmax) return 0.0;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -3;
    if (symmetry > EQ_SYMM && fabs(X[0]) < dX) imin = -3;
    if (symmetry > EQ_SYMM && fabs(Y[0]) < dY) jmin = -3;

    double SoA[3] = {SYM1, SYM2, SYM3};

    const auto fh = [&](int ii, int jj, int kk) -> double { // 0-based -> 1-based
        return d_symmetry_bd_1b(3, ex, f, ii + 1, jj + 1, kk + 1, SoA);
    };

    const int idx = d_idx3d(i, j, k, ex);

    const double vx = Sfx[idx];
    const double vy = Sfy[idx];
    const double vz = Sfz[idx];

    double rhs_add = 0.0;

    // --- X Direction ---
    if (vx > ZEO) {
        if (i + 3 <= imax) {
            rhs_add += vx * d12dx * (-F3*fh(i-1,j,k) - F10*fh(i,j,k) + F18*fh(i+1,j,k)
                                     -F6*fh(i+2,j,k) + fh(i+3,j,k));
        } else if (i + 2 <= imax) {
            rhs_add += vx * d12dx * (fh(i-2,j,k) - EIT*fh(i-1,j,k) + EIT*fh(i+1,j,k) - fh(i+2,j,k));
        } else if (i + 1 <= imax) {
            rhs_add -= vx * d12dx * (-F3*fh(i+1,j,k) - F10*fh(i,j,k) + F18*fh(i-1,j,k)
                                     -F6*fh(i-2,j,k) + fh(i-3,j,k));
        }
    } else if (vx < ZEO) {
        if (i - 3 >= imin) {
            rhs_add -= vx * d12dx * (-F3*fh(i+1,j,k) - F10*fh(i,j,k) + F18*fh(i-1,j,k)
                                     -F6*fh(i-2,j,k) + fh(i-3,j,k));
        } else if (i - 2 >= imin) {
            rhs_add += vx * d12dx * (fh(i-2,j,k) - EIT*fh(i-1,j,k) + EIT*fh(i+1,j,k) - fh(i+2,j,k));
        } else if (i - 1 >= imin) {
            rhs_add += vx * d12dx * (-F3*fh(i-1,j,k) - F10*fh(i,j,k) + F18*fh(i+1,j,k)
                                     -F6*fh(i+2,j,k) + fh(i+3,j,k));
        }
    }

    // --- Y Direction ---
    if (vy > ZEO) {
        if (j + 3 <= jmax) {
            rhs_add += vy * d12dy * (-F3*fh(i,j-1,k) - F10*fh(i,j,k) + F18*fh(i,j+1,k)
                                     -F6*fh(i,j+2,k) + fh(i,j+3,k));
        } else if (j + 2 <= jmax) {
            rhs_add += vy * d12dy * (fh(i,j-2,k) - EIT*fh(i,j-1,k) + EIT*fh(i,j+1,k) - fh(i,j+2,k));
        } else if (j + 1 <= jmax) {
            rhs_add -= vy * d12dy * (-F3*fh(i,j+1,k) - F10*fh(i,j,k) + F18*fh(i,j-1,k)
                                     -F6*fh(i,j-2,k) + fh(i,j-3,k));
        }
    } else if (vy < ZEO) {
        if (j - 3 >= jmin) {
            rhs_add -= vy * d12dy * (-F3*fh(i,j+1,k) - F10*fh(i,j,k) + F18*fh(i,j-1,k)
                                     -F6*fh(i,j-2,k) + fh(i,j-3,k));
        } else if (j - 2 >= jmin) {
            rhs_add += vy * d12dy * (fh(i,j-2,k) - EIT*fh(i,j-1,k) + EIT*fh(i,j+1,k) - fh(i,j+2,k));
        } else if (j - 1 >= jmin) {
            rhs_add += vy * d12dy * (-F3*fh(i,j-1,k) - F10*fh(i,j,k) + F18*fh(i,j+1,k)
                                     -F6*fh(i,j+2,k) + fh(i,j+3,k));
        }
    }

    // --- Z Direction ---
    if (vz > ZEO) {
        if (k + 3 <= kmax) {
            rhs_add += vz * d12dz * (-F3*fh(i,j,k-1) - F10*fh(i,j,k) + F18*fh(i,j,k+1)
                                     -F6*fh(i,j,k+2) + fh(i,j,k+3));
        } else if (k + 2 <= kmax) {
            rhs_add += vz * d12dz * (fh(i,j,k-2) - EIT*fh(i,j,k-1) + EIT*fh(i,j,k+1) - fh(i,j,k+2));
        } else if (k + 1 <= kmax) {
            rhs_add -= vz * d12dz * (-F3*fh(i,j,k+1) - F10*fh(i,j,k) + F18*fh(i,j,k-1)
                                     -F6*fh(i,j,k-2) + fh(i,j,k-3));
        }
    } else if (vz < ZEO) {
        if (k - 3 >= kmin) {
            rhs_add -= vz * d12dz * (-F3*fh(i,j,k+1) - F10*fh(i,j,k) + F18*fh(i,j,k-1)
                                     -F6*fh(i,j,k-2) + fh(i,j,k-3));
        } else if (k - 2 >= kmin) {
            rhs_add += vz * d12dz * (fh(i,j,k-2) - EIT*fh(i,j,k-1) + EIT*fh(i,j,k+1) - fh(i,j,k+2));
        } else if (k - 1 >= kmin) {
            rhs_add += vz * d12dz * (-F3*fh(i,j,k-1) - F10*fh(i,j,k) + F18*fh(i,j,k+1)
                                     -F6*fh(i,j,k+2) + fh(i,j,k+3));
        }
    }

    (void)f_rhs;
    return rhs_add;
}