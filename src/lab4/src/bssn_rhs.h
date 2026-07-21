
#ifndef BSSN_H
#define BSSN_H

#ifdef fortran1
#define f_compute_rhs_bssn compute_rhs_bssn
#endif
#ifdef fortran2
#define f_compute_rhs_bssn COMPUTE_RHS_BSSN
#endif
#ifdef fortran3
#define f_compute_rhs_bssn compute_rhs_bssn_
#endif
extern "C"
{
        int f_compute_rhs_bssn(int *, double &, double *, double *, double *,                                                      // ex,T,X,Y,Z
                               double *, double *,                                                                                 // chi, trK
                               double *, double *, double *, double *, double *, double *,                                         // gij
                               double *, double *, double *, double *, double *, double *,                                         // Aij
                               double *, double *, double *,                                                                       // Gam
                               double *, double *, double *, double *, double *, double *, double *,                               // Gauge
                               double *, double *,                                                                                 // chi, trK
                               double *, double *, double *, double *, double *, double *,                                         // gij
                               double *, double *, double *, double *, double *, double *,                                         // Aij
                               double *, double *, double *,                                                                       // Gam
                               double *, double *, double *, double *, double *, double *, double *,                               // Gauge
                               double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, // stress-energy
                               double *, double *, double *, double *, double *, double *,                                         // Christoffel
                               double *, double *, double *, double *, double *, double *,                                         // Christoffel
                               double *, double *, double *, double *, double *, double *,                                         // Christoffel
                               double *, double *, double *, double *, double *, double *,                                         // Ricci
                               double *, double *, double *, double *, double *, double *, double *,                               // constraint violation
                               int &, int &, double &, int &);
}

#ifdef USE_GPU
#include <cuda_runtime.h>
void gpu_compute_rhs_bssn_launch( // launch kernel with device pointers
    cudaStream_t &stream,
    int* ex, double T, double* d_X, double* d_Y, double* d_Z,
    double* d_chi, double* d_trK,
    double* d_dxx, double* d_gxy, double* d_gxz,
    double* d_dyy, double* d_gyz, double* d_dzz,
    double* d_Axx, double* d_Axy, double* d_Axz,
    double* d_Ayy, double* d_Ayz, double* d_Azz,
    double* d_Gamx, double* d_Gamy, double* d_Gamz,
    double* d_Lap,
    double* d_betax, double* d_betay, double* d_betaz,
    double* d_dtSfx, double* d_dtSfy, double* d_dtSfz,
    double* d_chi_rhs, double* d_trK_rhs,
    double* d_gxx_rhs, double* d_gxy_rhs, double* d_gxz_rhs,
    double* d_gyy_rhs, double* d_gyz_rhs, double* d_gzz_rhs,
    double* d_Axx_rhs, double* d_Axy_rhs, double* d_Axz_rhs,
    double* d_Ayy_rhs, double* d_Ayz_rhs, double* d_Azz_rhs,
    double* d_Gamx_rhs, double* d_Gamy_rhs, double* d_Gamz_rhs,
    double* d_Lap_rhs,
    double* d_betax_rhs, double* d_betay_rhs, double* d_betaz_rhs,
    double* d_dtSfx_rhs, double* d_dtSfy_rhs, double* d_dtSfz_rhs,
    double* d_rho, double* d_Sx, double* d_Sy, double* d_Sz,
    double* d_Sxx, double* d_Sxy, double* d_Sxz,
    double* d_Syy, double* d_Syz, double* d_Szz,
    double* d_Gamxxx, double* d_Gamxxy, double* d_Gamxxz,
    double* d_Gamxyy, double* d_Gamxyz, double* d_Gamxzz,
    double* d_Gamyxx, double* d_Gamyxy, double* d_Gamyxz,
    double* d_Gamyyy, double* d_Gamyyz, double* d_Gamyzz,
    double* d_Gamzxx, double* d_Gamzxy, double* d_Gamzxz,
    double* d_Gamzyy, double* d_Gamzyz, double* d_Gamzzz,
    double* d_Rxx, double* d_Rxy, double* d_Rxz,
    double* d_Ryy, double* d_Ryz, double* d_Rzz,
    double* d_ham_Res, double* d_movx_Res, double* d_movy_Res, double* d_movz_Res,
    double* d_Gmx_Res, double* d_Gmy_Res, double* d_Gmz_Res,
    int symmetry, int lev, double eps, int co
);
#endif

#endif /* BSSN_H */
