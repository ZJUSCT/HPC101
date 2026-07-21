
#ifndef FADMQUANTITES_H
#define FADMQUANTITES_H

#ifdef fortran1
#define f_admmass_bssn admmass_bssn
#define f_admmass_bssn_ss admmass_bssn_ss
#define f_admmomentum_bssn admmomentum_bssn
#endif
#ifdef fortran2
#define f_admmass_bssn ADMMASS_BSSN
#define f_admmass_bssn_ss ADMMASS_BSSN_SS
#define f_admmomentum_bssn ADMMOMENTUM_BSSN
#endif
#ifdef fortran3
#define f_admmass_bssn admmass_bssn_
#define f_admmass_bssn_ss admmass_bssn_ss_
#define f_admmomentum_bssn admmomentum_bssn_
#endif

extern "C"
{
	void f_admmass_bssn(int *, double *, double *, double *,
						double *, double *,
						double *, double *, double *, double *, double *, double *,
						double *, double *, double *, double *, double *, double *,
						double *, double *, double *,
						double *, double *, double *,
						int &);
}

#ifdef USE_GPU
#include <cuda_runtime.h>
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
);
#endif

#endif /* FADMQUANTITES_H */
