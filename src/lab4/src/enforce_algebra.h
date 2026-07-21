
#ifndef ENFORCE_ALGEBRA_H
#define ENFORCE_ALGEBRA_H

#ifdef fortran1
#define f_enforce_ag enforce_ag
#define f_enforce_ga enforce_ga
#endif
#ifdef fortran2
#define f_enforce_ag ENFORCE_AG
#define f_enforce_ga ENFORCE_GA
#endif
#ifdef fortran3
#define f_enforce_ag enforce_ag_
#define f_enforce_ga enforce_ga_
#endif

extern "C"
{
	void f_enforce_ag(int *,
					  double *, double *, double *, double *, double *, double *,
					  double *, double *, double *, double *, double *, double *);
}
extern "C"
{
	void f_enforce_ga(int *,
					  double *, double *, double *, double *, double *, double *,
					  double *, double *, double *, double *, double *, double *);
}

#ifdef USE_GPU
#include <cuda_runtime.h>
void gpu_enforce_ga_launch(
    cudaStream_t &stream,
    int ex[3],
    double* d_dxx, double* d_gxy, double* d_gxz,
    double* d_dyy, double* d_gyz, double* d_dzz,
    double* d_Axx, double* d_Axy, double* d_Axz,
    double* d_Ayy, double* d_Ayz, double* d_Azz
);
#endif

#endif /* ENFORCE_ALGEBRA_H */
