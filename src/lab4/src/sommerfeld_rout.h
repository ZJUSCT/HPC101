
#ifndef SOMMERFELD_ROUT_H
#define SOMMERFELD_ROUT_H

#ifdef fortran1
#define f_sommerfeld_rout sommerfeld_rout
#define f_sommerfeld_routbam sommerfeld_routbam
#define f_sommerfeld_routbam_ss sommerfeld_routbam_ss
#define f_falloff_ss falloff_ss
#endif
#ifdef fortran2
#define f_sommerfeld_rout SOMMERFELD_ROUT
#define f_sommerfeld_rout SOMMERFELD_ROUTBAM
#define f_sommerfeld_rout_ss SOMMERFELD_ROUTBAM_SS
#define f_falloff_ss FALLOFF_SS
#endif
#ifdef fortran3
#define f_sommerfeld_rout sommerfeld_rout_
#define f_sommerfeld_routbam sommerfeld_routbam_
#define f_sommerfeld_routbam_ss sommerfeld_routbam_ss_
#define f_falloff_ss falloff_ss_
#endif

extern "C"
{
	void f_sommerfeld_rout(int *, double *, double *, double *,
						   double &, double &, double &, double &, double &, double &, double &, double *,
						   double *, double *, double *, double *,
						   int &, int &);
}

extern "C"
{
	void f_sommerfeld_routbam(int *, double *, double *, double *,
							  double &, double &, double &, double &, double &, double &, double *,
							  double *, double &, double *, int &);
}

extern "C"
{
	void f_sommerfeld_routbam_ss(int *, double *, double *, double *,
								 double &, double &, double &, double &, double &, double &, double *,
								 double *, double &, double *, int &);
}

extern "C"
{
	void f_falloff_ss(int *, double *, double *, double *,
					  double &, double &, double &, double &, double &, double &, double *,
					  int &, double *, int &);
}

#ifdef USE_GPU
#include <cuda_runtime.h>
void gpu_sommerfeld_rout_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor
);

void gpu_sommerfeld_routbam_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double* d_f_rhs, const double* d_f0,
    double velocity, const double SoA[3], int Symmetry
);
#endif

#endif /* SOMMERFELD_ROUT_H */
