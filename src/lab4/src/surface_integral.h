//$Id: surface_integral.h,v 1.9 2013/08/20 11:49:05 zjcao Exp $
#ifndef SURFACE_INTEGRAL_H
#define SURFACE_INTEGRAL_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <strstream>
#include <cmath>
using namespace std;

#include "cgh.h"
#include "var.h"
#include "monitor.h"

class surface_integral
{

private:
	int Symmetry, factor;
	int N_theta, N_phi; // Number of points in Theta & Phi directions
	double dphi, dcostheta;
	double *arcostheta, *wtcostheta;
	int n_tot; // size of arrays

	double *nx_g, *ny_g, *nz_g; // global list of unit normals
	int myrank, cpusize;

#ifdef USE_GPU
	// Shadow variables for surface integral (GPU)
	cudaStream_t stream;
	double *d_arcostheta, *d_wtcostheta;
	double *d_nx_g, *d_ny_g, *d_nz_g;
#endif

public:
	surface_integral(int iSymmetry);
	~surface_integral();

	void surf_Wave(double rex, int lev, cgh *GH, var *Rpsi4, var *Ipsi4,
				   int spinw, int maxl, int NN, double *RP, double *IP,
				   monitor *Monitor); // NN is the length of RP and IP
									  // this routine can only deal with the symmetry of Psi4
	void surf_Wave(double rex, int lev, cgh *GH,
				   var *Ex, var *Ey, var *Ez, var *Bx, var *By, var *Bz,
				   var *chi, var *gxx, var *gxy, var *gxz, var *gyy, var *gyz, var *gzz,
				   int spinw, int maxl, int NN, double *RP, double *IP,
				   monitor *Monitor,
				   void (*funcs)(double &, double &, double &,
								 double &, double &, double &, double &, double &, double &, double &,
								 double &, double &, double &, double &, double &, double &,
								 double &, double &)); // NN is the length of RP and IP
	void surf_MassPAng(double rex, int lev, cgh *GH, var *chi, var *trK,
					   var *gxx, var *gxy, var *gxz, var *gyy, var *gyz, var *gzz,
					   var *Axx, var *Axy, var *Axz, var *Ayy, var *Ayz, var *Azz,
					   var *Gmx, var *Gmy, var *Gmz,
					   var *Sfx_rhs, var *Sfy_rhs, var *Sfz_rhs,
					   double *Rout, monitor *Monitor);
	void surf_MassPAng(double rex, int lev, cgh *GH, var *chi, var *trK,
					   var *gxx, var *gxy, var *gxz, var *gyy, var *gyz, var *gzz,
					   var *Axx, var *Axy, var *Axz, var *Ayy, var *Ayz, var *Azz,
					   var *Gmx, var *Gmy, var *Gmz,
					   var *Sfx_rhs, var *Sfy_rhs, var *Sfz_rhs, // temparay memory for mass^i
					   double *Rout, monitor *Monitor, MPI_Comm Comm_here);
	void surf_Wave(double rex, int lev, cgh *GH, var *Rpsi4, var *Ipsi4,
				   int spinw, int maxl, int NN, double *RP, double *IP,
				   monitor *Monitor, MPI_Comm Comm_here);
#ifdef USE_GPU
	void gpu_surf_MassPAng(
		double rex, int lev, cgh *GH, var *chi, var *trK,
		var *gxx, var *gxy, var *gxz, var *gyy, var *gyz, var *gzz,
		var *Axx, var *Axy, var *Axz, var *Ayy, var *Ayz, var *Azz,
		var *Gmx, var *Gmy, var *Gmz,
		var *Sfx_rhs, var *Sfy_rhs, var *Sfz_rhs, // temparay memory for mass^i
		double *Rout, monitor *Monitor
	);
	void gpu_surf_Wave(
		double rex, int lev, cgh *GH, var *Rpsi4, var *Ipsi4,
		int spinw, int maxl, int NN, double *RP, double *IP,
		monitor *Monitor
	);
#endif
};

#endif /* SURFACE_INTEGRAL_H */
