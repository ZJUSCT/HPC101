
#ifndef Ansorg_H
#define Ansorg_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

#include <mpi.h>

#define PI M_PI

class Ansorg
{
protected:
  int n1, n2, n3, ntotal;
  int order;
  double *coordA, *coordB, *coordphi;
  int ps_rxx, ps_rxy, ps_ryx, ps_ryy;
  double ps_b, ps_dx;
  double PIh;
  double *pu_ps;
  int myrank;

public:
  Ansorg(char *filename, int orderi);
  ~Ansorg();
  double ps_u_at_xyz(double x, double y, double z);
  void set_ABp();
  void xyz_to_ABp(double x, double y, double z,
                  double *A, double *B, double *phi);
  double interpolate_tri_bar(double x, double y, double z,
                             int n1, int n2, int n3,
                             double *x1, double *x2, double *x3, double *yp);
  int find_point_bisection(double x, int n, double *xp, int o);
  void barycentric_omega(int n, int s, double *x, double *omega);
  double barycentric(double x0, int n, int s, double *x, double *y,
                     double *omega);
};
#endif /* Ansorg_H */
