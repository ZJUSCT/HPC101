#include <iostream>
#include <iomanip>
#include <fstream>
#include <strstream>
#include <cmath>
#include <cstdio>
using namespace std;

#include "Ansorg.h"
#include <cstring>
/* read spectral data from file
   special: pad phi direction with ghosts for periodic interpolation
            order = 4:    (-2 -1) 0 ...  n-1 (n n+1)
*/
Ansorg::Ansorg(char *filename, int orderi) : pu_ps(0), coordA(0), coordB(0), coordphi(0)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  order = orderi / 2 * 2; // order must be even
  PIh = PI / 2.0;
  char s[1000], *t;
  FILE *fp;
  double *v;
  int nghosts;
  int i;

  double x1, y1, z1, x2, y2, z2, dx, dy;

  /* open file */
  fp = fopen(filename, "r");
  if (myrank == 0 && !fp)
  {
    cout << "could not open " << filename << " for reading Ansorg" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (myrank == 0)
    printf("  reading data from %s\n", filename);

  /* skip to line starting with data, extract size info */
  n1 = n2 = n3 = ntotal = -1;
  while (fgets(s, 1000, fp))
  {
    t = strstr(s, "bhx1 ");
    if (t == s)
      sscanf(s + 15, "%lf", &x1);
    t = strstr(s, "bhy1 ");
    if (t == s)
      sscanf(s + 15, "%lf", &y1);
    t = strstr(s, "bhz1 ");
    if (t == s)
      sscanf(s + 15, "%lf", &z1);
    t = strstr(s, "bhx2 ");
    if (t == s)
      sscanf(s + 15, "%lf", &x2);
    t = strstr(s, "bhy2 ");
    if (t == s)
      sscanf(s + 15, "%lf", &y2);
    t = strstr(s, "bhz2 ");
    if (t == s)
      sscanf(s + 15, "%lf", &z2);

    t = strstr(s, "data ");
    if (t != s)
      continue;
    sscanf(s + 5, "%d%d%d", &n1, &n2, &n3);
    ntotal = n1 * n2 * n3;
    if (myrank == 0)
      printf("  found data with dimensions %d x %d x %d = %d\n",
             n1, n2, n3, ntotal);
    break;
  }

  if (myrank == 0)
    cout << "  bhx1 = " << x1 << endl
         << "  bhy1 = " << y1 << endl
         << "  bhz1 = " << z1 << endl
         << "  bhx2 = " << x2 << endl
         << "  bhy2 = " << y2 << endl
         << "  bhz2 = " << z2 << endl;

  dx = x1 - x2;
  dy = y1 - y2;

  /* x-axis */
  if (dx != 0 && y1 == 0 && y2 == 0 && z1 == 0 && z2 == 0)
  {
    ps_b = dx / 2;
    ps_dx = (x1 + x2) / 2;
    ps_rxx = 1;
    ps_rxy = 0;
    ps_ryx = 0;
    ps_ryy = 1;
  }

  /* y-axis */
  else if (dy != 0 && x1 == 0 && x2 == 0 && z1 == 0 && z2 == 0)
  {
    ps_b = dy / 2;
    ps_dx = (y1 + y2) / 2;
    ps_rxx = 0;
    ps_rxy = +1;
    ps_ryx = -1;
    ps_ryy = 0;
  }

  /* else */
  else if (myrank == 0)
  {
    cout << "puncture location not allowed" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (ntotal == -1 && myrank == 0)
  {
    cout << "file does not contain the expected data" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* get storage if needed */
  int pad = order / 2;
  nghosts = n1 * n2 * pad;
  if (!(pu_ps))
    pu_ps = new double[ntotal + 2 * nghosts];
  v = pu_ps + nghosts;

  /* read data */
  i = 0;
  while (fgets(s, 1000, fp))
  {
    if (i < ntotal)
      v[i] = atof(t);
    i++;
  }
  if (myrank == 0)
  {
    printf("  read %d data lines\n", i);
    cout << endl;
  }
  if (myrank == 0 && i < ntotal)
  {
    cout << "file contains too few data lines" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (myrank == 0 && i > ntotal)
  {
    cout << "file contains too many data lines" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* copy data into ghosts */
  for (i = 0; i < nghosts; i++)
  {
    (pu_ps)[i] = v[i + ntotal - nghosts];
    (pu_ps)[i + ntotal + nghosts] = v[i];
  }

  /* done */
  fclose(fp);

  set_ABp();
}
Ansorg::~Ansorg()
{
  if (coordA)
    delete[] coordA;
  if (coordB)
    delete[] coordB;
  if (coordphi)
    delete[] coordphi;
  if (pu_ps)
    delete[] pu_ps;
}
/* interpolate to point given in Cartesian coordinates
   calls function in utility/interpolation/barycentric.c
*/
double Ansorg::ps_u_at_xyz(double x, double y, double z)
{
  double A, B, phi, u, U;
  /*
  // rotate THETA along clockwise direction
  #define THETA (PI*0.25)
     A = x;
     B = y;
     x = A*cos(THETA)+B*sin(THETA);
     y =-A*sin(THETA)+B*cos(THETA);
  */
  xyz_to_ABp(x, y, z, &A, &B, &phi);

  U = interpolate_tri_bar(A, B, phi, n1, n2, n3 + (order / 2) * 2,
                          coordA, coordB, coordphi, pu_ps);
  u = 2 * (A - 1) * U;
  if (U > 0.025)
    cout << x << "," << y << "," << z << "," << A << "," << B << "," << phi << "," << U << "," << u << endl;
  if (!finite(u))
  {
    cout << "find NaN in Ansorg::ps_u_at_xyz at (" << x << "," << y << "," << z << ")" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return u;
}
/* set 1d arrays for spectral coordinates
   see Punctures_functions.c for reference
   special: pad phi direction with ghosts for periodicity
*/
void Ansorg::set_ABp()
{
  int pad = order / 2;
  int i;
  double Acode;
  int pr = 0;

  coordA = new double[n1];
  coordB = new double[n2];
  coordphi = new double[n3 + 2 * pad];

  for (i = 0; i < n1; i++)
  {
    Acode = -cos(PIh * (2 * i + 1) / n1);
    coordA[i] = (Acode + 1) / 2;
    if (pr && myrank == 0)
      printf("coordA[%2d] = %f\n", i, coordA[i]);
  }

  for (i = 0; i < n2; i++)
  {
    coordB[i] = -cos(PIh * (2 * i + 1) / n2);
    if (pr && myrank == 0)
      printf("coordB[%2d] = %f\n", i, coordB[i]);
  }

  for (i = 0; i < n3 + 2 * pad; i++)
  {
    coordphi[i] = 2 * PI * (i - pad) / n3;
    if (pr && myrank == 0)
      printf("coordphi[%2d] = %f  %f\n",
             i, coordphi[i], coordphi[i] * 180 / PI);
  }
}
/* from cartesian to spectral
   see coordtrans.m etc
   The problem is that the inverse transformation requires several
   nested square roots with 8 possible solutions, only one of them relevant.
   We have picked the correct solution by testing in Mathematica.
   Furthermore, there are special coordinates where the formulas have
   to be specialized.

   fixme: needs proper treatment of quantities that are almost zero/singular
*/
// adopting the coordinate transformation in TwoPunctures Thorn on Jan 23, 2011
void Ansorg::xyz_to_ABp(double x, double y, double z,
                        double *A, double *B, double *phi)
{
  const double s2 = sqrt(2.0);
  const double exp = 3.0 / 2.0;

  double r, rr, xx;
  double t, st, u, su, v, sv, w, sw;

  /* rotate onto x-axis if required */
  w = x;
  x = ps_rxx * w + ps_rxy * y;
  y = ps_ryx * w + ps_ryy * y;

  /* center black holes at +b and -b */
  x -= ps_dx;

  /* offset parameter b rescales the coordinates */
  x /= ps_b;
  y /= ps_b;
  z /= ps_b;

  /* helpers */
  r = sqrt(y * y + z * z);
  rr = r * r;
  xx = x * x;
  /* this work worse than the next one
    *phi = atan2(z, y);
    if (*phi < 0) *phi += 2 * PI;
  */
  if (r > 0)
    *phi = (z < 0) ? 2 * PI - acos(y / r) : acos(y / r);
  else
    *phi = 0;

  double aux1 = 0.5 * (x * x + rr - 1);
  double aux2 = sqrt(aux1 * aux1 + rr);
  double X = asinh(sqrt(aux1 + aux2));
  double R = asin(min(1.0, sqrt(-aux1 + aux2)));
  if (x < 0)
    R = PI - R;

  *A = tanh(0.5 * X);
  *B = tan(0.5 * R - PI / 4);
}

/* three dimensional polynomial interpolation, barycentric */
double Ansorg::interpolate_tri_bar(double x, double y, double z,
                                   int n1, int n2, int n3,
                                   double *x1, double *x2, double *x3, double *yp)
{
  double u;
  double *w, *omega;
  double **v;

  int i, j, k, ijk;
  int i1, i2, i3;
  int di = 1, dj = n1, dk = n1 * n2;
  int order1 = order > n1 ? n1 : order;
  int order2 = order > n2 ? n2 : order;
  int order3 = order > n3 ? n3 : order;

  w = new double[order];
  omega = new double[order];
  v = new double *[order];
  for (int i = 0; i < order; i++)
    v[i] = new double[order];

  i1 = find_point_bisection(x, n1, x1, order1 / 2);
  i2 = find_point_bisection(y, n2, x2, order2 / 2);
  i3 = find_point_bisection(z, n3, x3, order3 / 2);
  ijk = i1 * di + i2 * dj + i3 * dk;
  if (0)
    printf("%d %d %d\n", i1, i2, i3);

  barycentric_omega(order1, 1, &x1[i1], omega);
  for (k = 0; k < order3; k++)
    for (j = 0; j < order2; j++)
      v[k][j] = barycentric(x, order1, 1, &x1[i1], &yp[ijk + j * dj + k * dk], omega);

  if (0)
    for (k = 0; k < order3; k++)
      for (j = 0; j < order2; j++)
        printf("%2d %2d   %.15f\n", k, j, v[k][j]);

  barycentric_omega(order2, 1, &x2[i2], omega);
  for (k = 0; k < order3; k++)
    w[k] = barycentric(y, order2, 1, &x2[i2], &v[k][0], omega);

  if (0)
    for (k = 0; k < order3; k++)
      printf("%2d %.15f\n", k, w[k]);

  barycentric_omega(order3, 1, &x3[i3], omega);
  u = barycentric(z, order3, 1, &x3[i3], w, omega);

  if (!finite(u))
  {
    cout << "find NaN in Ansorg::interpolate_tri_bar at (" << x << "," << y << "," << z << ")" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (i = 0; i < order; i++)
    delete[] v[i];

  delete[] w;
  delete[] omega;
  delete[] v;

  return u;
}
/* find index such that xp[i] <= x < xp[i+1]
   uses bisection, which relies on x being ordered
   o is "offset", number of points smaller than x that are required
   returns j = i-(o-1), i.e. if o = 2, then
     xp[j] < xp[j+1] <= x < xp[j+2] < xp[j+3]
   which is useful for interpolation
*/
int Ansorg::find_point_bisection(double x, int n, double *xp, int o)
{
  int i0 = o - 1, i1 = n - o;
  int i;

  if (n < 2 * o)
  {
    cout << "bisection failed" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (x <= xp[i0])
    return 0;
  if (x > xp[i1])
    return n - 2 * o;

  while (i0 != i1 - 1)
  {
    i = (i0 + i1) / 2;
    if (x < xp[i])
      i1 = i;
    else
      i0 = i;
  }

  return i0 - o + 1;
}
/* compute omega[] for barycentric interpolation */
// SIAM_review 46, 501 (2004)
void Ansorg::barycentric_omega(int n, int s, double *x, double *omega)
{
  double o;
  int i, j;

  if (0)
    printf("%d %d %p %p\n", n, s, x, omega);

  for (i = 0; i < n; i += s)
  {
    o = 1;
    for (j = 0; j < n; j += s)
    {
      if (j != i)
      {
        o /= (x[i] - x[j]);
      }
    }
    omega[i / s] = o;

    if (0)
      printf("x[%d] = %9.6f omega[%d] = %13.6e\n", i / s, x[i], i / s, o);
  }
}
/* barycentric interpolation with precomputed omega */
double Ansorg::barycentric(double x0, int n, int s, double *x, double *y,
                           double *omega)
{
  double a, b, c, d;
  int i;

  if (0)
    printf("%f %d %d %p %p %p\n", x0, n, s, x, y, omega);

  a = b = 0;
  for (i = 0; i < n; i += s)
  {
    d = x0 - x[i];
    if (d == 0)
      return y[i];
    c = omega[i / s] / d;
    b += c;
    a += c * y[i];
  }

  return a / b;
}
