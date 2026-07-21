
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <strstream>
#include <cmath>
using namespace std;

#include <mpi.h>

#include "misc.h"
#include "macrodef.h"

#define PI M_PI

void misc::tillherecheck(int myrank)
{
  int atp = 1, tatp;
  MPI_Allreduce(&atp, &tatp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (myrank == 0)
    cout << " here now: " << tatp << " processors." << endl;
}
void misc::tillherecheck(const char str[])
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  int atp = 1, tatp;
  MPI_Allreduce(&atp, &tatp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (myrank == 0)
  {
    cout << " here now: " << tatp << " processors." << endl;
    cout << str << endl;
  }
}
void misc::tillherecheck(MPI_Comm Comm_here, int out_rank, const char str[])
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  int atp = 1, tatp;

  MPI_Allreduce(&atp, &tatp, 1, MPI_INT, MPI_SUM, Comm_here);
  if (myrank == out_rank)
  {
    cout << " here now: " << tatp << " processors." << endl;
    cout << str << endl;
  }
}
void misc::tillherecheck(MPI_Comm Comm_here, int out_rank, const string str)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  int atp = 1, tatp;

  MPI_Allreduce(&atp, &tatp, 1, MPI_INT, MPI_SUM, Comm_here);
  if (myrank == out_rank)
  {
    cout << " here now: " << tatp << " processors." << endl;
    cout << str << endl;
  }
}
// pick out value from input string
int misc::parse_parts(string str, string &sgrp, string &skey, string &sval, int &ind)
{
  int pos1, pos2;
  string s0;

  ind = 0;

  // remove comments
  str = str.substr(0, str.find("#"));
  if (rTrim(str).empty())
    return 0; // continue;

  // parse {group, key, val}
  pos1 = str.find("::");
  pos2 = str.find("=");
  if (pos1 == string::npos || pos2 == string::npos)
    return -1;

  s0 = str.substr(0, pos1);
  sgrp = lTrim(s0);
  s0 = str.substr(pos1 + 2, pos2 - pos1 - 2);
  skey = rTrim(s0);
  s0 = str.substr(pos2 + 1);
  sval = Trim(s0);

  pos1 = sval.find("\"");
  pos2 = sval.rfind("\"");
  if (pos1 != string::npos)
  {
    sval = sval.substr(1, pos2 - 1);
  }

  pos1 = skey.find("[");
  pos2 = skey.find("]");
  if (pos1 != string::npos)
  {
    s0 = skey.substr(0, pos1);
    ind = atoi(skey.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
    skey = s0;
  }

  return 1;
}
int misc::parse_parts(string str, string &sgrp, string &skey, string &sval, int &ind1, int &ind2)
{
  int pos1, pos2;
  string s0, s1;

  ind1 = ind2 = 0;

  // remove comments
  str = str.substr(0, str.find("#"));
  if (rTrim(str).empty())
    return 0; // continue;

  // parse {group, key, val}
  pos1 = str.find("::");
  pos2 = str.find("=");
  if (pos1 == string::npos || pos2 == string::npos)
    return -1;

  s0 = str.substr(0, pos1);
  sgrp = lTrim(s0);
  s0 = str.substr(pos1 + 2, pos2 - pos1 - 2);
  skey = rTrim(s0);
  s0 = str.substr(pos2 + 1);
  sval = Trim(s0);

  pos1 = sval.find("\"");
  pos2 = sval.rfind("\"");
  if (pos1 != string::npos)
  {
    sval = sval.substr(1, pos2 - 1);
  }

  pos1 = skey.find("[");
  pos2 = skey.find("]");
  if (pos1 != string::npos)
  {
    s0 = skey.substr(0, pos1);
    s1 = skey.substr(pos2 + 1);
    ind1 = atoi(skey.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
    skey = s0;
  }

  pos1 = s1.find("[");
  pos2 = s1.find("]");
  if (pos1 != string::npos)
  {
    s0 = s1.substr(pos2 + 1);
    ind2 = atoi(s1.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
  }

  return 1;
}
int misc::parse_parts(string str, string &sgrp, string &skey, string &sval, int &ind1, int &ind2, int &ind3)
{
  int pos1, pos2;
  string s0, s1;

  ind1 = ind2 = ind3 = 0;

  // remove comments
  str = str.substr(0, str.find("#"));
  if (rTrim(str).empty())
    return 0; // continue;

  // parse {group, key, val}
  pos1 = str.find("::");
  pos2 = str.find("=");
  if (pos1 == string::npos || pos2 == string::npos)
    return -1;

  s0 = str.substr(0, pos1);
  sgrp = lTrim(s0);
  s0 = str.substr(pos1 + 2, pos2 - pos1 - 2);
  skey = rTrim(s0);
  s0 = str.substr(pos2 + 1);
  sval = Trim(s0);

  pos1 = sval.find("\"");
  pos2 = sval.rfind("\"");
  if (pos1 != string::npos)
  {
    sval = sval.substr(1, pos2 - 1);
  }

  pos1 = skey.find("[");
  pos2 = skey.find("]");
  if (pos1 != string::npos)
  {
    s0 = skey.substr(0, pos1);
    s1 = skey.substr(pos2 + 1);
    ind1 = atoi(skey.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
    skey = s0;
  }

  pos1 = s1.find("[");
  pos2 = s1.find("]");
  if (pos1 != string::npos)
  {
    s0 = s1.substr(pos2 + 1);
    ind2 = atoi(s1.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
  }

  pos1 = s0.find("[");
  pos2 = s0.find("]");
  if (pos1 != string::npos)
  {
    ind3 = atoi(s0.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
  }

  return 1;
}
// sent me from Roman Gold on 2010-10-8
void misc::gaulegf(double x1, double x2, double *x, double *w, int n)
{
  int i, j, m;
  double eps = 1.2E-16;
  double p1, p2, p3, pp, xl, xm, z, z1;

  m = (n + 1) / 2;
  xm = 0.5 * (x2 + x1);
  xl = 0.5 * (x2 - x1);
  for (i = 0; i < m; i++)
  {
    z = cos(PI * ((double)i + 0.75) / ((double)n + 0.5));
    do
    {
      p1 = 1.0;
      p2 = 0.0;
      for (j = 0; j < n; j++)
      {
        p3 = p2;
        p2 = p1;
        p1 = ((2 * (double)j + 1) * z * p2 - (double)j * p3) / ((double)j + 1);
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z = z1 - p1 / pp;
    } while (fabs(z - z1) > eps);
    x[i] = xm - xl * z;
    x[n - 1 - i] = xm + xl * z;
    w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
    w[n - 1 - i] = w[i];
  }
} /* end gaulegf */
void misc::inversearray(double *aa, int NN)
{
  int i, m;
  m = (NN + 1) / 2;
  double rr;
  for (i = 0; i < m; i++)
  {
    rr = aa[i];
    aa[i] = aa[NN - 1 - i];
    aa[NN - 1 - i] = rr;
  }
}
// Eq.(42) of PRD 77, 024027 (2008)
double misc::Wigner_d_function(int l, int m, int s, double costheta)
{
  // we consider only theta in [0,pi]
  int C1 = max(0, m - s), C2 = min(l + m, l - s);

  double vv = 0;
  double sinht = sqrt((1 - costheta) / 2.0), cosht = sqrt((1 + costheta) / 2.0);
  if (C1 % 2 == 0)
  {
    for (int t = C1; t < C2 + 1; t += 2)
      vv = vv + pow(cosht, 2 * l + m - s - 2 * t) * pow(sinht, 2 * t + s - m) /
                    (fact(l + m - t) * fact(l - s - t) * fact(t) * fact(t + s - m));
    for (int t = C1 + 1; t < C2 + 1; t += 2)
      vv = vv - pow(cosht, 2 * l + m - s - 2 * t) * pow(sinht, 2 * t + s - m) /
                    (fact(l + m - t) * fact(l - s - t) * fact(t) * fact(t + s - m));
  }
  else
  {
    for (int t = C1; t < C2 + 1; t += 2)
      vv = vv - pow(cosht, 2 * l + m - s - 2 * t) * pow(sinht, 2 * t + s - m) /
                    (fact(l + m - t) * fact(l - s - t) * fact(t) * fact(t + s - m));
    for (int t = C1 + 1; t < C2 + 1; t += 2)
      vv = vv + pow(cosht, 2 * l + m - s - 2 * t) * pow(sinht, 2 * t + s - m) /
                    (fact(l + m - t) * fact(l - s - t) * fact(t) * fact(t + s - m));
  }
  return vv * sqrt(fact(l + m) * fact(l - m) * fact(l + s) * fact(l - s));
}
double misc::fact(int N)
{
  if (N < 0)
    cout << "error input for factorial." << endl;
  double f;
  if (N == 0)
    f = 1;
  else
    f = N * fact(N - 1);
  return f;
}
int misc::num_of_str(char *c)
{
  int NN = 0, N1 = 0;
  std::istringstream iss;
  iss.str(c);

  char c1[1000];
  while (!iss.eof())
  {
    iss >> c1;
    if (int(c1[0]) == 45 || int(c1[0]) == 46 || (int(c1[0]) > 47 && int(c1[0]) < 58))
      NN++;
    N1++;
  }

  char *c2 = c;
  while (*(c2 + 1))
    c2++;
  if (int(*c2) == 32)
  {
    NN--;
    N1--;
  }

  //      cout<<"found "<<N1<<" generalized data including "<<NN<<" number type data"<<endl;
  return NN;
}
// MNRAS 411, 2461 (2010)
// Eq.(20)-(22)
void misc::TVDrungekutta3(const int N, const double dT, double *f0,
                          double *f1, double *f_rhs, const int RK4)
{
  const double F3o4 = 0.75, F1o4 = 0.25, F1o3 = 1.0 / 3, F2o3 = 2.0 / 3;
  switch (RK4)
  {
  case 0:
    for (int i = 0; i < N; i++)
    {
      f1[i] = f0[i] + dT * f_rhs[i];
      f_rhs[i] = F1o4 * f1[i];
    }
    break;
  case 1:
    for (int i = 0; i < N; i++)
    {
      f1[i] = F3o4 * f0[i] + f_rhs[i] + F1o4 * dT * f1[i];
      f_rhs[i] = F2o3 * f1[i];
    }
    break;
  case 2:
    for (int i = 0; i < N; i++)
    {
      f1[i] = F1o3 * f0[i] + f_rhs[i] + F2o3 * dT * f1[i];
    }
    break;
  case 3:
    break;
  default:
    cout << "misc::rungekutta4: something is wrong in RK4 counting!!" << endl;
  }
}
void misc::rungekutta4(const int N, const double dT, double *f0,
                       double *f1, double *f_rhs, const int RK4)
{
  const double F1o6 = 1.0 / 6, HLF = 0.5, TWO = 2;
  switch (RK4)
  {
  case 0:
    for (int i = 0; i < N; i++)
      f1[i] = f0[i] + HLF * dT * f_rhs[i];
    break;
  case 1:
    for (int i = 0; i < N; i++)
    {
      f_rhs[i] = f_rhs[i] + TWO * f1[i];
      f1[i] = f0[i] + HLF * dT * f1[i];
    }
    break;
  case 2:
    for (int i = 0; i < N; i++)
    {
      f_rhs[i] = f_rhs[i] + TWO * f1[i];
      f1[i] = f0[i] + dT * f1[i];
    }
    break;
  case 3:
    for (int i = 0; i < N; i++)
      f1[i] = f0[i] + F1o6 * dT * (f1[i] + f_rhs[i]);
    break;
  default:
    cout << "misc::rungekutta4: something is wrong in RK4 counting!!" << endl;
  }
}
void misc::rungekutta4(const double dT, const std::vector<double> &f0,
                       std::vector<double> &f1, std::vector<double> &f_rhs, const int RK4)
{
  const int N = f0.size();
  const double F1o6 = 1.0 / 6, HLF = 0.5, TWO = 2;
  switch (RK4)
  {
  case 0:
    for (int i = 0; i < N; i++)
      f1[i] = f0[i] + HLF * dT * f_rhs[i];
    break;
  case 1:
    for (int i = 0; i < N; i++)
    {
      f_rhs[i] = f_rhs[i] + TWO * f1[i];
      f1[i] = f0[i] + HLF * dT * f1[i];
    }
    break;
  case 2:
    for (int i = 0; i < N; i++)
    {
      f_rhs[i] = f_rhs[i] + TWO * f1[i];
      f1[i] = f0[i] + dT * f1[i];
    }
    break;
  case 3:
    for (int i = 0; i < N; i++)
      f1[i] = f0[i] + F1o6 * dT * (f1[i] + f_rhs[i]);
    break;
  default:
    cout << "misc::rungekutta4: something is wrong in RK4 counting!!" << endl;
  }
}
void misc::dividBlock(const int DIM, int *shape_here, double *bbox_here, const int pices, double *picef, int *shape_res, double *bbox_res,
                      const int min_width)
{
  if (pices < 1)
  {
    cerr << "error in dividBlock: pices = " << pices << endl;
    return;
  }
  if (pices == 1)
  {
    for (int i = 0; i < DIM; i++)
    {
      shape_res[i] = shape_here[i];
      bbox_res[i] = bbox_here[i];
      bbox_res[DIM + i] = bbox_here[DIM + i];
    }
    return;
  }

  double dd = picef[0];
  for (int i = 1; i < pices; i++)
    dd += picef[i];

  if (feq(dd, 1, 1e-8))
  {
    int leg = shape_here[0];
    int legi = 0;
    for (int i = 1; i < DIM; i++)
    {
      if (leg < shape_here[i])
      {
        leg = shape_here[i];
        legi = i;
      }
    }

    int pic = 0;

    for (int ip = 0; ip < pices; ip++)
    {
      for (int i = 0; i < DIM; i++)
      {
        if (i == legi)
        {
          if (ip == pices - 1)
            shape_res[ip * DIM + i] = shape_here[i] - pic;
          else
          {
            shape_res[ip * DIM + i] = shape_here[i] * picef[ip];
            pic += shape_res[ip * DIM + i];
          }
        }
        else
          shape_res[ip * DIM + i] = shape_here[i];
      }
    }

    for (int ip = 0; ip < pices; ip++)
    {
      for (int i = 0; i < DIM; i++)
      {
        dd = (bbox_here[DIM + i] - bbox_here[i]) / shape_here[i];

        if (i == legi)
        {
          if (shape_res[ip * DIM + i] < min_width)
          {
            cerr << "dividBlock: resulted too small shape, shapeo = " << shape_here[i] << ", shape = " << shape_res[ip * DIM + i] << ", min_width = " << min_width << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
          }

          if (ip == 0)
            bbox_res[ip * 2 * DIM + i] = bbox_here[i];
          else
            bbox_res[ip * 2 * DIM + i] = bbox_res[(ip - 1) * 2 * DIM + DIM + i] - ghost_width * dd;

          if (ip == pices - 1)
            bbox_res[ip * 2 * DIM + DIM + i] = bbox_here[DIM + i];
          else
            bbox_res[ip * 2 * DIM + DIM + i] = bbox_res[ip * 2 * DIM + i] + shape_res[ip * DIM + i] * dd;

          if (ip > 0)
          {
            shape_res[ip * DIM + i] += ghost_width;
            bbox_res[ip * 2 * DIM + i] -= ghost_width * dd;
          }
          if (ip < pices - 1)
          {
            shape_res[ip * DIM + i] += ghost_width;
            bbox_res[ip * 2 * DIM + DIM + i] += ghost_width * dd;
          }
        }
        else
        {
          bbox_res[ip * 2 * DIM + i] = bbox_here[i];
          bbox_res[ip * 2 * DIM + DIM + i] = bbox_here[DIM + i];
        }
      }
    }
  }
  else
  {
    cerr << "error in dividBlock: ";
    for (int i = 0; i < pices; i++)
      cerr << picef[i] << " ";
    cerr << endl;
  }
}
void misc::swapvector(std::vector<double> &f0, std::vector<double> &f1)
{
  const int N = f0.size();
  double tt;
  for (int i = 0; i < N; i++)
  {
    tt = f0[i];
    f0[i] = f1[i];
    f1[i] = tt;
  }
}
complex<double> misc::complex_gamma(complex<double> z)
{
  const double p[9] = {0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                       771.32342877765313, -176.61502916214059, 12.507343278686905,
                       -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

  if (real(z) < 0.5)
  {
    return PI / (sin(PI * z) * complex_gamma(1.0 - z));
  }
  z -= 1.0;
  complex<double> x = p[0];
  for (int i = 1; i < 9; i++)
  {
    x += p[i] / (z + complex<double>(i, 0));
  }
  complex<double> t = z + (7 + 0.5);
  t = sqrt(2 * PI) * pow(t, z + 0.5) * exp(-t) * x;

  return t;
}
// also called Kummer function,
// Confluent hypergeometric function 1F1
complex<double> misc::KummerComplex(const complex<double> a, const complex<double> b, complex<double> x)
{
  // Default tolerance is tol = 1e-10.  Feel free to change this as needed.
  const double tol = 1e-10;

  // Estimates the value by summing powers of the generalized hypergeometric
  // series:
  //
  //       sum(n=0-->Inf)[(a)_n*x^n/{(b)_n*n!}]
  //
  // until the specified tolerance is acheived.

  complex<double> term = x * a / b;
  complex<double> f = 1.0 + term;
  int n = 1;
  complex<double> an = a;
  complex<double> bn = b;
  int nmin = 100000;

  while (n < nmin && (abs(term)) > tol)
  {
    n = n + 1;
    an = an + 1.0;
    bn = bn + 1.0;
    term = x * term * an / bn / double(n);
    f = f + term;
  }

  if ((abs(term)) > tol && n == nmin)
    cout << "misc::KummerComplex has n > " << nmin << " with error " << abs(term) << endl
         << "a = " << a << " b = " << b << " x = " << x << endl;

  return f;
}

complex<double> misc::Rec_Int(const double xmin, const double xmax, complex<double> fun(double x))
{
  // Default tolerance is tol = 1e-10.  Feel free to change this as needed.
  const double tol = 1e-8;

  int N = int(xmax - xmin) * 10;
  if (N < 1000)
    N = 1000;
  double dx = (xmax - xmin) / (N - 1);
  complex<double> sum = 0, sum2 = 0;
  for (int i = 0; i < N; i++)
  {
    sum2 += fun(xmin + i * dx);
  }
  sum2 = sum2 * dx;

  int j = 1;
  const int jmax = 10;
  while (j < jmax && abs(sum2 - sum) > tol)
  {
    j++;
    N = N * 2;
    dx = (xmax - xmin) / (N - 1);
    sum = sum2;
    sum2 = 0;
    for (int i = 0; i < N; i++)
    {
      sum2 += fun(xmin + i * dx);
    }
    sum2 = sum2 * dx;

    //    cout<<"j = "<<j<<" error = "<<abs(sum2-sum)<<endl;
  }

  if (j == jmax)
    cout << "misc::Rec_Int has j > " << jmax << ", error = " << abs(sum2 - sum) << endl;

  return sum2;
}
complex<double> misc::Simpson_Int(const double xmin, const double xmax, complex<double> fun(double x))
{
  // Default tolerance is tol = 1e-10.  Feel free to change this as needed.
  const double tol = 1e-8;

  int N = 1000;
  double dx = (xmax - xmin) / (N - 1);
  complex<double> sum = 0, sum2 = 0;
  sum2 = 17.0 * fun(xmin) + 59.0 * fun(xmin + dx) + 43.0 * fun(xmin + 2 * dx) + 49.0 * fun(xmin + 3 * dx);
  for (int i = 4; i < N - 3; i++)
  {
    sum2 += 48.0 * fun(xmin + i * dx);
  }
  sum2 = sum2 + 17.0 * fun(xmax) + 59.0 * fun(xmax - dx) + 43.0 * fun(xmax - 2 * dx) + 49.0 * fun(xmax - 3 * dx);
  sum2 = sum2 * dx / 48.0;

  int j = 1;
  const int jmax = 50;
  while (j < jmax && abs(sum2 - sum) > tol)
  {
    j++;
    N = N * 2;
    dx = (xmax - xmin) / (N - 1);
    sum = sum2;
    sum2 = 17.0 * fun(xmin) + 59.0 * fun(xmin + dx) + 43.0 * fun(xmin + 2 * dx) + 49.0 * fun(xmin + 3 * dx);
    for (int i = 4; i < N - 3; i++)
    {
      sum2 += 48.0 * fun(xmin + i * dx);
    }
    sum2 = sum2 + 17.0 * fun(xmax) + 59.0 * fun(xmax - dx) + 43.0 * fun(xmax - 2 * dx) + 49.0 * fun(xmax - 3 * dx);
    sum2 = sum2 * dx / 48.0;

    //    cout<<"j = "<<j<<" error = "<<abs(sum2-sum)<<endl;
  }

  if (j == jmax)
    cout << "misc::Simpson_Int has j > " << jmax << ", error = " << abs(sum2 - sum) << endl;

  return sum2;
}
complex<double> misc::Simpson3o8_Int(const double xmin, const double xmax, complex<double> fun(double x))
{
  // Default tolerance is tol = 1e-10.  Feel free to change this as needed.
  const double tol = 1e-8;

  int m = 300, N;
  N = 3 * m + 2;
  double dx = (xmax - xmin) / (N - 1);
  complex<double> sum = 0, sum2;
  sum2 = fun(xmin) + fun(xmax);
  for (int i = 0; i < m; i++)
  {
    sum2 += 3.0 * (fun(xmin + (3 * i + 1) * dx) + fun(xmin + (3 * i + 2) * dx)) + 2.0 * fun(xmin + (3 * i + 3) * dx);
    //     cout<<sum2<<endl;
    //     cout<<fun(xmin+(3*i+1)*dx)<<endl;
    //     cout<<fun(xmin+(3*i+2)*dx)<<endl;
    //     cout<<fun(xmin+(3*i+3)*dx)<<endl;
    //     if(abs(sum2) != abs(sum2)) exit(0);
  }
  sum2 = sum2 * dx * 3.0 / 8.0;

  int j = 1;
  const int jmax = 10;
  while (j < jmax && abs(sum2 - sum) > tol)
  {
    j++;
    m = m * 2;
    N = 3 * m + 2;
    dx = (xmax - xmin) / (N - 1);
    sum = sum2;
    sum2 = fun(xmin) + fun(xmax);
    for (int i = 0; i < m; i++)
    {
      sum2 += 3.0 * (fun(xmin + (3 * i + 1) * dx) + fun(xmin + (3 * i + 2) * dx)) + 2.0 * fun(xmin + (3 * i + 3) * dx);
    }
    sum2 = sum2 * dx * 3.0 / 8.0;

    //    cout<<"j = "<<j<<" error = "<<abs(sum2-sum)<<endl;
  }

  if (j == jmax)
    cout << "misc::Simpson3o8_Int has j > " << jmax << ", error = " << abs(sum2 - sum) << endl;

  return sum2;
}
complex<double> misc::Gauss_Int(const double xmin, const double xmax, complex<double> fun(double x))
{
  // Default tolerance is tol = 1e-10.  Feel free to change this as needed.
  const double tol = 1e-8;

  //  int N=int(xmax-xmin)*10;
  //  if(N<1000) N = 1000;

  int N = 1000;
  complex<double> sum = 0, sum2 = 0;
  sum2 = gaulegf(xmin, xmax, N, fun);

  int j = 1;
  const int jmax = 10;
  while (j < jmax && abs(sum2 - sum) > tol)
  {
    j++;
    N = N * 2;
    sum = sum2;
    sum2 = gaulegf(xmin, xmax, N, fun);

    cout << "j = " << j << " error = " << abs(sum2 - sum) << endl;
  }

  // if(j == jmax)
  cout << "misc::Gauss_Int has j > " << jmax << ", error = " << abs(sum2 - sum) << endl;

  return sum2;
}
complex<double> misc::gaulegf(double x1, double x2, int n, complex<double> fun(double x))
{
  int i, j, m;
  double eps = 1.2E-16;
  double p1, p2, p3, pp, xl, xm, z, z1;
  double w;

  m = (n + 1) / 2;
  xm = 0.5 * (x2 + x1);
  xl = 0.5 * (x2 - x1);
  complex<double> sum = 0;
  for (i = 0; i < m; i++)
  {
    z = cos(PI * ((double)i + 0.75) / ((double)n + 0.5));
    do
    {
      p1 = 1.0;
      p2 = 0.0;
      for (j = 0; j < n; j++)
      {
        p3 = p2;
        p2 = p1;
        p1 = ((2 * (double)j + 1) * z * p2 - (double)j * p3) / ((double)j + 1);
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z = z1 - p1 / pp;
      //	cout<<"here"<<endl;
    } while (fabs(z - z1) > eps);

    //      cout<<"there"<<endl;
    w = 2.0 * xl / ((1.0 - z * z) * pp * pp);
    sum += w * (fun(xm - xl * z) + fun(xm + xl * z));

    if (!isfinite(abs(sum)))
    {
      cout << xm - xl * z << "," << xm + xl * z << endl;
      cout << fun(xm - xl * z) << "," << fun(xm + xl * z) << endl;
    }
    //      cout<<i<<","<<m<<endl;
  }

  return sum;
}
/*
   This computes an in-place (in data and out data are all stored in x and y) complex-to-complex FFT
   x and y are the real and imaginary arrays of 2^m points.
   dir =  1 gives forward transform
X(n) = 1/N sum_{k=0} ^{N-1} x(k) exp(-jk2 pi n/N) for n=0...N-1
   dir = -1 gives reverse transform
x(n) =     sum_{k=0} ^{N-1} X(k) exp( jk2 pi n/N) for n=0...N-1
*/
void misc::FFT(short int dir, long m, double *x, double *y)
{
  long n, i, i1, j, k, i2, l, l1, l2;
  double c1, c2, tx, ty, t1, t2, u1, u2, z;

  /* Calculate the number of points */
  n = 1;
  for (i = 0; i < m; i++)
    n *= 2;

  /* Do the bit reversal */
  i2 = n >> 1;
  j = 0;
  for (i = 0; i < n - 1; i++)
  {
    if (i < j)
    {
      tx = x[i];
      ty = y[i];
      x[i] = x[j];
      y[i] = y[j];
      x[j] = tx;
      y[j] = ty;
    }
    k = i2;
    while (k <= j)
    {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  /* Compute the FFT */
  c1 = -1.0;
  c2 = 0.0;
  l2 = 1;
  for (l = 0; l < m; l++)
  {
    l1 = l2;
    l2 <<= 1;
    u1 = 1.0;
    u2 = 0.0;
    for (j = 0; j < l1; j++)
    {
      for (i = j; i < n; i += l2)
      {
        i1 = i + l1;
        t1 = u1 * x[i1] - u2 * y[i1];
        t2 = u1 * y[i1] + u2 * x[i1];
        x[i1] = x[i] - t1;
        y[i1] = y[i] - t2;
        x[i] += t1;
        y[i] += t2;
      }
      z = u1 * c1 - u2 * c2;
      u2 = u1 * c2 + u2 * c1;
      u1 = z;
    }
    c2 = sqrt((1.0 - c1) / 2.0);
    if (dir == 1)
      c2 = -c2;
    c1 = sqrt((1.0 + c1) / 2.0);
  }

  /* Scaling for forward transform */
  if (dir == 1)
  {
    for (i = 0; i < n; i++)
    {
      x[i] /= n;
      y[i] /= n;
    }
  }
}
// assume a[0] a[1]......a[NN/2-1]       a[NN/2]           ...... a[NN-1]
//         0    df      (NN/2-1)*df  combine of \pm NN/2*df        -df
//         0 1 2 3 4 5
//         ^ ^ ^ o ^ ^
//         0 1 2 3
//         ^ ^ o ^
void misc::Low_Pass_Filt(const int NN, double *a)
{
  // we use 2/3 law, NN/2 * 2/3 = NN/3
  for (int i = 0; i < NN / 3; i++)
  {
    a[NN / 2 + i] = 0;
    a[NN / 2 - i] = 0;
  }
}
void misc::polyinterp(double t, double &rr, double *ti, double *ri, const int ORD)
{
  //  (x  -x_1)...(x  -x_i-1)(x  -x_i+1)...(x  -x_N)
  // ------------------------------------------------f_i
  //  (x_i-x_1)...(x_i-x_i-1)(x_i-x_i+1)...(x_i-x_N)

  rr = 0;
  for (int i = 0; i < ORD; i++)
  {
    double ss = 1, xx = 1;
    for (int j = 0; j < ORD; j++)
    {
      if (j != i)
      {
        ss *= t - ti[j];
        xx *= ti[i] - ti[j];
      }
    }
    rr += ss / xx * ri[i];
  }
}
void misc::polyinterp_d1(double t, double &rr, double *ti, double *ri, const int ORD)
{
  //  sum_{j != i}[(x  -x_1)...(x  -x_j-1)(x  -x_j+1)...(x  -x_i-1)(x  -x_i+1)...(x  -x_N)]
  // ----------------------------------------------------------------------------------------f_i
  //                       (x_i-x_1)...(x_i-x_i-1)(x_i-x_i+1)...(x_i-x_N)

  rr = 0;
  for (int i = 0; i < ORD; i++)
  {
    double ss = 0, xx = 1;
    for (int j = 0; j < ORD; j++)
    {
      if (j != i)
      {
        double tt = 1;
        for (int k = 0; k < ORD; k++)
        {
          if (k != j)
            tt *= t - ti[k];
        }
        ss += tt;
        xx *= ti[i] - ti[j];
      }
    }
    rr += ss / xx * ri[i];
  }
}
void misc::next2power(long int Nin, long int &Nout, int &M)
{
  M = 0;
  Nout = 1;
  while (Nout < Nin)
  {
    M++;
    Nout *= 2;
  }
  // return Nout=2^M > Nin
}
int misc::MYpow2(int i)
{
  if (i == 0)
    return 1;
  else if (i > 0)
    return 2 * MYpow2(i - 1);
  else
    return MYpow2(i + 1) / 2;
}
