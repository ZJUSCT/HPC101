
#ifndef MISC_H
#define MISC_H

#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
using namespace std;

#include <mpi.h>
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

namespace misc
{
    inline string &lTrim(string &ss)
    {
        string::iterator p = find_if(ss.begin(), ss.end(), not1(ptr_fun<int, int>(isspace)));
        ss.erase(ss.begin(), p);
        return ss;
    }
    inline string &rTrim(string &ss)
    {
        string::reverse_iterator p = find_if(ss.rbegin(), ss.rend(), not1(ptr_fun<int, int>(isspace)));
        ss.erase(p.base(), ss.end());
        return ss;
    }
    inline string &Trim(string &st)
    {
        lTrim(rTrim(st));
        return st;
    }

    template <typename T>
    void swap(T &a, T &b)
    {
        T c = a;
        a = b;
        b = c;
    }
    void tillherecheck(int myrank);
    void tillherecheck(const char str[]);
    void tillherecheck(MPI_Comm Comm_here, int out_rank, const char str[]);
    void tillherecheck(MPI_Comm Comm_here, int out_rank, const string str);
    int parse_parts(string str, string &sgrp, string &skey, string &sval, int &ind);
    int parse_parts(string str, string &sgrp, string &skey, string &sval, int &ind1, int &ind2);
    int parse_parts(string str, string &sgrp, string &skey, string &sval, int &ind1, int &ind2, int &ind3);
    void gaulegf(double x1, double x2, double *x, double *w, int n);
    complex<double> gaulegf(double x1, double x2, int n, complex<double> fun(double x));
    void inversearray(double *aa, int NN);
    double fact(int N);
    double Wigner_d_function(int l, int m, int s, double costheta);
    int num_of_str(char *c);
    void TVDrungekutta3(const int N, const double dT, double *f0, double *f1, double *f_rhs, const int RK4);
    void rungekutta4(const int N, const double dT, double *f0, double *f1, double *f_rhs, const int RK4);
    void rungekutta4(const double dT, const std::vector<double> &f0,
                     std::vector<double> &f1, std::vector<double> &f_rhs, const int RK4);
    void dividBlock(const int DIM, int *shape_here, double *bbox_here, const int pices, double *picef, int *shape_res, double *bbox_res, const int min_width);
    void swapvector(std::vector<double> &f0, std::vector<double> &f1);
    complex<double> complex_gamma(complex<double> z);
    complex<double> KummerComplex(const complex<double> a, const complex<double> b, complex<double> x);
#if 0 
complex<double> First_Bessel(const complex<double> a,complex<double> x);
#else
    complex<double> First_Bessel(double a, complex<double> x);
#endif
    complex<double> Rec_Int(const double xmin, const double xmax, complex<double> fun(double x));
    complex<double> Simpson_Int(const double xmin, const double xmax, complex<double> fun(double x));
    complex<double> Simpson3o8_Int(const double xmin, const double xmax, complex<double> fun(double x));
    complex<double> Gauss_Int(const double xmin, const double xmax, complex<double> fun(double x));

    void FFT(short int dir, long m, double *x, double *y);
    void Low_Pass_Filt(const int NN, double *a);
    void polyinterp(double t, double &rr, double *ti, double *ri, const int ORD);
    void polyinterp_d1(double t, double &rr, double *ti, double *ri, const int ORD);
    void next2power(long int Nin, long int &Nout, int &M);
    int MYpow2(int i);

#ifdef USE_GPU
    __device__ double wigner_d_device(int l, int m, int s, double costheta);
#endif
}
#endif /* MISC_H */
