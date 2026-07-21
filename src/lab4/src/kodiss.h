
#ifndef KODISS_H
#define KODISS_H

#ifdef USE_GPU
#include <cuda_runtime.h>

__device__ double d_kodis_point(
    const int ex[3], const double* f,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, double eps,
    int i, int j, int k
);
#endif

#endif /* KODISS_H */
