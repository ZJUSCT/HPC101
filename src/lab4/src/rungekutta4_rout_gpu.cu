#include "rungekutta4_rout.h"

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_manager.h"

__global__ void rungekutta4_rout_kernel(
    int n,
    const double* f0, double* f1, double* f_rhs, double dT, int RK4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double f0_val = f0[idx];
    double f1_val = f1[idx];
    double f_rhs_val = f_rhs[idx];

    if (RK4 == 0) {
        f1[idx] = f0_val + 0.5 * dT * f_rhs_val;
    } else if (RK4 == 1) {
        f_rhs[idx] = f_rhs_val + 2.0 * f1_val;
        f1[idx]    = f0_val + 0.5 * dT * f1_val;
    } else if (RK4 == 2) {
        f_rhs[idx] = f_rhs_val + 2.0 * f1_val;
        f1[idx]    = f0_val + dT * f1_val;
    } else if (RK4 == 3) {
        f1[idx] = f0_val + (1.0 / 6.0) * dT * (f1_val + f_rhs_val);
    }
}

void gpu_rungekutta4_rout_launch(
    cudaStream_t &stream,
    int ex[3], double dT,
    double* d_f0, double* d_f1, double* d_f_rhs, int RK4
) {
    int n = ex[0] * ex[1] * ex[2];

    int block = 256;
    int grid = (n + block - 1) / block;

    rungekutta4_rout_kernel<<<grid, block, 0, stream>>>(
        n,
        d_f0, d_f1, d_f_rhs, dT, RK4
    );
}
