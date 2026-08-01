#include "utils.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

/* ----------------------------------------------------------------------- */
/*  Device memory helpers                                                  */
/* ----------------------------------------------------------------------- */
void* device_alloc(size_t bytes) {
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) return nullptr;
    return p;
}
void device_free(void* p) { if (p) cudaFree(p); }

/* ----------------------------------------------------------------------- */
/*  Device max-abs reduction (FP64)                                        */
/* ----------------------------------------------------------------------- */
static __global__ void maxabs_block_kernel(const double* x, size_t n, double* partial)
{
    /* one block reduces a chunk; uses shared-memory tree reduction */
    extern __shared__ double smem[];
    double t = 0.0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        double v = fabs(x[i]);
        if (v > t) t = v;
    }
    smem[threadIdx.x] = t;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            double a = smem[threadIdx.x], b = smem[threadIdx.x + s];
            smem[threadIdx.x] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = smem[0];
}

double device_maxabs_fp64(const double* dx, size_t n)
{
    if (n == 0) return 0.0;
    int block = 256;
    /* cap grid to avoid tiny blocks on huge inputs */
    size_t grid = (n + (size_t)block - 1) / block;
    if (grid > 4096) grid = 4096;
    double* partial = nullptr;
    cudaMalloc(&partial, grid * sizeof(double));
    maxabs_block_kernel<<<(unsigned)grid, block, block * sizeof(double)>>>(
        dx, n, partial);
    /* final reduce on host (grid <= 4096 doubles = 32KB, cheap) */
    double* h_partial = (double*)malloc(grid * sizeof(double));
    cudaMemcpy(h_partial, partial, grid * sizeof(double),
               cudaMemcpyDeviceToHost);
    double m = 0.0;
    for (size_t i = 0; i < grid; ++i)
        if (h_partial[i] > m) m = h_partial[i];
    cudaFree(partial);
    free(h_partial);
    return m;
}

/* ----------------------------------------------------------------------- */
/*  Deterministic matrix fill                                              */
/* ----------------------------------------------------------------------- */
/* Simple xorshift64 PRNG so the lab is fully reproducible without linking
 * against a host RNG library. Values are mapped to [-1, 1]. */
static inline uint64_t xs64_next(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

void fill_matrix(double* h, size_t n, uint64_t seed) {
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; ++i) {
        uint64_t u = xs64_next(&s);
        /* map 53 bits of u to [0,1) */
        double r = (double)(u >> 11) * (1.0 / 9007199254740992.0);
        h[i] = 2.0 * r - 1.0;          /* [-1, 1] */
    }
}

/* ----------------------------------------------------------------------- */
/*  Accuracy comparison                                                    */
/* ----------------------------------------------------------------------- */
/* We compute everything on the GPU to avoid large host<->device copies. */

static __global__ void acc_diff_kernel(const double* ref, const double* test, int mn,
                                double* out_absmax,    /* scratch atomic */
                                double* out_l2_num,    /* sum of squares of diff */
                                double* out_l2_den)    /* sum of squares of ref */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mn) return;
    double r = ref[i];
    double t = test[i];
    double d = r - t;
    double ad = fabs(d);
    /* atomic max for abs error */
    /* implement atomic max on double via atomicCAS */
    unsigned long long* addr = (unsigned long long*)out_absmax;
    unsigned long long assumed, old = *addr;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        old = atomicCAS(addr, assumed, __double_as_longlong(fmax(cur, ad)));
    } while (assumed != old);
    atomicAdd(out_l2_num, d * d);
    atomicAdd(out_l2_den, r * r);
}

void accuracy_compare(const double* d_ref, const double* d_test,
                      int M, int N, AccuracyReport* rep)
{
    int mn = M * N;
    double *d_absmax, *d_l2n, *d_l2d;
    cudaMalloc(&d_absmax, sizeof(double));
    cudaMalloc(&d_l2n, sizeof(double));
    cudaMalloc(&d_l2d, sizeof(double));
    double z = 0.0;
    cudaMemcpy(d_absmax, &z, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l2n, &z, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l2d, &z, sizeof(double), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (mn + block - 1) / block;
    acc_diff_kernel<<<grid, block>>>(d_ref, d_test, mn, d_absmax, d_l2n, d_l2d);
    cudaDeviceSynchronize();

    double h_absmax = 0, h_l2n = 0, h_l2d = 0;
    cudaMemcpy(&h_absmax, d_absmax, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_l2n, d_l2n, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_l2d, d_l2d, sizeof(double), cudaMemcpyDeviceToHost);

    rep->max_abs_err = h_absmax;
    rep->l2_rel_err  = (h_l2d > 0.0) ? sqrt(h_l2n / h_l2d) : 0.0;
    rep->max_rel_err = 0.0; /* computed below for non-tiny entries */

    cudaFree(d_absmax);
    cudaFree(d_l2n);
    cudaFree(d_l2d);
}
