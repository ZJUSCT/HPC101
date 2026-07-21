#include <cuda_runtime.h>
#include <math.h>

#include <iostream>
#include "gpu_manager.h"

#define MAX_ORDN 6
#ifndef GPU_DEBUG_PRINT
#define GPU_DEBUG_PRINT 0
#endif

__device__ __forceinline__ void gpu_stop() {
#if GPU_STRICT_STOP
    asm("trap;");
#endif
}

__device__ __forceinline__ double X_at_1b(const double* X, int i1b) {
	return X[i1b - 1];
}
__device__ __forceinline__ double f_at_1b(const double* f, const int ex[3], int i1b, int j1b, int k1b) {
	return f[((k1b - 1) * ex[1] + (j1b - 1)) * ex[0] + (i1b - 1)];
}

__device__ void polint(const double* xa, const double* ya, double x, double& y, double& dy, int ordn) {
	double c[MAX_ORDN], d[MAX_ORDN], den[MAX_ORDN], ho[MAX_ORDN];
	int ns = 1;
	double dif = fabs(x - xa[0]);
	for (int m = 0; m < ordn; ++m) {
		c[m] = ya[m];
		d[m] = ya[m];
		ho[m] = xa[m] - x;
		double dift = fabs(x - xa[m]);
		if (dift < dif) { ns = m + 1; dif = dift; }
	}
	y = ya[ns - 1];
	ns = ns - 1;
	for (int m = 1; m < ordn; ++m) {
		for (int i = 0; i < ordn - m; ++i) {
			den[i] = ho[i] - ho[i + m];
			if (den[i] == 0.0) {
#if GPU_DEBUG_PRINT
                printf("failure in polint for point %f\n", x);
                printf("with input points: ");
                for (int t = 0; t < ordn; ++t) printf("%f ", xa[t]);
                printf("\n");
#endif
				y = NAN; dy = NAN; gpu_stop(); return;
			}
			den[i] = (c[i + 1] - d[i]) / den[i];
			d[i] = ho[i + m] * den[i];
			c[i] = ho[i] * den[i];
		}
		if (2 * ns < (ordn - m)) {
			dy = c[ns];
		} else {
			dy = d[ns - 1];
			ns = ns - 1;
		}
		y = y + dy;
	}
}

__device__ void d_polin3_1b(
	const double* x1a, const double* x2a, const double* x3a,
	const double* ya, double x1, double x2, double x3,
	double& y, double& dy, int ordn
) {
	double yatmp[MAX_ORDN * MAX_ORDN];
	double ymtmp[MAX_ORDN];
	double yntmp[MAX_ORDN];
	double yqtmp[MAX_ORDN];

	for (int i = 0; i < ordn; ++i) {
		for (int j = 0; j < ordn; ++j) {
			for (int k = 0; k < ordn; ++k) {
				yqtmp[k] = ya[(k * ordn + j) * ordn + i];
			}
			polint(x3a, yqtmp, x3, yatmp[j * ordn + i], dy, ordn);
		}
		for (int j = 0; j < ordn; ++j) yntmp[j] = yatmp[j * ordn + i];
		polint(x2a, yntmp, x2, ymtmp[i], dy, ordn);
	}
	polint(x1a, ymtmp, x1, y, dy, ordn);
}

__device__ bool d_decide3d(
	const int ex[3], const double* f, const double* fpi,
	const int cxB[3], const int cxT[3], const double SoA[3],
	double* ya, int ordn, int Symmetry
) {
	(void)fpi;
	(void)Symmetry;
	bool gont = false;
	int fmin1[3], fmin2[3], fmax1[3], fmax2[3];

	for (int m = 0; m < 3; ++m) {
		if (!(abs(cxB[m]) >= 0)) gont = true;
		if (!(abs(cxT[m]) >= 0)) gont = true;
		fmin1[m] = max(1, cxB[m]);
		fmax1[m] = cxT[m];
		fmin2[m] = cxB[m];
		fmax2[m] = min(0, cxT[m]);
		if ((fmin1[m] <= fmax1[m]) && (fmin1[m] < 1 || fmax1[m] > ex[m])) gont = true;
		if ((fmin2[m] <= fmax2[m]) && (1 - fmax2[m] < 1 || 1 - fmin2[m] > ex[m])) gont = true;
	}
	if (gont) {
#if GPU_DEBUG_PRINT
        printf("error in decide3d\n");
        printf("cxB: %d %d %d, cxT: %d %d %d, ex: %d %d %d\n",
               cxB[0], cxB[1], cxB[2], cxT[0], cxT[1], cxT[2], ex[0], ex[1], ex[2]);
        printf("fmin1: %d %d %d, fmax1: %d %d %d\n",
               fmin1[0], fmin1[1], fmin1[2], fmax1[0], fmax1[1], fmax1[2]);
        printf("fmin2: %d %d %d, fmax2: %d %d %d\n",
               fmin2[0], fmin2[1], fmin2[2], fmax2[0], fmax2[1], fmax2[2]);
#endif
		return true;
	}

	auto idx = [&](int i, int j, int k) {
		return ((k - cxB[2]) * ordn + (j - cxB[1])) * ordn + (i - cxB[0]);
	};

	for (int k = fmin1[2]; k <= fmax1[2]; ++k) {
		for (int j = fmin1[1]; j <= fmax1[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, j, k);
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, j, k) * SoA[0];
		}
		for (int j = fmin2[1]; j <= fmax2[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, 1 - j, k) * SoA[1];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, 1 - j, k) * SoA[0] * SoA[1];
		}
	}

	for (int k = fmin2[2]; k <= fmax2[2]; ++k) {
		for (int j = fmin1[1]; j <= fmax1[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, j, 1 - k) * SoA[2];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, j, 1 - k) * SoA[0] * SoA[2];
		}
		for (int j = fmin2[1]; j <= fmax2[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, 1 - j, 1 - k) * SoA[1] * SoA[2];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, 1 - j, 1 - k) * SoA[0] * SoA[1] * SoA[2];
		}
	}
	return false;
}

__device__ void global_interp_device(
	const int* ex, const double* X, const double* Y, const double* Z,
	const double* f, double* f_int,
	double x1, double y1, double z1,
	int ORDN, const double* SoA, int symmetry
) {
	if (ORDN > MAX_ORDN) { f_int[0] = NAN; gpu_stop(); return; }

	const int NO_SYMM = 0, EQUATORIAL = 1, OCTANT = 2;
	int imin = 1, jmin = 1, kmin = 1;

	double dX = X_at_1b(X, imin + 1) - X_at_1b(X, imin);
	double dY = X_at_1b(Y, jmin + 1) - X_at_1b(Y, jmin);
	double dZ = X_at_1b(Z, kmin + 1) - X_at_1b(Z, kmin);

	double x1a[MAX_ORDN];
	for (int j = 0; j < ORDN; ++j) x1a[j] = (double)j;

	int cxI[3];
	cxI[0] = (int)((x1 - X_at_1b(X, 1)) / dX + 0.4) + 1;
	cxI[1] = (int)((y1 - X_at_1b(Y, 1)) / dY + 0.4) + 1;
	cxI[2] = (int)((z1 - X_at_1b(Z, 1)) / dZ + 0.4) + 1;

	int cxB[3], cxT[3], cmin[3], cmax[3];
	for (int m = 0; m < 3; ++m) {
		cxB[m] = cxI[m] - ORDN / 2 + 1;
		cxT[m] = cxB[m] + ORDN - 1;
		cmin[m] = 1;
		cmax[m] = ex[m];
	}
	if (symmetry == OCTANT && fabs(X_at_1b(X, 1)) < dX) cmin[0] = -ORDN / 2 + 1;
	if (symmetry == OCTANT && fabs(X_at_1b(Y, 1)) < dY) cmin[1] = -ORDN / 2 + 1;
	if (symmetry != NO_SYMM && fabs(X_at_1b(Z, 1)) < dZ) cmin[2] = -ORDN / 2 + 1;

	for (int m = 0; m < 3; ++m) {
		if (cxB[m] < cmin[m]) { cxB[m] = cmin[m]; cxT[m] = cxB[m] + ORDN - 1; }
		if (cxT[m] > cmax[m]) { cxT[m] = cmax[m]; cxB[m] = cxT[m] + 1 - ORDN; }
	}

	double cx[3];
	cx[0] = (cxB[0] > 0) ? (x1 - X_at_1b(X, cxB[0])) / dX : (x1 + X_at_1b(X, 1 - cxB[0])) / dX;
	cx[1] = (cxB[1] > 0) ? (y1 - X_at_1b(Y, cxB[1])) / dY : (y1 + X_at_1b(Y, 1 - cxB[1])) / dY;
	cx[2] = (cxB[2] > 0) ? (z1 - X_at_1b(Z, cxB[2])) / dZ : (z1 + X_at_1b(Z, 1 - cxB[2])) / dZ;

	double ya[MAX_ORDN * MAX_ORDN * MAX_ORDN];
	if (d_decide3d(ex, f, f, cxB, cxT, SoA, ya, ORDN, symmetry)) {
#if GPU_DEBUG_PRINT
        printf("global_interp position: %f %f %f\n", x1, y1, z1);
        printf("data range: %f %f %f %f %f %f\n",
               X_at_1b(X, 1), X_at_1b(X, ex[0]),
               X_at_1b(Y, 1), X_at_1b(Y, ex[1]),
               X_at_1b(Z, 1), X_at_1b(Z, ex[2]));
#endif
		f_int[0] = NAN;
		gpu_stop();
		return;
	}

	double ddy = 0.0;
	d_polin3_1b(x1a, x1a, x1a, ya, cx[0], cx[1], cx[2], f_int[0], ddy, ORDN);
}

__device__ double d_symmetry_bd_1b(
	int ord, const int extc[3], const double* func,
	int i1b, int j1b, int k1b, const double SoA[3]
) {
	// out-of-range stays zero, matching funcc = 0.d0 initialization
	if (i1b < -ord + 1 || i1b > extc[0]) return 0.0;
	if (j1b < -ord + 1 || j1b > extc[1]) return 0.0;
	if (k1b < -ord + 1 || k1b > extc[2]) return 0.0;

	int ii = i1b, jj = j1b, kk = k1b;
	double factor = 1.0;

	// apply symmetry in x, then y, then z (same order as Fortran)
	if (ii <= 0) { ii = 1 - ii; factor *= SoA[0]; }
	if (jj <= 0) { jj = 1 - jj; factor *= SoA[1]; }
	if (kk <= 0) { kk = 1 - kk; factor *= SoA[2]; }

	if (ii < 1 || ii > extc[0]) return 0.0;
	if (jj < 1 || jj > extc[1]) return 0.0;
	if (kk < 1 || kk > extc[2]) return 0.0;

	return f_at_1b(func, extc, ii, jj, kk) * factor;
}

__global__ void lowerboundset_kernel(int n, double* chi0, double TINNY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (chi0[idx] < TINNY) {
            chi0[idx] = TINNY;
        }
    }
}

void gpu_lowerboundset_launch(
    cudaStream_t &stream,
    int ex[3],
    double* d_chi0, double TINNY
) {
    int n = ex[0] * ex[1] * ex[2];
    int block = 256;
    int grid = (n + block - 1) / block;

    lowerboundset_kernel<<<grid, block, 0, stream>>>(n, d_chi0, TINNY);
}

__global__ void gpu_pack_kernel(
    const double* __restrict__ src_3d, double* __restrict__ dst_1d,
    int src_nx, int src_ny, 
    int dst_nx, int dst_ny, int dst_nz,
    int off_x, int off_y, int off_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dst_nx * dst_ny * dst_nz;
    if (idx >= total) return;
    
    int i = idx % dst_nx;
    int j = (idx / dst_nx) % dst_ny;
    int k = idx / (dst_nx * dst_ny);
    
    int src_idx = (k + off_z) * (src_nx * src_ny) + (j + off_y) * src_nx + (i + off_x);
    dst_1d[idx] = src_3d[src_idx];
}

__global__ void gpu_unpack_kernel(
    const double* __restrict__ src_1d, double* __restrict__ dst_3d,
    int dst_nx, int dst_ny, 
    int src_nx, int src_ny, int src_nz,
    int off_x, int off_y, int off_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = src_nx * src_ny * src_nz;
    if (idx >= total) return;
    
    int i = idx % src_nx;
    int j = (idx / src_nx) % src_ny;
    int k = idx / (src_nx * src_ny);
    
    int dst_idx = (k + off_z) * (dst_nx * dst_ny) + (j + off_y) * dst_nx + (i + off_x);
    dst_3d[dst_idx] = src_1d[idx];
}

void gpu_pack_launch(
	cudaStream_t stream, const double* d_src_3d, double* d_dst_1d,
	int src_nx, int src_ny, int dst_nx, int dst_ny, int dst_nz,
	int off_x, int off_y, int off_z
) {
	int n = dst_nx * dst_ny * dst_nz;
	int block = 256;
	int grid = (n + block - 1) / block;
	gpu_pack_kernel<<<grid, block, 0, stream>>>(d_src_3d, d_dst_1d, src_nx, src_ny, dst_nx, dst_ny, dst_nz, off_x, off_y, off_z);
}

void gpu_unpack_launch(
	cudaStream_t stream, const double* d_src_1d, double* d_dst_3d,
	int dst_nx, int dst_ny, int src_nx, int src_ny, int src_nz,
	int off_x, int off_y, int off_z
) {
	int n = src_nx * src_ny * src_nz;
	int block = 256;
	int grid = (n + block - 1) / block;
	gpu_unpack_kernel<<<grid, block, 0, stream>>>(d_src_1d, d_dst_3d, dst_nx, dst_ny, src_nx, src_ny, src_nz, off_x, off_y, off_z);
}

// =====================================================================
// Time Level Interpolation Kernels
// =====================================================================

__global__ void average_kernel(int n, const double* f1, const double* f2, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = 0.5 * (f1[idx] + f2[idx]);
    }
}

__global__ void average3_kernel(int n, const double* f1, const double* f2, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = 0.75 * f1[idx] + 0.25 * f2[idx];
    }
}

__global__ void average2_kernel(int n, const double* f1, const double* f2, const double* f3, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = (3.0 / 8.0) * f1[idx] + (3.0 / 4.0) * f2[idx] - (1.0 / 8.0) * f3[idx];
    }
}

__global__ void average2p_kernel(int n, const double* f1, const double* f2, const double* f3, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = (21.0 / 32.0) * f1[idx] + (7.0 / 16.0) * f2[idx] - (3.0 / 32.0) * f3[idx];
    }
}

__global__ void average2m_kernel(int n, const double* f1, const double* f2, const double* f3, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = (5.0 / 32.0) * f1[idx] + (15.0 / 16.0) * f2[idx] - (3.0 / 32.0) * f3[idx];
    }
}

void gpu_average_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_fout);
}

void gpu_average3_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average3_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_fout);
}

void gpu_average2_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average2_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_f3, d_fout);
}

void gpu_average2p_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average2p_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_f3, d_fout);
}

void gpu_average2m_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average2m_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_f3, d_fout);
}

__global__ void global_interp_kernel(
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int ex0, int ex1, int ex2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, double SoA_0, double SoA_1, double SoA_2, int Symmetry,
    int var_idx, int num_var,
    double* d_shellf, int* d_weight
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NN) return;

    // 获取当前点的坐标
    double px = d_XX_0[j];
    double py = d_XX_1[j];
    double pz = d_XX_2[j];

    // 边界检查（Bounding Box 判断）— 使用 DH/2 容差以匹配 CPU 浮点比较行为
    double tol_0 = DH_0 / 2.0;
    double tol_1 = DH_1 / 2.0;
    double tol_2 = DH_2 / 2.0;

    if (px - llb_0 < -tol_0 || px - uub_0 > tol_0) return;
    if (DIM > 1 && (py - llb_1 < -tol_1 || py - uub_1 > tol_1)) return;
    if (DIM > 2 && (pz - llb_2 < -tol_2 || pz - uub_2 > tol_2)) return;

    // 组装传给已有插值库的指针
    double* d_X_arr[3] = {d_X_0, d_X_1, d_X_2};
	double SoA_arr[3] = {SoA_0, SoA_1, SoA_2};
	const int ex[3] = {ex0, ex1, ex2};

    // 调用你们原有的设备端插值函数
    double val = 0.0;
    global_interp_device(
        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
        d_field, &val,
        px, py, pz,
        ordn, SoA_arr, Symmetry
    );

    // 将结果原子累加到对应位置（处理 Ghost Zone 多个 Block 重叠的情况）
    atomicAdd(&d_shellf[j * num_var + var_idx], val);

    if (var_idx == 0) {
        atomicAdd(&d_weight[j], 1);
    }
}

void gpu_global_interp_launch(
	cudaStream_t stream,
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, double SoA_0, double SoA_1, double SoA_2,
    int Symmetry, int var_idx, int num_var,
    double* d_shellf, int* d_weight
) {
	int blockSize = 256;
    int gridSize = (NN + blockSize - 1) / blockSize;

	global_interp_kernel<<<gridSize, blockSize, 0, stream>>>(
		NN, DIM,
        d_XX_0, d_XX_1, d_XX_2,
        shape_0, shape_1, shape_2, d_X_0, d_X_1, d_X_2, d_field,
        llb_0, llb_1, llb_2, uub_0, uub_1, uub_2,
        DH_0, DH_1, DH_2,
        ordn, SoA_0, SoA_1, SoA_2, Symmetry,
        var_idx, num_var, d_shellf, d_weight
	);
}

__forceinline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l2normhelper_kernel(
    const double* __restrict__ f,
    int imin, int imax,
    int jmin, int jmax,
    int kmin, int kmax,
    int nx, int ny, int nz,
    double* __restrict__ d_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + imin;
    int j = blockIdx.y * blockDim.y + threadIdx.y + jmin;
    int k = blockIdx.z * blockDim.z + threadIdx.z + kmin;

    double my_val = 0.0;

    if (i <= imax && j <= jmax && k <= kmax) {
        long long idx = (long long)i + (long long)j * nx + (long long)k * nx * ny;
        double val = f[idx];
        my_val = val * val;
    }

    // Warp 内规约求和 (假设你的 warpReduceSum 内部使用了正确的 __shfl_down_sync)
    my_val = warpReduceSum(my_val);

    // 修正: 计算当前线程在 Block 内的 1D 线性 ID
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // 只有每个 Warp 的第 0 个线程 (线性 ID 是 warpSize 的整数倍) 负责写入
    if ((tid % warpSize) == 0) {
        atomicAdd(d_out, my_val);
    }
}

void gpu_l2normhelper_launch(
	cudaStream_t stream, 
	const int* ex, 
	const double* X, const double* Y, const double* Z,
	double xmin, double ymin, double zmin,
	double xmax, double ymax, double zmax,
	const double* d_f, double& f_out, int gw
) {
    double dX = X[1] - X[0];
    double dY = Y[1] - Y[0];
    double dZ = Z[1] - Z[0];

    // 将 Fortran 的 1-indexed 逻辑转换为 C/C++ 的 0-indexed 逻辑
    int imin = gw;
    int jmin = gw;
    int kmin = gw;

    int imax = ex[0] - gw - 1;
    int jmax = ex[1] - gw - 1;
    int kmax = ex[2] - gw - 1;

    // 边界判断 (与 Fortran 逻辑完全一致)
    if (fabs(X[ex[0] - 1] - xmax) < dX) imax = ex[0] - 1;
    if (fabs(Y[ex[1] - 1] - ymax) < dY) jmax = ex[1] - 1;
    if (fabs(Z[ex[2] - 1] - zmax) < dZ) kmax = ex[2] - 1;
    
    if (fabs(X[0] - xmin) < dX) imin = 0;
    if (fabs(Y[0] - ymin) < dY) jmin = 0;
    if (fabs(Z[0] - zmin) < dZ) kmin = 0;

    int nx_proc = imax - imin + 1;
    int ny_proc = jmax - jmin + 1;
    int nz_proc = kmax - kmin + 1;

    if (nx_proc <= 0 || ny_proc <= 0 || nz_proc <= 0) {
        f_out = 0.0;
        return;
    }

    // 分配设备端内存用于保存累加结果 (使用 Async API 降低分配延迟，需 CUDA 11.2+)
    double* d_sum = GPUManager::getInstance().allocate_device_memory(1);
    cudaMemsetAsync(d_sum, 0, sizeof(double), stream);

    // 设置线程块大小，通常 8x8x8 = 512 线程效率较好
    dim3 blockDim(8, 8, 8); 
    dim3 gridDim((nx_proc + blockDim.x - 1) / blockDim.x,
                 (ny_proc + blockDim.y - 1) / blockDim.y,
                 (nz_proc + blockDim.z - 1) / blockDim.z);

    // 启动 Kernel
    l2normhelper_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_f, 
        imin, imax, jmin, jmax, kmin, kmax, 
        ex[0], ex[1], ex[2], 
        d_sum
    );

    double h_sum = 0.0;
    // 将结果拷回 CPU
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
    
    // 强制同步：由于后续的 MPI_Allreduce 立刻需要用到 f_out 的 CPU 数据，这里必须等待 GPU 计算并拷贝完成
    cudaStreamSynchronize(stream);
    GPUManager::getInstance().free_device_memory(d_sum, 1);

    f_out = h_sum * dX * dY * dZ;
}

__global__ void global_interp_amr_kernel(
    int active_count, int DIM,
    int* d_active_indices,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int ex0, int ex1, int ex2, 
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    int ordn, double SoA_0, double SoA_1, double SoA_2, int Symmetry,
    int var_idx, int num_var,
    double* d_shellf
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    // 获取该点在全局 n=1000 数组中的真实索引
    int j = d_active_indices[idx]; 

    double px = d_XX_0[j];
    double py = d_XX_1[j];
    double pz = d_XX_2[j];

    double* d_X_arr[3] = {d_X_0, d_X_1, d_X_2};
    double SoA_arr[3] = {SoA_0, SoA_1, SoA_2};
    const int ex[3] = {ex0, ex1, ex2};

    double val = 0.0;
    // 调用现有的设备端插值核心
    global_interp_device(
        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
        d_field, &val,
        px, py, pz,
        ordn, SoA_arr, Symmetry
    );

    // 直接赋值，不再需要 atomicAdd，因为每个点已被 CPU 保证全局唯一认领
    d_shellf[j * num_var + var_idx] = val;
}

void gpu_global_interp_amr_launch(
    cudaStream_t stream,
    int active_count, int DIM,
    int* d_active_indices,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    int ordn, double SoA_0, double SoA_1, double SoA_2, 
    int Symmetry, int var_idx, int num_var,
    double* d_shellf
) {
    if (active_count == 0) return;
    int blockSize = 256;
    int gridSize = (active_count + blockSize - 1) / blockSize;

    global_interp_amr_kernel<<<gridSize, blockSize, 0, stream>>>(
        active_count, DIM, d_active_indices,
        d_XX_0, d_XX_1, d_XX_2, 
        shape_0, shape_1, shape_2, d_X_0, d_X_1, d_X_2, d_field,
        ordn, SoA_0, SoA_1, SoA_2, Symmetry, 
        var_idx, num_var, d_shellf
    );
}