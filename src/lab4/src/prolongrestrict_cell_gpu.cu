#include "prolongrestrict.h"

#include "fmisc.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// ==========================================
// 1. Constants & Helper Functions
// ==========================================

// Prolongation Coefficients (5th order)
__constant__ double C_PROLONG[6] = {
    77.0 / 8192.0,    // C1
    -693.0 / 8192.0,  // C2
    3465.0 / 4096.0,  // C3
    1155.0 / 4096.0,  // C4
    -495.0 / 8192.0,  // C5
    63.0 / 8192.0     // C6
};

// Restriction Coefficients
__constant__ double C_RESTRICT[3] = {
    3.0 / 256.0,      // C1
    -25.0 / 256.0,    // C2
    75.0 / 128.0      // C3
};

// Fortran IDINT equivalent
__device__ int d_idint(double a) {
    if (fabs(a) < 1.0) return 0;
    return (int)(a);
}

// Memory Access Helper: Column-Major (Fortran Layout)
// i, j, k are 0-based indices
__device__ __forceinline__ int get_col_major_idx(int i, int j, int k, int nx, int ny, int nz) {
    return k * (nx * ny) + j * nx + i;
}

// ==========================================
// 2. Prolongation Device Function
// ==========================================

// Calculate prolonged value for a single point (i, j, k) on the FINE grid.
// i, j, k are 0-based indices [0, extf-1]
__device__ void d_prolong3_device(
    int i, int j, int k, 
    const double* llbc, const double* uubc, const int* extc, const double* func,
    const double* llbf, const double* uubf, const int* extf, double* funf, 
    const double* llbp, const double* uubp,
    const double* SoA, int Symmetry
) {
    // --- 1. Geometry & Alignment (Calculated exactly like Fortran to ensure bitwise match) ---
    double CD[3], FD[3];
    double base[3];
    int lbc[3], lbf[3], lbp[3], ubp[3], lbpc[3], ubpc[3];
    
    // Calculate cell dimensions
    for (int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
    }

    // Alignment Logic
    for (int d = 0; d < 3; d++) {
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = d_idint((llbc[d] - llbf[d]) / FD[d] + 0.4f);
            if ((j_val / 2) * 2 == j_val) {
                base[d] = llbf[d];
            } else {
                base[d] = llbf[d] - CD[d] / 2.0;
            }
        }
    }

    // Calculate integer bounds (Resulting values are "1-based" relative to base)
    for (int d = 0; d < 3; d++) {
        lbf[d] = d_idint((llbf[d] - base[d]) / FD[d] + 0.4f) + 1;
        lbc[d] = d_idint((llbc[d] - base[d]) / CD[d] + 0.4f) + 1;
        
        lbp[d] = d_idint((llbp[d] - base[d]) / FD[d] + 0.4f) + 1; 
        ubp[d] = d_idint((uubp[d] - base[d]) / FD[d] + 0.4f);
        
        lbpc[d] = d_idint((llbp[d] - base[d]) / CD[d] + 0.4f) + 1;
        ubpc[d] = d_idint((uubp[d] - base[d]) / CD[d] + 0.4f); // Not strictly used for bounds check here
    }

    // Calculate valid range (1-based relative to loop start)
    // In Fortran: do i = imino, imaxo
    // imino = lbp - lbf + 1
    // In CUDA 0-based: valid i is [imino-1, imaxo-1]
    int imino = lbp[0] - lbf[0] + 1;
    int imaxo = ubp[0] - lbf[0] + 1;
    int jmino = lbp[1] - lbf[1] + 1;
    int jmaxo = ubp[1] - lbf[1] + 1;
    int kmino = lbp[2] - lbf[2] + 1;
    int kmaxo = ubp[2] - lbf[2] + 1;

    // Convert to 1-based for comparison
    int i_1b = i + 1;
    int j_1b = j + 1;
    int k_1b = k + 1;

    if (i_1b < imino || i_1b > imaxo || j_1b < jmino || j_1b > jmaxo || k_1b < kmino || k_1b > kmaxo) {
        return; 
    }

    // --- 2. Index Mapping ---
    
    // Global index (Still effectively 1-based logic for parity check)
    // Fortran: ii = i + lbf - 1. Since our i is 0-based, ii = (i+1) + lbf - 1 = i + lbf
    int ii = i + lbf[0];
    int jj = j + lbf[1];
    int kk = k + lbf[2];

    // Coarse Index Calculation
    // Fortran: cxI = i; cxI = (cxI + lbf - 1)/2; cxI = cxI - lbc + 1
    // CUDA: use i_1b for 'i'
    int cxI_i = (i_1b + lbf[0] - 1) / 2 - lbc[0] + 1;
    int cxI_j = (j_1b + lbf[1] - 1) / 2 - lbc[1] + 1;
    int cxI_k = (k_1b + lbf[2] - 1) / 2 - lbc[2] + 1;

    // Parity Checks (Even/Odd)
    bool k_even = ((kk / 2) * 2 == kk);
    bool j_even = ((jj / 2) * 2 == jj);
    bool i_even = ((ii / 2) * 2 == ii);

    // --- 3. Interpolation ---
    double tmp2[6][6];
    double tmp1[6];

    // Z-Direction Interpolation
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_ic = cxI_i - 2 + n;
            int cur_jc = cxI_j - 2 + m;
            
            double val = 0.0;
            // 1-based indices passed to d_get_sym_val
            if (k_even) {
                val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
                val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
                val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
                val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
                val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
                val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
            } else {
                val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
                val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
                val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
                val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
                val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
                val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
            }
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Interpolation
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        if (j_even) {
            val += C_PROLONG[0] * tmp2[0][n] + C_PROLONG[1] * tmp2[1][n] + C_PROLONG[2] * tmp2[2][n] +
                   C_PROLONG[3] * tmp2[3][n] + C_PROLONG[4] * tmp2[4][n] + C_PROLONG[5] * tmp2[5][n];
        } else {
            val += C_PROLONG[5] * tmp2[0][n] + C_PROLONG[4] * tmp2[1][n] + C_PROLONG[3] * tmp2[2][n] +
                   C_PROLONG[2] * tmp2[3][n] + C_PROLONG[1] * tmp2[4][n] + C_PROLONG[0] * tmp2[5][n];
        }
        tmp1[n] = val;
    }

    // X-Direction Interpolation
    double final_val = 0.0;
    if (i_even) {
        final_val += C_PROLONG[0] * tmp1[0] + C_PROLONG[1] * tmp1[1] + C_PROLONG[2] * tmp1[2] +
                     C_PROLONG[3] * tmp1[3] + C_PROLONG[4] * tmp1[4] + C_PROLONG[5] * tmp1[5];
    } else {
        final_val += C_PROLONG[5] * tmp1[0] + C_PROLONG[4] * tmp1[1] + C_PROLONG[3] * tmp1[2] +
                     C_PROLONG[2] * tmp1[3] + C_PROLONG[1] * tmp1[4] + C_PROLONG[0] * tmp1[5];
    }

    // Write Output (0-based index)
    int out_idx = get_col_major_idx(i, j, k, extf[0], extf[1], extf[2]);
    funf[out_idx] = final_val;
}

// ==========================================
// 3. Restriction Device Function
// ==========================================

// Calculate restricted value for a single point (i, j, k) on the COARSE grid.
// i, j, k are 0-based indices [0, extc-1] (conceptually within the valid restricted range)
__device__ void d_restrict3_device(
    int i, int j, int k, 
    const double* llbc, const double* uubc, const int* extc, double* func, // func is output
    const double* llbf, const double* uubf, const int* extf, const double* funf, // funf is input
    const double* llbr, const double* uubr,
    const double* SoA, int Symmetry
) {
    // --- 1. Geometry & Alignment ---
    double CD[3], FD[3];
    double base[3];
    int lbc[3], lbf[3], lbr[3], ubr[3];

    for (int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
    }

    for (int d = 0; d < 3; d++) {
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = d_idint((llbc[d] - llbf[d]) / FD[d] + 0.4f);
            if ((j_val / 2) * 2 == j_val) {
                base[d] = llbf[d];
            } else {
                base[d] = llbf[d] - CD[d] / 2.0;
            }
        }
    }

    for (int d = 0; d < 3; d++) {
        lbf[d] = d_idint((llbf[d] - base[d]) / FD[d] + 0.4f) + 1;
        lbc[d] = d_idint((llbc[d] - base[d]) / CD[d] + 0.4f) + 1;
        lbr[d] = d_idint((llbr[d] - base[d]) / CD[d] + 0.4f) + 1;
        ubr[d] = d_idint((uubr[d] - base[d]) / CD[d] + 0.4f);
    }

    // Range Check
    int imino = lbr[0] - lbc[0] + 1;
    int imaxo = ubr[0] - lbc[0] + 1;
    int jmino = lbr[1] - lbc[1] + 1;
    int jmaxo = ubr[1] - lbc[1] + 1;
    int kmino = lbr[2] - lbc[2] + 1;
    int kmaxo = ubr[2] - lbc[2] + 1;

    int i_1b = i + 1;
    int j_1b = j + 1;
    int k_1b = k + 1;

    if (i_1b < imino || i_1b > imaxo || j_1b < jmino || j_1b > jmaxo || k_1b < kmino || k_1b > kmaxo) {
        return;
    }

    // --- 2. Index Mapping ---
    
    // Coarse to Fine mapping
    // Fortran: cxI = i; cxI = 2*(cxI+lbc-1) - 1; cxI = cxI - lbf + 1
    // CUDA: use i_1b for 'i'
    int if_fine = 2 * (i_1b + lbc[0] - 1) - 1 - lbf[0] + 1;
    int jf_fine = 2 * (j_1b + lbc[1] - 1) - 1 - lbf[1] + 1;
    int kf_fine = 2 * (k_1b + lbc[2] - 1) - 1 - lbf[2] + 1;

    // --- 3. Restriction ---
    double tmp2[6][6];
    double tmp1[6];

    // Z-Direction Restriction
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_jf = jf_fine - 2 + m;
            int cur_if = if_fine - 2 + n;
            
            double val = 0.0;
            // Ord=2 passed to symmetry_bd as per Fortran restrict3
            // Indices: -2, -1, 0, 1, 2, 3 relative to fine center
            val += C_RESTRICT[0] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 2, SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 3, SoA)
            );
            val += C_RESTRICT[1] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 1, SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 2, SoA)
            );
            val += C_RESTRICT[2] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine    , SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 1, SoA)
            );
            
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Restriction
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        val += C_RESTRICT[0] * (tmp2[0][n] + tmp2[5][n]);
        val += C_RESTRICT[1] * (tmp2[1][n] + tmp2[4][n]);
        val += C_RESTRICT[2] * (tmp2[2][n] + tmp2[3][n]);
        tmp1[n] = val;
    }

    // X-Direction Restriction
    double final_val = 0.0;
    final_val += C_RESTRICT[0] * (tmp1[0] + tmp1[5]);
    final_val += C_RESTRICT[1] * (tmp1[1] + tmp1[4]);
    final_val += C_RESTRICT[2] * (tmp1[2] + tmp1[3]);

    // Write Output (0-based index)
    int out_idx = get_col_major_idx(i, j, k, extc[0], extc[1], extc[2]);
    func[out_idx] = final_val;
}

// ++++++++++++++ Kernel Implementation ++++++++++++++
// ---------------------------------------------------------
// 1. 直接面向显存的单任务 Prolong Kernel
// ---------------------------------------------------------
__global__ void prolong3_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    double llbc0, double llbc1, double llbc2,
    double uubc0, double uubc1, double uubc2,
    int extc0, int extc1, int extc2,
    const double* __restrict__ d_src_c,
    double llbf0, double llbf1, double llbf2,
    double uubf0, double uubf1, double uubf2,
    int extf0, int extf1, int extf2,
    double* __restrict__ d_dst_f,
    double llbt0, double llbt1, double llbt2,
    double uubt0, double uubt1, double uubt2,
    double SoA0, double SoA1, double SoA2,
    int Symmetry
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ni * nj * nk;
    if (idx >= total) return;

    // 1D to 3D mapping
    int k_local = idx / (ni * nj);
    int rem     = idx % (ni * nj);
    int j_local = rem / ni;
    int i_local = rem % ni;

    // 0-based Fortran equivalent loop indices
    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;

    // Kernel 内局部组装为数组，安全调用底层的 device 函数
    double arr_llbc[3] = {llbc0, llbc1, llbc2};
    double arr_uubc[3] = {uubc0, uubc1, uubc2};
    int    arr_extc[3] = {extc0, extc1, extc2};
    
    double arr_llbf[3] = {llbf0, llbf1, llbf2};
    double arr_uubf[3] = {uubf0, uubf1, uubf2};
    int    arr_extf[3] = {extf0, extf1, extf2};

    double arr_llbt[3] = {llbt0, llbt1, llbt2};
    double arr_uubt[3] = {uubt0, uubt1, uubt2};
    double arr_SoA[3]  = {SoA0, SoA1, SoA2};

    // 执行插值，直接将结果写入细网格显存
    d_prolong3_device(
        i, j, k,
        arr_llbc, arr_uubc, arr_extc, d_src_c,
        arr_llbf, arr_uubf, arr_extf, d_dst_f,
        arr_llbt, arr_uubt,
        arr_SoA, Symmetry
    );
}

// ---------------------------------------------------------
// 2. 直接面向显存的单任务 Restrict Kernel
// ---------------------------------------------------------
__global__ void restrict3_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    double llbc0, double llbc1, double llbc2,
    double uubc0, double uubc1, double uubc2,
    int extc0, int extc1, int extc2,
    double* __restrict__ d_dst_c, // Restrict 目标是粗网格
    double llbf0, double llbf1, double llbf2,
    double uubf0, double uubf1, double uubf2,
    int extf0, int extf1, int extf2,
    const double* __restrict__ d_src_f,
    double llbt0, double llbt1, double llbt2,
    double uubt0, double uubt1, double uubt2,
    double SoA0, double SoA1, double SoA2,
    int Symmetry
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ni * nj * nk;
    if (idx >= total) return;

    int k_local = idx / (ni * nj);
    int rem     = idx % (ni * nj);
    int j_local = rem / ni;
    int i_local = rem % ni;

    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;

    double arr_llbc[3] = {llbc0, llbc1, llbc2};
    double arr_uubc[3] = {uubc0, uubc1, uubc2};
    int    arr_extc[3] = {extc0, extc1, extc2};
    
    double arr_llbf[3] = {llbf0, llbf1, llbf2};
    double arr_uubf[3] = {uubf0, uubf1, uubf2};
    int    arr_extf[3] = {extf0, extf1, extf2};

    double arr_llbt[3] = {llbt0, llbt1, llbt2};
    double arr_uubt[3] = {uubt0, uubt1, uubt2};
    double arr_SoA[3]  = {SoA0, SoA1, SoA2};

    d_restrict3_device(
        i, j, k,
        arr_llbc, arr_uubc, arr_extc, d_dst_c,
        arr_llbf, arr_uubf, arr_extf, d_src_f,
        arr_llbt, arr_uubt,
        arr_SoA, Symmetry
    );
}

// ---------------------------------------------------------
// 3. Host 端启动接口
// ---------------------------------------------------------
void gpu_prolong3_launch(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {
    double CD[3], FD[3], base[3];
    for(int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            // 修正：使用 std::trunc 完美对齐 Fortran 的 idint (向零取整)
            int j_val = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            if ((j_val / 2) * 2 == j_val) base[d] = llbf[d];
            else base[d] = llbf[d] - CD[d] / 2.0;
        }
    }

    int i_start, i_end, j_start, j_end, k_start, k_end;
    for(int d = 0; d < 3; d++) {
        // 修正：使用 std::trunc
        int lbp = (int)std::trunc((llbt[d] - base[d]) / FD[d] + 0.4) + 1;
        int ubp = (int)std::trunc((uubt[d] - base[d]) / FD[d] + 0.4);
        int lbf = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        
        if (d == 0) { i_start = lbp - lbf; i_end = ubp - lbf; }
        if (d == 1) { j_start = lbp - lbf; j_end = ubp - lbf; }
        if (d == 2) { k_start = lbp - lbf; k_end = ubp - lbf; }
    }

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return; // 剔除空操作

    int total_points = ni * nj * nk;
    int block = 256;
    int grid = (total_points + block - 1) / block;

    prolong3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        d_src_c,
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        d_dst_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}

void gpu_restrict3_launch(
    cudaStream_t stream,
    const double* d_src_f, double* d_dst_c,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {
    double CD[3], FD[3], base[3];
    for(int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            if ((j_val / 2) * 2 == j_val) base[d] = llbf[d];
            else base[d] = llbf[d] - CD[d] / 2.0;
        }
    }

    int i_start, i_end, j_start, j_end, k_start, k_end;
    for(int d = 0; d < 3; d++) {
        int lbr = (int)std::trunc((llbt[d] - base[d]) / CD[d] + 0.4) + 1;
        int ubr = (int)std::trunc((uubt[d] - base[d]) / CD[d] + 0.4);
        int lbc = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        
        if (d == 0) { i_start = lbr - lbc; i_end = ubr - lbc; }
        if (d == 1) { j_start = lbr - lbc; j_end = ubr - lbc; }
        if (d == 2) { k_start = lbr - lbc; k_end = ubr - lbc; }
    }

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return;

    int total_points = ni * nj * nk;
    int block = 256;
    int grid = (total_points + block - 1) / block;

    restrict3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        d_dst_c,
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        d_src_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}