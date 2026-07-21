#include "surface_integral.h"

#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include "helper.h"
#include "fadmquantites_bssn.h"
#include "misc.h"

#define PI M_PI

__global__ void scale_normals_kernel(
    int n_tot, double rex, 
    const double* d_nx, const double* d_ny, const double* d_nz,
    double* d_pox_0, double* d_pox_1, double* d_pox_2
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < n_tot) {
        d_pox_0[n] = rex * d_nx[n];
        d_pox_1[n] = rex * d_ny[n];
        d_pox_2[n] = rex * d_nz[n];
    }
}

void gpu_scale_normals_launch(
    cudaStream_t stream, int n_tot, double rex, 
    const double* d_nx, const double* d_ny, const double* d_nz,
    double* d_pox_0, double* d_pox_1, double* d_pox_2
) {
    int blockSize = 256;
    int gridSize = (n_tot + blockSize - 1) / blockSize;
    
    scale_normals_kernel<<<gridSize, blockSize, 0, stream>>>(
        n_tot, rex, d_nx, d_ny, d_nz, d_pox_0, d_pox_1, d_pox_2
    );
}

__global__ void surf_MassPAng_kernel(
    int Nmin, int Nmax, int N_phi, int InList, int Symmetry,
    const double* d_shellf, 
    const double* d_pox_0, const double* d_pox_1, const double* d_pox_2,
    const double* d_nx, const double* d_ny, const double* d_nz,
    const double* d_wtcostheta,
    double* d_reductions
) {
    int n = Nmin + blockIdx.x * blockDim.x + threadIdx.x;
    
    // 线程局部的累加器
    double mass = 0, px = 0, py = 0, pz = 0, sx = 0, sy = 0, sz = 0;

    if (n <= Nmax) {
        int i = n / N_phi; 
        
        // 提取物理量
        double Chi = d_shellf[InList * n + 3];
        double TRK = d_shellf[InList * n + 4];
        double Gxx = d_shellf[InList * n + 5] + 1.0;
        double Gxy = d_shellf[InList * n + 6];
        double Gxz = d_shellf[InList * n + 7];
        double Gyy = d_shellf[InList * n + 8] + 1.0;
        double Gyz = d_shellf[InList * n + 9];
        double Gzz = d_shellf[InList * n + 10] + 1.0;
        double axx = d_shellf[InList * n + 11];
        double axy = d_shellf[InList * n + 12];
        double axz = d_shellf[InList * n + 13];
        double ayy = d_shellf[InList * n + 14];
        double ayz = d_shellf[InList * n + 15];
        double azz = d_shellf[InList * n + 16];

        Chi = 1.0 / (1.0 + Chi); // exp(4*phi)
        double Psi = Chi * sqrt(Chi); // Psi^6

        double wt = d_wtcostheta[i];
        double nx = d_nx[n];
        double ny = d_ny[n];
        double nz = d_nz[n];
        
        // 1. 计算 Mass
        mass = (d_shellf[InList * n] * nx + d_shellf[InList * n + 1] * ny + d_shellf[InList * n + 2] * nz) * wt;

        // 2. 逆度规运算
        double gupzz_denom = Gxx * Gyy * Gzz + Gxy * Gyz * Gxz + Gxz * Gxy * Gyz -
                             Gxz * Gyy * Gxz - Gxy * Gxy * Gzz - Gxx * Gyz * Gyz;
        double gupxx = (Gyy * Gzz - Gyz * Gyz) / gupzz_denom;
        double gupxy = -(Gxy * Gzz - Gyz * Gxz) / gupzz_denom;
        double gupxz = (Gxy * Gyz - Gyy * Gxz) / gupzz_denom;
        double gupyy = (Gxx * Gzz - Gxz * Gxz) / gupzz_denom;
        double gupyz = -(Gxx * Gyz - Gxy * Gxz) / gupzz_denom;
        double gupzz = (Gxx * Gyy - Gxy * Gxy) / gupzz_denom;

        double aupxx = gupxx * axx + gupxy * axy + gupxz * axz;
        double aupxy = gupxx * axy + gupxy * ayy + gupxz * ayz;
        double aupxz = gupxx * axz + gupxy * ayz + gupxz * azz;
        double aupyx = gupxy * axx + gupyy * axy + gupyz * axz;
        double aupyy = gupxy * axy + gupyy * ayy + gupyz * ayz;
        double aupyz = gupxy * axz + gupyy * ayz + gupyz * azz;
        double aupzx = gupxz * axx + gupyz * axy + gupzz * axz;
        double aupzy = gupxz * axy + gupyz * ayy + gupzz * ayz;
        double aupzz = gupxz * axz + gupyz * ayz + gupzz * azz;

        double px_coord = d_pox_0[n];
        double py_coord = d_pox_1[n];
        double pz_coord = d_pox_2[n];
        const double f1o8 = 0.125;

        // 3. 计算 Angular Momentum
        if (Symmetry == 0) {
            sx = f1o8 * Psi * (nx * (py_coord * aupxz - pz_coord * aupxy) + ny * (py_coord * aupyz - pz_coord * aupyy) + nz * (py_coord * aupzz - pz_coord * aupzy)) * wt;
            sy = f1o8 * Psi * (nx * (pz_coord * aupxx - px_coord * aupxz) + ny * (pz_coord * aupyx - px_coord * aupyz) + nz * (pz_coord * aupzx - px_coord * aupzz)) * wt;
            sz = f1o8 * Psi * (nx * (px_coord * aupxy - py_coord * aupxx) + ny * (px_coord * aupyy - py_coord * aupyx) + nz * (px_coord * aupzy - py_coord * aupzx)) * wt;
        } else if (Symmetry == 1) {
            sz = f1o8 * Psi * (nx * (px_coord * aupxy - py_coord * aupxx) + ny * (px_coord * aupyy - py_coord * aupyx) + nz * (px_coord * aupzy - py_coord * aupzx)) * wt;
        }

        // 4. 计算 Linear Momentum
        axx = Chi * (axx + Gxx * TRK / 3.0) - TRK;
        axy = Chi * (axy + Gxy * TRK / 3.0);
        axz = Chi * (axz + Gxz * TRK / 3.0);
        ayy = Chi * (ayy + Gyy * TRK / 3.0) - TRK;
        ayz = Chi * (ayz + Gyz * TRK / 3.0);
        azz = Chi * (azz + Gzz * TRK / 3.0) - TRK;

        if (Symmetry == 0) {
            px = f1o8 * Psi * (nx * axx + ny * axy + nz * axz) * wt;
            py = f1o8 * Psi * (nx * axy + ny * ayy + nz * ayz) * wt;
            pz = f1o8 * Psi * (nx * axz + ny * ayz + nz * azz) * wt;
        } else if (Symmetry == 1) {
            px = f1o8 * Psi * (nx * axx + ny * axy + nz * axz) * wt;
            py = f1o8 * Psi * (nx * axy + ny * ayy + nz * ayz) * wt;
        }
    }

    __shared__ double s_red[7];
    if (threadIdx.x == 0) {
        for (int k = 0; k < 7; k++) s_red[k] = 0.0;
    }
    __syncthreads();

    atomicAdd(&s_red[0], mass);
    atomicAdd(&s_red[1], px);
    atomicAdd(&s_red[2], py);
    atomicAdd(&s_red[3], pz);
    atomicAdd(&s_red[4], sx);
    atomicAdd(&s_red[5], sy);
    atomicAdd(&s_red[6], sz);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&d_reductions[0], s_red[0]);
        atomicAdd(&d_reductions[1], s_red[1]);
        atomicAdd(&d_reductions[2], s_red[2]);
        atomicAdd(&d_reductions[3], s_red[3]);
        atomicAdd(&d_reductions[4], s_red[4]);
        atomicAdd(&d_reductions[5], s_red[5]);
        atomicAdd(&d_reductions[6], s_red[6]);
    }
}

void gpu_surf_MassPAng_launch(
    cudaStream_t stream,
    int Nmin, int Nmax, int N_phi, int InList, int Symmetry,
    const double* d_shellf, 
    const double* d_pox_0, const double* d_pox_1, const double* d_pox_2,
    const double* d_nx, const double* d_ny, const double* d_nz,
    const double* d_wtcostheta, double* d_reductions
) {
    int num_elements = Nmax - Nmin + 1;
    if (num_elements <= 0) return;

    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;

    surf_MassPAng_kernel<<<gridSize, blockSize, 0, stream>>>(
        Nmin, Nmax, N_phi, InList, Symmetry,
        d_shellf, 
        d_pox_0, d_pox_1, d_pox_2,
        d_nx, d_ny, d_nz, d_wtcostheta,
        d_reductions
    );
}

void surface_integral::gpu_surf_MassPAng(
    double rex, int lev, cgh *GH, var *chi, var *trK,
    var *gxx, var *gxy, var *gxz, var *gyy, var *gyz, var *gzz,
    var *Axx, var *Axy, var *Axz, var *Ayy, var *Ayz, var *Azz,
    var *Gmx, var *Gmy, var *Gmz,
    var *Sfx_rhs, var *Sfy_rhs, var *Sfz_rhs, // temparay memory for mass^i
    double *Rout, monitor *Monitor
) {
    if (myrank == 0 && GH->grids[lev] != 1)
        if (Monitor && Monitor->outfile)
            Monitor->outfile << "WARNING: surface integral on multipatches" << endl;
        else
            cout << "WARNING: surface integral on multipatches" << endl;

    double mass, px, py, pz, sx, sy, sz;

    const int InList = 17;

    MyList<var> *DG_List = new MyList<var>(Sfx_rhs);
    DG_List->insert(Sfy_rhs);
    DG_List->insert(Sfz_rhs);
    DG_List->insert(chi);
    DG_List->insert(trK);
    DG_List->insert(gxx);
    DG_List->insert(gxy);
    DG_List->insert(gxz);
    DG_List->insert(gyy);
    DG_List->insert(gyz);
    DG_List->insert(gzz);
    DG_List->insert(Axx);
    DG_List->insert(Axy);
    DG_List->insert(Axz);
    DG_List->insert(Ayy);
    DG_List->insert(Ayz);
    DG_List->insert(Azz);

    // Helper::move_to_gpu_whole(GH->PatL[lev], myrank, DG_List);

    MyList<Patch> *Pp = GH->PatL[lev];
    while (Pp) {
        MyList<Block> *BP = Pp->data->blb;
        while (BP) {
            Block *cg = BP->data;
            if (myrank == cg->rank) {
                gpu_admmass_bssn_launch(
                    cg->stream,
                    cg->shape, cg->d_X[0], cg->d_X[1], cg->d_X[2],
                    cg->d_fgfs[chi->sgfn], cg->d_fgfs[trK->sgfn],
                    cg->d_fgfs[gxx->sgfn], cg->d_fgfs[gxy->sgfn], cg->d_fgfs[gxz->sgfn], cg->d_fgfs[gyy->sgfn], cg->d_fgfs[gyz->sgfn], cg->d_fgfs[gzz->sgfn],
                    cg->d_fgfs[Axx->sgfn], cg->d_fgfs[Axy->sgfn], cg->d_fgfs[Axz->sgfn], cg->d_fgfs[Ayy->sgfn], cg->d_fgfs[Ayz->sgfn], cg->d_fgfs[Azz->sgfn],
                    cg->d_fgfs[Gmx->sgfn], cg->d_fgfs[Gmy->sgfn], cg->d_fgfs[Gmz->sgfn],
                    cg->d_fgfs[Sfx_rhs->sgfn], cg->d_fgfs[Sfy_rhs->sgfn], cg->d_fgfs[Sfz_rhs->sgfn],
                    Symmetry
                );
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }
    GPUManager::getInstance().synchronize_all();

    // int n;
    double *d_pox[3];
    for (int i = 0; i < 3; i++) d_pox[i] = GPUManager::getInstance().allocate_device_memory(n_tot);
    gpu_scale_normals_launch(this->stream, n_tot, rex, d_nx_g, d_ny_g, d_nz_g, d_pox[0], d_pox[1], d_pox[2]);

    double* d_shellf = GPUManager::getInstance().allocate_device_memory(n_tot * InList);
    CUDA_CHECK(cudaMemset(d_shellf, 0, n_tot * InList * sizeof(double)));

    GH->PatL[lev]->data->Interp_Points_GPU(this->stream, DG_List, n_tot, d_pox, d_shellf, Symmetry);

    const double f1o8 = 0.125;

    int mp, Lp, Nmin, Nmax;

    mp = n_tot / cpusize;
    Lp = n_tot - cpusize * mp;

    if (Lp > myrank)
    {
        Nmin = myrank * mp + myrank;
        Nmax = Nmin + mp;
    }
    else
    {
        Nmin = myrank * mp + Lp;
        Nmax = Nmin + mp - 1;
    }

    double h_reductions[7] = {0}; 
    double* d_reductions = GPUManager::getInstance().allocate_device_memory(7 * sizeof(double));
    cudaMemset(d_reductions, 0, 7 * sizeof(double));
    gpu_surf_MassPAng_launch(
        this->stream,
        Nmin, Nmax, N_phi, InList, Symmetry,
        d_shellf, d_pox[0], d_pox[1], d_pox[2], d_nx_g, d_ny_g, d_nz_g, d_wtcostheta, d_reductions
    );
    GPUManager::getInstance().synchronize_all();
    GPUManager::getInstance().sync_to_cpu(h_reductions, d_reductions, 7);
    double Mass_out = h_reductions[0];
    double p_outx   = h_reductions[1];
    double p_outy   = h_reductions[2];
    double p_outz   = h_reductions[3];
    double ang_outx = h_reductions[4];
    double ang_outy = h_reductions[5];
    double ang_outz = h_reductions[6];

    MPI_Allreduce(&Mass_out, &mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&ang_outx, &sx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ang_outy, &sy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ang_outz, &sz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&p_outx, &px, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&p_outy, &py, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&p_outz, &pz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mass = mass * rex * rex * dphi * factor;

    sx = sx * rex * rex * dphi * (1.0 / PI) * factor;
    sy = sy * rex * rex * dphi * (1.0 / PI) * factor;
    sz = sz * rex * rex * dphi * (1.0 / PI) * factor;

    px = px * rex * rex * dphi * (1.0 / PI) * factor;
    py = py * rex * rex * dphi * (1.0 / PI) * factor;
    pz = pz * rex * rex * dphi * (1.0 / PI) * factor;

    Rout[0] = mass;
    Rout[1] = px;
    Rout[2] = py;
    Rout[3] = pz;
    Rout[4] = sx;
    Rout[5] = sy;
    Rout[6] = sz;

    GPUManager::getInstance().free_device_memory(d_pox[0], n_tot);
    GPUManager::getInstance().free_device_memory(d_pox[1], n_tot);
    GPUManager::getInstance().free_device_memory(d_pox[2], n_tot);
    GPUManager::getInstance().free_device_memory(d_shellf, n_tot * InList);

    DG_List->clearList();
}

__global__ void surf_Wave_kernel(
    int Nmin, int Nmax, int N_phi, int Symmetry,
    int spinw, int maxl, int NN,
    const double* d_shellf, 
    const double* d_wtcostheta, const double* d_arcostheta,
    double dphi,
    double Rpsi4_SoA_0, double Rpsi4_SoA_1, double Rpsi4_SoA_2, // 通过值传递小数组
    double Ipsi4_SoA_0, double Ipsi4_SoA_1, double Ipsi4_SoA_2,
    double* d_RP_out, double* d_IP_out
) {
    int n = Nmin + blockIdx.x * blockDim.x + threadIdx.x;
    if (n > Nmax) return;

    int i = n / N_phi;
    int j = n - i * N_phi;

    int lpsy = 0;
    if (Symmetry == 0) lpsy = 1;
    else if (Symmetry == 1) lpsy = 2;
    else if (Symmetry == 2) lpsy = 8;

    double wt = d_wtcostheta[i];
    double costheta_base = d_arcostheta[i];

    int countlm = 0;
    for (int pl = spinw; pl <= maxl; pl++) {
        for (int pm = -pl; pm <= pl; pm++) {
            double costheta_base = d_arcostheta[i];
            double phi_base = (j + 0.5) * dphi;
            double psi4RR_base = d_shellf[2 * n];
            double psi4II_base = d_shellf[2 * n + 1];

            double local_RP = 0.0;
            double local_IP = 0.0;

            for (int lp = 0; lp < lpsy; lp++) {
                double costheta = costheta_base;
                double phi_angle = phi_base;
                double psi4RR = psi4RR_base;
                double psi4II = psi4II_base;

                switch (lp) {
                    case 0: // +++ (theta, phi)
                        break;
                    case 1: // ++- (pi-theta, phi)
                        costheta = -costheta_base;
                        psi4RR *= Rpsi4_SoA_2;
                        psi4II *= Ipsi4_SoA_2;
                        break;
                    case 2: // +-+ (theta, 2*pi-phi) => 等价于 -phi
                        phi_angle = -phi_base; 
                        psi4RR *= Rpsi4_SoA_1;
                        psi4II *= Ipsi4_SoA_1;
                        break;
                    case 3: // +-- (pi-theta, 2*pi-phi)
                        costheta = -costheta_base;
                        phi_angle = -phi_base;
                        psi4RR *= (Rpsi4_SoA_2 * Rpsi4_SoA_1);
                        psi4II *= (Ipsi4_SoA_2 * Ipsi4_SoA_1);
                        break;
                    case 4: // -++ (theta, pi-phi)
                        phi_angle = M_PI - phi_base;
                        psi4RR *= Rpsi4_SoA_0;
                        psi4II *= Ipsi4_SoA_0;
                        break;
                    case 5: // -+- (pi-theta, pi-phi)
                        costheta = -costheta_base;
                        phi_angle = M_PI - phi_base;
                        psi4RR *= (Rpsi4_SoA_2 * Rpsi4_SoA_0);
                        psi4II *= (Ipsi4_SoA_2 * Ipsi4_SoA_0);
                        break;
                    case 6: // --+ (theta, pi+phi)
                        phi_angle = M_PI + phi_base;
                        psi4RR *= (Rpsi4_SoA_1 * Rpsi4_SoA_0);
                        psi4II *= (Ipsi4_SoA_1 * Ipsi4_SoA_0);
                        break;
                    case 7: // --- (pi-theta, pi+phi)
                        costheta = -costheta_base;
                        phi_angle = M_PI + phi_base;
                        psi4RR *= (Rpsi4_SoA_2 * Rpsi4_SoA_1 * Rpsi4_SoA_0);
                        psi4II *= (Ipsi4_SoA_2 * Ipsi4_SoA_1 * Ipsi4_SoA_0);
                        break;
                }

                double cosmphi = cos(pm * phi_angle);
                double sinmphi = sin(pm * phi_angle);

                double thetap = sqrt((2.0 * pl + 1.0) / (4.0 * M_PI)) * misc::wigner_d_device(pl, pm, spinw, costheta);

                local_RP += thetap * (psi4RR * cosmphi + psi4II * sinmphi) * wt;
                local_IP += thetap * (psi4II * cosmphi - psi4RR * sinmphi) * wt;
            }

            atomicAdd(&d_RP_out[countlm], local_RP);
            atomicAdd(&d_IP_out[countlm], local_IP);
            
            countlm++;
        }
    }
}

void gpu_surf_Wave_launch(
    cudaStream_t stream,
    int Nmin, int Nmax, int N_phi, int Symmetry,
    int spinw, int maxl, int NN,
    const double* d_shellf,
    const double* d_wtcostheta, const double* d_arcostheta,
    double dphi,
    double Rpsi4_SoA_0, double Rpsi4_SoA_1, double Rpsi4_SoA_2,
    double Ipsi4_SoA_0, double Ipsi4_SoA_1, double Ipsi4_SoA_2,
    double* d_RP_out, double* d_IP_out
) {
    int num_elements = Nmax - Nmin + 1;
    if (num_elements <= 0) return;

    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;

    surf_Wave_kernel<<<gridSize, blockSize, 0, stream>>>(
        Nmin, Nmax, N_phi, Symmetry,
        spinw, maxl, NN,
        d_shellf,
        d_wtcostheta, d_arcostheta,
        dphi,
        Rpsi4_SoA_0, Rpsi4_SoA_1, Rpsi4_SoA_2,
        Ipsi4_SoA_0, Ipsi4_SoA_1, Ipsi4_SoA_2,
        d_RP_out, d_IP_out
    );
}

void surface_integral::gpu_surf_Wave(
    double rex, int lev, cgh *GH, var *Rpsi4, var *Ipsi4,
    int spinw, int maxl, int NN, double *RP, double *IP,
    monitor *Monitor
) {
    if (myrank == 0 && GH->grids[lev] != 1) {
        if (Monitor && Monitor->outfile)
            Monitor->outfile << "WARNING: surface integral on multipatches" << endl;
        else
            cout << "WARNING: surface integral on multipatches" << endl;
    }

    const int InList = 2; // Rpsi4 和 Ipsi4

    MyList<var> *DG_List = new MyList<var>(Rpsi4);
    DG_List->insert(Ipsi4);

    // Helper::move_to_gpu_whole(GH->PatL[lev], myrank, DG_List);

    // 1. 生成球面插值点坐标 (完全在 GPU 上，无 Host 开销)
    double *d_pox[3];
    for (int i = 0; i < 3; i++) {
        d_pox[i] = GPUManager::getInstance().allocate_device_memory(n_tot);
    }
    gpu_scale_normals_launch(this->stream, n_tot, rex, d_nx_g, d_ny_g, d_nz_g, d_pox[0], d_pox[1], d_pox[2]);

    // 2. 准备接收插值结果的 GPU 数组
    double* d_shellf = GPUManager::getInstance().allocate_device_memory(n_tot * InList);
    CUDA_CHECK(cudaMemset(d_shellf, 0, n_tot * InList * sizeof(double)));

    // 3. 执行全 GPU 的插值过程
    GH->PatL[lev]->data->Interp_Points_GPU(this->stream, DG_List, n_tot, d_pox, d_shellf, Symmetry);

    // 4. 任务分配，计算当前 rank 负责的积分球点范围 (Nmin 到 Nmax)
    int mp, Lp, Nmin, Nmax;
    mp = n_tot / cpusize;
    Lp = n_tot - cpusize * mp;

    if (Lp > myrank) {
        Nmin = myrank * mp + myrank;
        Nmax = Nmin + mp;
    } else {
        Nmin = myrank * mp + Lp;
        Nmax = Nmin + mp - 1;
    }

    // 5. 准备 GPU 端的输出数组，并清零
    double *d_RP_out = GPUManager::getInstance().allocate_device_memory(NN * sizeof(double));
    double *d_IP_out = GPUManager::getInstance().allocate_device_memory(NN * sizeof(double));
    CUDA_CHECK(cudaMemset(d_RP_out, 0, NN * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_IP_out, 0, NN * sizeof(double)));

    // 6. 提取变量自带的对称性 SoA 标记（用于传入 Kernel 做边界判断）
    double R_SoA[3] = {Rpsi4->SoA[0], Rpsi4->SoA[1], Rpsi4->SoA[2]};
    double I_SoA[3] = {Ipsi4->SoA[0], Ipsi4->SoA[1], Ipsi4->SoA[2]};

    // 7. 启动引力波提取核心积分 Kernel
    gpu_surf_Wave_launch(
        this->stream,
        Nmin, Nmax, N_phi, Symmetry,
        spinw, maxl, NN,
        d_shellf, d_wtcostheta, d_arcostheta, dphi,
        R_SoA[0], R_SoA[1], R_SoA[2],
        I_SoA[0], I_SoA[1], I_SoA[2],
        d_RP_out, d_IP_out
    );

    // 同步并拉取积分计算结果
    GPUManager::getInstance().synchronize_all();
    
    double *h_RP_out = new double[NN];
    double *h_IP_out = new double[NN];
    GPUManager::getInstance().sync_to_cpu(h_RP_out, d_RP_out, NN);
    GPUManager::getInstance().sync_to_cpu(h_IP_out, d_IP_out, NN);

    // 8. 全局规约 (还原 Host 侧的乘法系数及 MPI 聚合)
    for (int ii = 0; ii < NN; ii++) {
        h_RP_out[ii] = h_RP_out[ii] * rex * dphi;
        h_IP_out[ii] = h_IP_out[ii] * rex * dphi;
    }

    MPI_Allreduce(h_RP_out, RP, NN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(h_IP_out, IP, NN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 9. 清理内存
    GPUManager::getInstance().free_device_memory(d_pox[0], n_tot);
    GPUManager::getInstance().free_device_memory(d_pox[1], n_tot);
    GPUManager::getInstance().free_device_memory(d_pox[2], n_tot);
    GPUManager::getInstance().free_device_memory(d_shellf, n_tot * InList);
    GPUManager::getInstance().free_device_memory(d_RP_out, NN);
    GPUManager::getInstance().free_device_memory(d_IP_out, NN);

    delete[] h_RP_out;
    delete[] h_IP_out;
    DG_List->clearList();
}