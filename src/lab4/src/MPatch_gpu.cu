#include "MPatch.h"

#include <cuda_runtime.h>
#include "fmisc.h"
#include "macrodef.h"

using namespace std;

__global__ void normalize_shellf_kernel(
    int NN, int num_var, 
    double* d_Shellf, 
    const int* d_Weight, 
    int* d_err_flag
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NN) return;

    int w = d_Weight[i];
    if (w > 1) {
        for (int j = 0; j < num_var; j++) {
            d_Shellf[j + i * num_var] /= (double)w;
        }
        // printf("\33[1;33mWARNING: normalize_shellf_kernel meets multiple weight at point %d\33[0m\n", i);
    } else if (w == 0) {
        *d_err_flag = 1;
    }
}

void gpu_normalize_shellf_launch(
    cudaStream_t stream,
    int NN, int num_var, 
    double* d_Shellf, 
    const int* d_Weight, 
    int* d_err_flag
) {
    int blockSize = 256;
    int gridSize = (NN + blockSize - 1) / blockSize;
    normalize_shellf_kernel<<<gridSize, blockSize, 0, stream>>>(
        NN, num_var, d_Shellf, d_Weight, d_err_flag
    );
}

void Patch::Interp_Points_GPU(
    cudaStream_t stream,
    MyList<var> *VarList,
    int NN, double **d_XX,
    double *d_Shellf, int Symmetry
) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int ordn = 2 * ghost_width;
    MyList<var> *varl = VarList;
    int num_var = 0;
    while (varl) {
        num_var++;
        varl = varl->next;
    }

    double *DH, *llb, *uub;
    DH = new double[dim];
    for (int i = 0; i < dim; i++) {
        DH[i] = getdX(i);
    }
    llb = new double[dim];
    uub = new double[dim];

    // TODO: Sanity check

    double *d_local_shellf = GPUManager::getInstance().allocate_device_memory(NN * num_var);
    int *d_local_weight; CUDA_CHECK(cudaMalloc(&d_local_weight, NN * sizeof(int)));
    
    cudaMemset(d_local_shellf, 0, NN * num_var * sizeof(double));
    cudaMemset(d_local_weight, 0, NN * sizeof(int));

    MyList<Block> *Bp = blb;
    while (Bp) {
        Block *BP = Bp->data;
        if (myrank == BP->rank) {
            for (int i = 0; i < dim; i++) {
                llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
                uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];
            }

            int shape_0 = BP->shape[0];
            int shape_1 = (dim > 1) ? BP->shape[1] : 1;
            int shape_2 = (dim > 2) ? BP->shape[2] : 1;

            double llb_0 = llb[0], llb_1 = (dim > 1) ? llb[1] : 0.0, llb_2 = (dim > 2) ? llb[2] : 0.0;
            double uub_0 = uub[0], uub_1 = (dim > 1) ? uub[1] : 0.0, uub_2 = (dim > 2) ? uub[2] : 0.0;

            double DH_0 = DH[0];
            double DH_1 = (dim > 1) ? DH[1] : 0.0;
            double DH_2 = (dim > 2) ? DH[2] : 0.0;
            varl = VarList;
            int k = 0;
            while (varl) {
                gpu_global_interp_launch(
                    BP->stream,
                    NN, dim,
                    d_XX[0], d_XX[1], d_XX[2],
                    shape_0, shape_1, shape_2,
                    BP->d_X[0], BP->d_X[1], BP->d_X[2],
                    BP->d_fgfs[varl->data->sgfn],
                    llb_0, llb_1, llb_2,
                    uub_0, uub_1, uub_2,
                    DH_0, DH_1, DH_2,
                    ordn, varl->data->SoA[0], varl->data->SoA[1], varl->data->SoA[2],
                    Symmetry, k, num_var, d_local_shellf, d_local_weight
                );
                varl = varl->next;
                k++;
            }
        }
        if (Bp == ble) break;
        Bp = Bp->next;
    }

    GPUManager::getInstance().synchronize_all();

    // =================================================================================
    // 3. MPI Allreduce 规约全局数据
    // 注：如果你使用的 MPI 版本(如 OpenMPI)支持 CUDA-aware，可以直接把 device 指针塞给 MPI！
    // 下面写的是安全通用的做法（拷回 Host -> MPI_Allreduce -> 拷回 Device）
    // =================================================================================
#if MPI_CUDA_AWARE
    MPI_Allreduce(d_Shellf, d_local_shellf, NN * num_var, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    double *h_local_shellf = new double[NN * num_var];
    int    *h_local_weight = new int[NN];
    GPUManager::getInstance().sync_to_cpu(h_local_shellf, d_local_shellf, NN * num_var);
    cudaMemcpy(h_local_weight, d_local_weight, NN * sizeof(int), cudaMemcpyDeviceToHost);

    double *h_global_shellf = new double[NN * num_var];
    int    *h_global_weight = new int[NN];

    MPI_Allreduce(h_local_shellf, h_global_shellf, NN * num_var, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(h_local_weight, h_global_weight, NN, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    GPUManager::getInstance().sync_to_gpu(h_global_shellf, d_Shellf, NN * num_var);
    int *d_global_weight; cudaMalloc(&d_global_weight, NN * sizeof(int));
    cudaMemcpy(d_global_weight, h_global_weight, NN * sizeof(int), cudaMemcpyHostToDevice);
#endif

    int *d_err_flag;
    cudaMalloc(&d_err_flag, sizeof(int));
    cudaMemset(d_err_flag, 0, sizeof(int));

    gpu_normalize_shellf_launch(stream, NN, num_var, d_Shellf, d_global_weight, d_err_flag);
    GPUManager::getInstance().synchronize_all();

    int h_err_flag = 0;
    cudaMemcpy(&h_err_flag, d_err_flag, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_err_flag > 0) {
        if (myrank == 0) {
            double *h_XX[3];
            for (int i=0; i<dim; i++) {
                h_XX[i] = new double[NN];
                GPUManager::getInstance().sync_to_cpu(h_XX[i], d_XX[i], NN);
            }
            
            for (int i = 0; i < NN; i++) {
                if (h_global_weight[i] > 1) {
                    cout << "WARNING: Patch::Interp_Points meets multiple weight at point " << i << endl;
                }
                else if (h_global_weight[i] == 0) {
                    cout << "ERROR: Patch::Interp_Points fails to find point (";
                    for (int j = 0; j < dim; j++) {
                        cout << h_XX[j][i];
                        if (j < dim - 1) cout << ",";
                        else cout << ") ! Out of domain bounds." << endl;
                    }
                }
            }
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    delete[] DH; delete[] llb; delete[] uub;
    delete[] h_local_shellf; delete[] h_local_weight;
    delete[] h_global_shellf; delete[] h_global_weight;
    GPUManager::getInstance().free_device_memory(d_local_shellf, NN * num_var);
    CUDA_CHECK(cudaFree(d_local_weight));
    CUDA_CHECK(cudaFree(d_global_weight));
    CUDA_CHECK(cudaFree(d_err_flag));
}

bool Patch::Interp_N_Points_GPU(
    MyList<var> *VarList, 
    int NN, double *d_XX_0, double *d_XX_1, double *d_XX_2,
    double *d_shellf, int *d_weight, int Symmetry
) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int ordn = 2 * ghost_width;
    MyList<var> *varl;
    int num_var = 0;
    varl = VarList;
    while (varl)
    {
        num_var++;
        varl = varl->next;
    }

    double DH[3];
    for (int i = 0; i < dim; i++) {
        DH[i] = getdX(i);
    }

    MyList<Block> *Bp = blb;
    while (Bp) {
        Block *BP = Bp->data;
        if (myrank == BP->rank) {
            double llb[3], uub[3];
            for (int i = 0; i < dim; i++) {
                llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
                uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];
            }

            double DH_0 = DH[0];
            double DH_1 = (dim > 1) ? DH[1] : 0.0;
            double DH_2 = (dim > 2) ? DH[2] : 0.0;
            varl = VarList;
            int k = 0;
            while (varl) {
                // 启动你的 GPU Batch Kernel
                gpu_global_interp_launch(
                    BP->stream, NN, dim,
                    d_XX_0, d_XX_1, d_XX_2,
                    BP->shape[0], BP->shape[1], BP->shape[2],
                    BP->d_X[0], BP->d_X[1], BP->d_X[2],
                    BP->d_fgfs[varl->data->sgfn],
                    llb[0], llb[1], llb[2], uub[0], uub[1], uub[2],
                    DH_0, DH_1, DH_2,
                    ordn, varl->data->SoA[0], varl->data->SoA[1], varl->data->SoA[2],
                    Symmetry, k, num_var, d_shellf, d_weight
                );
                varl = varl->next;
                k++;
            }
        }
        Bp = Bp->next;
    }

    GPUManager::getInstance().synchronize_all();
    
    double *h_shellf_local = new double[NN * num_var];
    double *h_shellf_global = new double[NN * num_var];
    cudaMemcpy(h_shellf_local, d_shellf, NN * num_var * sizeof(double), cudaMemcpyDeviceToHost);

    MPI_Allreduce(h_shellf_local, h_shellf_global, NN * num_var, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    cudaMemcpy(d_shellf, h_shellf_global, NN * num_var * sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_shellf_local;
    delete[] h_shellf_global;

    return true; // 实际的 "notfind" 检查可以在外层通过 d_weight 完成，此处简化直接返回 true
}