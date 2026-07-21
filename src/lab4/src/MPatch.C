
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cmath>
#include <new>
using namespace std;

#include "misc.h"
#include "MPatch.h"
#include "Parallel.h"
#include "fmisc.h"

Patch::Patch(int DIM, int *shapei, double *bboxi, int levi, bool buflog, int Symmetry) : lev(levi)
{

	int hbuffer_width = buffer_width;
	if (lev == 0)
		hbuffer_width = CS_width; // specific for shell-box coulping

	if (DIM != dim)
	{
		cout << "dimension is not consistent in Patch construction" << endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	for (int i = 0; i < dim; i++)
	{
		shape[i] = shapei[i];
		bbox[i] = bboxi[i];
		bbox[dim + i] = bboxi[dim + i];
		lli[i] = uui[i] = 0;
		if (buflog)
		{
			double DH;
			DH = (bbox[dim + i] - bbox[i]) / shape[i];
			uui[i] = hbuffer_width;
			bbox[dim + i] = bbox[dim + i] + uui[i] * DH;
			shape[i] = shape[i] + uui[i];
		}
	}

	if (buflog)
	{
		if (DIM != 3)
		{
			cout << "Symmetry in Patch construction only support 3 yet but dim = " << DIM << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		double tmpb, DH;
		if (Symmetry > 0)
		{
			DH = (bbox[5] - bbox[2]) / shape[2];
			tmpb = Mymax(0, bbox[2] - hbuffer_width * DH);
			lli[2] = int((bbox[2] - tmpb) / DH + 0.4);
			bbox[2] = bbox[2] - lli[2] * DH;
			shape[2] = shape[2] + lli[2];
			if (lli[2] < hbuffer_width)
			{
				if (feq(bbox[2], 0, DH / 2))
					lli[2] = 0;
				else
				{
					cout << "Code mistake for lli[2] = " << lli[2] << ", bbox[2] = " << bbox[2] << endl;
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
			}
			if (Symmetry > 1)
			{
				DH = (bbox[3] - bbox[0]) / shape[0];
				tmpb = Mymax(0, bbox[0] - hbuffer_width * DH);
				lli[0] = int((bbox[0] - tmpb) / DH + 0.4);
				bbox[0] = bbox[0] - lli[0] * DH;
				shape[0] = shape[0] + lli[0];
				if (lli[0] < hbuffer_width)
				{
					if (feq(bbox[0], 0, DH / 2))
						lli[0] = 0;
					else
					{
						cout << "Code mistake for lli[0] = " << lli[0] << ", bbox[0] = " << bbox[0] << endl;
						MPI_Abort(MPI_COMM_WORLD, 1);
					}
				}
				DH = (bbox[4] - bbox[1]) / shape[1];
				tmpb = Mymax(0, bbox[1] - hbuffer_width * DH);
				lli[1] = int((bbox[1] - tmpb) / DH + 0.4);
				bbox[1] = bbox[1] - lli[1] * DH;
				shape[1] = shape[1] + lli[1];
				if (lli[1] < hbuffer_width)
				{
					if (feq(bbox[1], 0, DH / 2))
						lli[1] = 0;
					else
					{
						cout << "Code mistake for lli[1] = " << lli[1] << ", bbox[1] = " << bbox[1] << endl;
						MPI_Abort(MPI_COMM_WORLD, 1);
					}
				}
			}
			else
			{
				for (int i = 0; i < 2; i++)
				{
					DH = (bbox[dim + i] - bbox[i]) / shape[i];
					lli[i] = hbuffer_width;
					bbox[i] = bbox[i] - lli[i] * DH;
					shape[i] = shape[i] + lli[i];
				}
			}
		}
		else
		{
			for (int i = 0; i < dim; i++)
			{
				DH = (bbox[dim + i] - bbox[i]) / shape[i];
				lli[i] = hbuffer_width;
				bbox[i] = bbox[i] - lli[i] * DH;
				shape[i] = shape[i] + lli[i];
			}
		}
	}

	blb = ble = 0;
}
Patch::~Patch()
{
}
// buflog 1: with buffer points; 0 without
void Patch::checkPatch(bool buflog)
{
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if (myrank == 0)
	{
		if (buflog)
		{
			cout << " belong to level " << lev << endl;
			cout << " shape: [";
			for (int i = 0; i < dim; i++)
			{
				cout << shape[i];
				if (i < dim - 1)
					cout << ",";
				else
					cout << "]";
			}
			cout << " resolution: [";
			for (int i = 0; i < dim; i++)
			{
				cout << getdX(i);
				if (i < dim - 1)
					cout << ",";
				else
					cout << "]" << endl;
			}
			cout << " range:" << "(";
			for (int i = 0; i < dim; i++)
			{
				cout << bbox[i] << ":" << bbox[dim + i];
				if (i < dim - 1)
					cout << ",";
				else
					cout << ")" << endl;
			}
		}
		else
		{
			cout << " belong to level " << lev << endl;
			cout << " shape: [";
			for (int i = 0; i < dim; i++)
			{
				cout << shape[i] - lli[i] - uui[i];
				if (i < dim - 1)
					cout << ",";
				else
					cout << "]";
			}
			cout << " resolution: [";
			for (int i = 0; i < dim; i++)
			{
				cout << getdX(i);
				if (i < dim - 1)
					cout << ",";
				else
					cout << "]" << endl;
			}
			cout << " range:" << "(";
			for (int i = 0; i < dim; i++)
			{
				cout << bbox[i] + lli[i] * getdX(i) << ":" << bbox[dim + i] - uui[i] * getdX(i);
				if (i < dim - 1)
					cout << ",";
				else
					cout << ")" << endl;
			}
		}
	}
}
void Patch::checkPatch(bool buflog, const int out_rank)
{
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if (myrank == out_rank)
	{
		cout << " out_rank = " << out_rank << endl;
		if (buflog)
		{
			cout << " belong to level " << lev << endl;
			cout << " shape: [";
			for (int i = 0; i < dim; i++)
			{
				cout << shape[i];
				if (i < dim - 1)
					cout << ",";
				else
					cout << "]";
			}
			cout << " resolution: [";
			for (int i = 0; i < dim; i++)
			{
				cout << getdX(i);
				if (i < dim - 1)
					cout << ",";
				else
					cout << "]" << endl;
			}
			cout << " range:" << "(";
			for (int i = 0; i < dim; i++)
			{
				cout << bbox[i] << ":" << bbox[dim + i];
				if (i < dim - 1)
					cout << ",";
				else
					cout << ")" << endl;
			}
		}
		else
		{
			cout << " belong to level " << lev << endl;
			cout << " shape: [";
			for (int i = 0; i < dim; i++)
			{
				cout << shape[i] - lli[i] - uui[i];
				if (i < dim - 1)
					cout << ",";
				else
					cout << "]";
			}
			cout << " resolution: [";
			for (int i = 0; i < dim; i++)
			{
				cout << getdX(i);
				if (i < dim - 1)
					cout << ",";
				else
					cout << "]" << endl;
			}
			cout << " range:" << "(";
			for (int i = 0; i < dim; i++)
			{
				cout << bbox[i] + lli[i] * getdX(i) << ":" << bbox[dim + i] - uui[i] * getdX(i);
				if (i < dim - 1)
					cout << ",";
				else
					cout << ")" << endl;
			}
		}
	}
}

void Patch::Interp_Points(MyList<var> *VarList,
                          int NN, double **XX,
                          double *Shellf, int Symmetry)
{
    // NOTE: we do not Synchnize variables here, make sure of that before calling this routine
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

    double *shellf;
    shellf = new double[NN * num_var];
    memset(shellf, 0, sizeof(double) * NN * num_var);

    int *weight;
    weight = new int[NN];
    memset(weight, 0, sizeof(int) * NN);

    double *DH, *llb, *uub;
    DH = new double[dim];

    for (int i = 0; i < dim; i++)
    {
        DH[i] = getdX(i);
    }
    llb = new double[dim];
    uub = new double[dim];

    // --------------------------------------------------------------------------
    // 保留 CPU 端的安全检查（只执行一次，开销极低）
    for (int j = 0; j < NN; j++) 
    {
        for (int i = 0; i < dim; i++)
        {
            if (isnan(XX[i][j])) {
                puts("Error: Patch::Interp_Points encounters NaN in input point coordinates.");
            }
            if (myrank == 0 && (XX[i][j] < bbox[i] + lli[i] * DH[i] || XX[i][j] > bbox[dim + i] - uui[i] * DH[i]))
            {
                cout << "Patch::Interp_Points: point (";
                for (int k = 0; k < dim; k++)
                {
                    cout << XX[k][j];
                    if (k < dim - 1)
                        cout << ",";
                    else
                        cout << ") is out of current Patch." << endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // ================== GPU 零拷贝重构部分 开始 ==================
#ifdef USE_GPU
    // 1. 分配 GPU 内存并将所有插值点坐标一次性拷贝至 GPU
    double *d_XX[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < dim; i++) {
        cudaMalloc((void**)&d_XX[i], NN * sizeof(double));
        cudaMemcpy(d_XX[i], XX[i], NN * sizeof(double), cudaMemcpyHostToDevice);
    }    double *d_shellf;
    cudaMalloc((void**)&d_shellf, NN * num_var * sizeof(double));
    cudaMemset(d_shellf, 0, NN * num_var * sizeof(double));

    int *d_weight;
    cudaMalloc((void**)&d_weight, NN * sizeof(int));
    cudaMemset(d_weight, 0, NN * sizeof(int));

    // 2. Block-Centric 遍历：以 Block 为单位启动 Kernel
    MyList<Block> *Bp = blb;
    while (Bp)
    {        Block *BP = Bp->data;

        if (myrank == BP->rank) 
        {
            // 计算当前 Block 包含幽灵区(ghost zones)的 Bounding Box
            for (int i = 0; i < dim; i++)
            {
                llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
                uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];
            }

            // 提取多维参数（展平传递给 Kernel 以避免隐式 Host 指针问题）
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
            while (varl)
            {
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
                    Symmetry, k, num_var, d_shellf, d_weight
                );
                varl = varl->next;
                k++;
            }
        }
        if (Bp == ble) break;
        Bp = Bp->next;
    }

    GPUManager::getInstance().synchronize_all();
    cudaMemcpy(shellf, d_shellf, NN * num_var * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(weight, d_weight, NN * sizeof(int), cudaMemcpyDeviceToHost);

    // 4. 清理 GPU 临时内存
    for (int i = 0; i < dim; i++) {
        if (d_XX[i]) cudaFree(d_XX[i]);
    }
    cudaFree(d_shellf);
    cudaFree(d_weight);
#else
    for (int j = 0; j < NN; j++)
    {
        double pox[dim];
        for (int i = 0; i < dim; i++)
            pox[i] = XX[i][j];

        MyList<Block> *Bp = blb;
        bool notfind = true;
        while (notfind && Bp)
        {
            Block *BP = Bp->data;

            bool flag = true;
            for (int i = 0; i < dim; i++)
            {
                llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
                uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];

                if (XX[i][j] - llb[i] < -DH[i] / 2 || XX[i][j] - uub[i] > DH[i] / 2)
                {
                    flag = false;
                    break;
                }
            }

            if (flag)
            {
                notfind = false;
                if (myrank == BP->rank)
                {
                    varl = VarList;
                    int k = 0;
                    while (varl)
                    {
                        f_global_interp(BP->shape, BP->X[0], BP->X[1], BP->X[2], BP->fgfs[varl->data->sgfn], shellf[j * num_var + k],
                                        pox[0], pox[1], pox[2], ordn, varl->data->SoA, Symmetry);
                        varl = varl->next;
                        k++;
                    }
                    weight[j] = 1;
                }
            }
            if (Bp == ble)
                break;
            Bp = Bp->next;
        }
    }
#endif

    // ================== GPU 零拷贝重构部分 结束 ==================

#ifdef USE_GPU
    // GPU produced shellf/weight directly on host (above). Compute Weight via Allreduce.
    int *Weight = new int[NN];
    MPI_Allreduce(weight, Weight, NN, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // (shellf was already reduced into Shellf inside the GPU block.)
#else
    // CPU path: Allreduce the local shellf/weight into Shellf/Weight.
    MPI_Allreduce(shellf, Shellf, NN * num_var, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int *Weight;
    Weight = new int[NN];
    MPI_Allreduce(weight, Weight, NN, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    for (int i = 0; i < NN; i++)
    {
        if (Weight[i] > 1)
        {
            if (myrank == 0)
                cout << "WARNING: Patch::Interp_Points meets multiple weight" << endl;
            for (int j = 0; j < num_var; j++)
                Shellf[j + i * num_var] = Shellf[j + i * num_var] / Weight[i];
        }
        else if (Weight[i] == 0 && myrank == 0)
        {
            cout << "ERROR: Patch::Interp_Points fails to find point (";
            for (int j = 0; j < dim; j++)
            {
                cout << XX[j][i];
                if (j < dim - 1)
                    cout << ",";
                else
                    cout << ")";
            }
            cout << " on Patch (";
            for (int j = 0; j < dim; j++)
            {
                cout << bbox[j] << "+" << lli[j] * getdX(j);
                if (j < dim - 1)
                    cout << ",";
                else
                    cout << ")--";
            }
            cout << "(";
            for (int j = 0; j < dim; j++)
            {
                cout << bbox[dim + j] << "-" << uui[j] * getdX(j);
                if (j < dim - 1)
                    cout << ",";
                else
                    cout << ")" << endl;
            }
            cout << "splited domains:" << endl;
            {
                MyList<Block> *Bp = blb;
                while (Bp)
                {
                    Block *BP = Bp->data;

                    for (int i = 0; i < dim; i++)
                    {
                        llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
                        uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];
                    }
                    cout << "(";
                    for (int j = 0; j < dim; j++)
                    {
                        cout << llb[j] << ":" << uub[j];
                        if (j < dim - 1)
                            cout << ",";
                        else
                            cout << ")" << endl;
                    }
                    if (Bp == ble)
                        break;
                    Bp = Bp->next;
                }
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    delete[] shellf;
    delete[] weight;
    delete[] Weight;
    delete[] DH;
    delete[] llb;
    delete[] uub;
}

void Patch::Interp_Points(MyList<var> *VarList,
													int NN, double **XX,
													double *Shellf, int Symmetry, MPI_Comm Comm_here)
{
	// NOTE: we do not Synchnize variables here, make sure of that before calling this routine
	int myrank, lmyrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_rank(Comm_here, &lmyrank);

	int ordn = 2 * ghost_width;
	MyList<var> *varl;
	int num_var = 0;
	varl = VarList;
	while (varl)
	{
		num_var++;
		varl = varl->next;
	}

	double *shellf;
	shellf = new double[NN * num_var];
	memset(shellf, 0, sizeof(double) * NN * num_var);

	// we use weight to monitor code, later some day we can move it for optimization
	int *weight;
	weight = new int[NN];
	memset(weight, 0, sizeof(int) * NN);

	double *DH, *llb, *uub;
	DH = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		DH[i] = getdX(i);
	}
	llb = new double[dim];
	uub = new double[dim];

	for (int j = 0; j < NN; j++) // run along points
	{
		double pox[dim];
		for (int i = 0; i < dim; i++)
		{
			pox[i] = XX[i][j];
			if (lmyrank == 0 && (XX[i][j] < bbox[i] + lli[i] * DH[i] || XX[i][j] > bbox[dim + i] - uui[i] * DH[i]))
			{
				cout << "Patch::Interp_Points: point (";
				for (int k = 0; k < dim; k++)
				{
					cout << XX[k][j];
					if (k < dim - 1)
						cout << ",";
					else
						cout << ") is out of current Patch." << endl;
				}
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}

		MyList<Block> *Bp = blb;
		bool notfind = true;
		while (notfind && Bp) // run along Blocks
		{
			Block *BP = Bp->data;

			bool flag = true;
			for (int i = 0; i < dim; i++)
			{
// NOTE: our dividing structure is (exclude ghost)
// -1 0
//       1  2
// so (0,1) does not belong to any part for vertex structure
// here we put (0,0.5) to left part and (0.5,1) to right part
// BUT for cell structure the bbox is (-1.5,0.5) and (0.5,2.5), there is no missing region at all
				llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
				uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];

				if (XX[i][j] - llb[i] < -DH[i] / 2 || XX[i][j] - uub[i] > DH[i] / 2)
				{
					flag = false;
					break;
				}
			}

			if (flag)
			{
				notfind = false;
				if (myrank == BP->rank)
				{
					//---> interpolation
					varl = VarList;
					int k = 0;
					while (varl) // run along variables
					{
						//              shellf[j*num_var+k] = Parallel::global_interp(dim,BP->shape,BP->X,BP->fgfs[varl->data->sgfn],
						//	  		                                    pox,ordn,varl->data->SoA,Symmetry);
						f_global_interp(BP->shape, BP->X[0], BP->X[1], BP->X[2], BP->fgfs[varl->data->sgfn], shellf[j * num_var + k],
														pox[0], pox[1], pox[2], ordn, varl->data->SoA, Symmetry);
						varl = varl->next;
						k++;
					}
					weight[j] = 1;
				}
			}
			if (Bp == ble)
				break;
			Bp = Bp->next;
		}
	}

	MPI_Allreduce(shellf, Shellf, NN * num_var, MPI_DOUBLE, MPI_SUM, Comm_here);
	int *Weight;
	Weight = new int[NN];
	MPI_Allreduce(weight, Weight, NN, MPI_INT, MPI_SUM, Comm_here);

	//  misc::tillherecheck("print me");
	//  if(lmyrank == 0) cout<<"myrank = "<<myrank<<"print me"<<endl;

	for (int i = 0; i < NN; i++)
	{
		if (Weight[i] > 1)
		{
			if (lmyrank == 0)
				cout << "WARNING: Patch::Interp_Points meets multiple weight" << endl;
			for (int j = 0; j < num_var; j++)
				Shellf[j + i * num_var] = Shellf[j + i * num_var] / Weight[i];
		}
	}

	delete[] shellf;
	delete[] weight;
	delete[] Weight;
	delete[] DH;
	delete[] llb;
	delete[] uub;
}
void Patch::checkBlock()
{
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if (myrank == 0)
	{
		MyList<Block> *BP = blb;
		while (BP)
		{
			BP->data->checkBlock();
			if (BP == ble)
				break;
			BP = BP->next;
		}
	}
}
double Patch::getdX(int dir)
{
	if (dir < 0 || dir >= dim)
	{
		cout << "Patch::getdX: error input dir = " << dir << ", this Patch has direction (0," << dim - 1 << ")" << endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	double h;
	h = (bbox[dim + dir] - bbox[dir]) / shape[dir];
	return h;
}
bool Patch::Interp_ONE_Point(MyList<var> *VarList, double *XX,
														 double *Shellf, int Symmetry)
{
	// NOTE: we do not Synchnize variables here, make sure of that before calling this routine
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

	double *shellf;
	shellf = new double[num_var];
	memset(shellf, 0, sizeof(double) * num_var);

	double *DH, *llb, *uub;
	DH = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		DH[i] = getdX(i);
	}
	llb = new double[dim];
	uub = new double[dim];

	double pox[dim];
	for (int i = 0; i < dim; i++)
	{
		pox[i] = XX[i];
		// has excluded the buffer points
		if (XX[i] < bbox[i] + lli[i] * DH[i] - DH[i] / 100 || XX[i] > bbox[dim + i] - uui[i] * DH[i] + DH[i] / 100)
		{
			delete[] shellf;
			delete[] DH;
			delete[] llb;
			delete[] uub;
			return false; // out of current patch,
										// remember to delete the allocated arrays before return!!!
		}
	}

	MyList<Block> *Bp = blb;
	bool notfind = true;
	while (notfind && Bp) // run along Blocks
	{
		Block *BP = Bp->data;

		bool flag = true;
		for (int i = 0; i < dim; i++)
		{
// NOTE: our dividing structure is (exclude ghost)
// -1 0
//       1  2
// so (0,1) does not belong to any part for vertex structure
// here we put (0,0.5) to left part and (0.5,1) to right part
// BUT for cell structure the bbox is (-1.5,0.5) and (0.5,2.5), there is no missing region at all
			llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
			uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];
			if (XX[i] - llb[i] < -DH[i] / 2 || XX[i] - uub[i] > DH[i] / 2)
			{
				flag = false;
				break;
			}
		}

		if (flag)
		{
			notfind = false;
			if (myrank == BP->rank)
			{
				//---> interpolation
				varl = VarList;
				int k = 0;
				while (varl) // run along variables
				{
					//              shellf[j*num_var+k] = Parallel::global_interp(dim,BP->shape,BP->X,BP->fgfs[varl->data->sgfn],
					//	  		                                    pox,ordn,varl->data->SoA,Symmetry);
					f_global_interp(BP->shape, BP->X[0], BP->X[1], BP->X[2], BP->fgfs[varl->data->sgfn], shellf[k],
													pox[0], pox[1], pox[2], ordn, varl->data->SoA, Symmetry);
					varl = varl->next;
					k++;
				}
			}
		}
		if (Bp == ble)
			break;
		Bp = Bp->next;
	}

	if (notfind && myrank == 0)
	{
		cout << "ERROR: Patch::Interp_Points fails to find point (";
		for (int j = 0; j < dim; j++)
		{
			cout << XX[j];
			if (j < dim - 1)
				cout << ",";
			else
				cout << ")";
		}
		cout << " on Patch (";
		for (int j = 0; j < dim; j++)
		{
			cout << bbox[j] << "+" << lli[j] * getdX(j);
			if (j < dim - 1)
				cout << ",";
			else
				cout << ")--";
		}
		cout << "(";
		for (int j = 0; j < dim; j++)
		{
			cout << bbox[dim + j] << "-" << uui[j] * getdX(j);
			if (j < dim - 1)
				cout << ",";
			else
				cout << ")" << endl;
		}
		cout << "splited domains:" << endl;
		{
			MyList<Block> *Bp = blb;
			while (Bp)
			{
				Block *BP = Bp->data;

				for (int i = 0; i < dim; i++)
				{
					llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
					uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];
				}
				cout << "(";
				for (int j = 0; j < dim; j++)
				{
					cout << llb[j] << ":" << uub[j];
					if (j < dim - 1)
						cout << ",";
					else
						cout << ")" << endl;
				}
				if (Bp == ble)
					break;
				Bp = Bp->next;
			}
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Allreduce(shellf, Shellf, num_var, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	delete[] shellf;
	delete[] DH;
	delete[] llb;
	delete[] uub;

	return true;
}
bool Patch::Interp_ONE_Point(MyList<var> *VarList, double *XX,
														 double *Shellf, int Symmetry, MPI_Comm Comm_here)
{
	// NOTE: we do not Synchnize variables here, make sure of that before calling this routine
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

	double *shellf;
	shellf = new double[num_var];
	memset(shellf, 0, sizeof(double) * num_var);

	double *DH, *llb, *uub;
	DH = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		DH[i] = getdX(i);
	}
	llb = new double[dim];
	uub = new double[dim];

	double pox[dim];
	for (int i = 0; i < dim; i++)
	{
		pox[i] = XX[i];
		// has excluded the buffer points
		if (XX[i] < bbox[i] + lli[i] * DH[i] - DH[i] / 100 || XX[i] > bbox[dim + i] - uui[i] * DH[i] + DH[i] / 100)
		{
			delete[] shellf;
			delete[] DH;
			delete[] llb;
			delete[] uub;
			return false; // out of current patch,
										// remember to delete the allocated arrays before return!!!
		}
	}

	MyList<Block> *Bp = blb;
	bool notfind = true;
	while (notfind && Bp) // run along Blocks
	{
		Block *BP = Bp->data;

		bool flag = true;
		for (int i = 0; i < dim; i++)
		{
// NOTE: our dividing structure is (exclude ghost)
// -1 0
//       1  2
// so (0,1) does not belong to any part for vertex structure
// here we put (0,0.5) to left part and (0.5,1) to right part
// BUT for cell structure the bbox is (-1.5,0.5) and (0.5,2.5), there is no missing region at all
			llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
			uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];
			if (XX[i] - llb[i] < -DH[i] / 2 || XX[i] - uub[i] > DH[i] / 2)
			{
				flag = false;
				break;
			}
		}

		if (flag)
		{
			notfind = false;
			if (myrank == BP->rank)
			{
// test old code
				//---> interpolation
				varl = VarList;
				int k = 0;
				while (varl) // run along variables
				{
					//              shellf[j*num_var+k] = Parallel::global_interp(dim,BP->shape,BP->X,BP->fgfs[varl->data->sgfn],
					//	  		                                    pox,ordn,varl->data->SoA,Symmetry);
					f_global_interp(BP->shape, BP->X[0], BP->X[1], BP->X[2], BP->fgfs[varl->data->sgfn], shellf[k],
													pox[0], pox[1], pox[2], ordn, varl->data->SoA, Symmetry);
					varl = varl->next;
					k++;
				}
			}
		}
		if (Bp == ble)
			break;
		Bp = Bp->next;
	}

	if (notfind && myrank == 0)
	{
		cout << "ERROR: Patch::Interp_Points fails to find point (";
		for (int j = 0; j < dim; j++)
		{
			cout << XX[j];
			if (j < dim - 1)
				cout << ",";
			else
				cout << ")";
		}
		cout << " on Patch (";
		for (int j = 0; j < dim; j++)
		{
			cout << bbox[j] << "+" << lli[j] * getdX(j);
			if (j < dim - 1)
				cout << ",";
			else
				cout << ")--";
		}
		cout << "(";
		for (int j = 0; j < dim; j++)
		{
			cout << bbox[dim + j] << "-" << uui[j] * getdX(j);
			if (j < dim - 1)
				cout << ",";
			else
				cout << ")" << endl;
		}
		cout << "splited domains:" << endl;
		{
			MyList<Block> *Bp = blb;
			while (Bp)
			{
				Block *BP = Bp->data;

				for (int i = 0; i < dim; i++)
				{
					llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? BP->bbox[i] + lli[i] * DH[i] : BP->bbox[i] + ghost_width * DH[i];
					uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? BP->bbox[dim + i] - uui[i] * DH[i] : BP->bbox[dim + i] - ghost_width * DH[i];
				}
				cout << "(";
				for (int j = 0; j < dim; j++)
				{
					cout << llb[j] << ":" << uub[j];
					if (j < dim - 1)
						cout << ",";
					else
						cout << ")" << endl;
				}
				if (Bp == ble)
					break;
				Bp = Bp->next;
			}
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Allreduce(shellf, Shellf, num_var, MPI_DOUBLE, MPI_SUM, Comm_here);

	delete[] shellf;
	delete[] DH;
	delete[] llb;
	delete[] uub;

	return true;
}
// find maximum of abstract value, XX store position for maximum, Shellf store maximum themselvs
void Patch::Find_Maximum(MyList<var> *VarList, double *XX,
												 double *Shellf)
{
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	MyList<var> *varl;
	int num_var = 0;
	varl = VarList;
	while (varl)
	{
		num_var++;
		varl = varl->next;
	}

	double *shellf, *xx;
	shellf = new double[num_var];
	xx = new double[dim * num_var];
	memset(shellf, 0, sizeof(double) * num_var);
	memset(xx, 0, sizeof(double) * dim * num_var);

	double *DH;
	int *llb, *uub;
	DH = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		DH[i] = getdX(i);
	}

	llb = new int[dim];
	uub = new int[dim];

	MyList<Block> *Bp = blb;
	while (Bp) // run along Blocks
	{
		Block *BP = Bp->data;

		if (myrank == BP->rank)
		{

			for (int i = 0; i < dim; i++)
			{
				llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? lli[i] : ghost_width;
				uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? uui[i] : ghost_width;
			}

			varl = VarList;
			int k = 0;
			double tmp, tmpx[dim];
			while (varl) // run along variables
			{
				f_find_maximum(BP->shape, BP->X[0], BP->X[1], BP->X[2], BP->fgfs[varl->data->sgfn], tmp, tmpx, llb, uub);
				if (tmp > shellf[k])
				{
					shellf[k] = tmp;
					for (int i = 0; i < dim; i++)
						xx[dim * k + i] = tmpx[i];
				}
				varl = varl->next;
				k++;
			}
		}

		if (Bp == ble)
			break;
		Bp = Bp->next;
	}

	struct mloc
	{
		double val;
		int rank;
	};

	mloc *IN, *OUT;
	IN = new mloc[num_var];
	OUT = new mloc[num_var];
	for (int i = 0; i < num_var; i++)
	{
		IN[i].val = shellf[i];
		IN[i].rank = myrank;
	}

	MPI_Allreduce(IN, OUT, num_var, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

	for (int i = 0; i < num_var; i++)
	{
		Shellf[i] = OUT[i].val;
		if (myrank != OUT[i].rank)
			for (int k = 0; k < 3; k++)
				xx[3 * i + k] = 0;
	}

	MPI_Allreduce(xx, XX, dim * num_var, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	delete[] IN;
	delete[] OUT;
	delete[] shellf;
	delete[] xx;
	delete[] DH;
	delete[] llb;
	delete[] uub;
}
void Patch::Find_Maximum(MyList<var> *VarList, double *XX,
												 double *Shellf, MPI_Comm Comm_here)
{
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	MyList<var> *varl;
	int num_var = 0;
	varl = VarList;
	while (varl)
	{
		num_var++;
		varl = varl->next;
	}

	double *shellf, *xx;
	shellf = new double[num_var];
	xx = new double[dim * num_var];
	memset(shellf, 0, sizeof(double) * num_var);
	memset(xx, 0, sizeof(double) * dim * num_var);

	double *DH;
	int *llb, *uub;
	DH = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		DH[i] = getdX(i);
	}

	llb = new int[dim];
	uub = new int[dim];

	MyList<Block> *Bp = blb;
	while (Bp) // run along Blocks
	{
		Block *BP = Bp->data;

		if (myrank == BP->rank)
		{

			for (int i = 0; i < dim; i++)
			{
				llb[i] = (feq(BP->bbox[i], bbox[i], DH[i] / 2)) ? lli[i] : ghost_width;
				uub[i] = (feq(BP->bbox[dim + i], bbox[dim + i], DH[i] / 2)) ? uui[i] : ghost_width;
			}

			varl = VarList;
			int k = 0;
			double tmp, tmpx[dim];
			while (varl) // run along variables
			{
				f_find_maximum(BP->shape, BP->X[0], BP->X[1], BP->X[2], BP->fgfs[varl->data->sgfn], tmp, tmpx, llb, uub);
				if (tmp > shellf[k])
				{
					shellf[k] = tmp;
					for (int i = 0; i < dim; i++)
						xx[dim * k + i] = tmpx[i];
				}
				varl = varl->next;
				k++;
			}
		}

		if (Bp == ble)
			break;
		Bp = Bp->next;
	}

	struct mloc
	{
		double val;
		int rank;
	};

	mloc *IN, *OUT;
	IN = new mloc[num_var];
	OUT = new mloc[num_var];
	for (int i = 0; i < num_var; i++)
	{
		IN[i].val = shellf[i];
		IN[i].rank = myrank;
	}

	MPI_Allreduce(IN, OUT, num_var, MPI_DOUBLE_INT, MPI_MAXLOC, Comm_here);

	for (int i = 0; i < num_var; i++)
	{
		Shellf[i] = OUT[i].val;
		if (myrank != OUT[i].rank)
			for (int k = 0; k < 3; k++)
				xx[3 * i + k] = 0;
	}

	MPI_Allreduce(xx, XX, dim * num_var, MPI_DOUBLE, MPI_SUM, Comm_here);

	delete[] IN;
	delete[] OUT;
	delete[] shellf;
	delete[] xx;
	delete[] DH;
	delete[] llb;
	delete[] uub;
}
// if the given point locates in the present Patch return true
// otherwise return false
bool Patch::Find_Point(double *XX)
{
	double *DH;
	DH = new double[dim];

	for (int i = 0; i < dim; i++)
	{
		DH[i] = getdX(i);
	}

	for (int i = 0; i < dim; i++)
	{
		// has excluded the buffer points
		if (XX[i] < bbox[i] + lli[i] * DH[i] - DH[i] / 100 || XX[i] > bbox[dim + i] - uui[i] * DH[i] + DH[i] / 100)
		{
			delete[] DH;
			return false; // out of current patch,
										// remember to delete the allocated arrays before return!!!
		}
	}

	delete[] DH;

	return true;
}
