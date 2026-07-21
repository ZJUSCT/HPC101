
#include "Parallel.h"
#include "fmisc.h"
#include "prolongrestrict.h"
#include "misc.h"
#include "parameters.h"

#ifdef USE_GPU
#include "gpu_manager.h"
#include "helper.h"
#endif

int Parallel::partition1(int &nx, int split_size, int min_width, int cpusize, int shape) // special for 1 diemnsion
{
    nx = Mymax(1, shape / min_width);
    nx = Mymin(cpusize, nx);

    return nx;
}
int Parallel::partition2(int *nxy, int split_size, int *min_width, int cpusize, int *shape) // special for 2 diemnsions
{
#define SEARCH_SIZE 5
    int i, j, nx, ny;
    int maxnx, maxny;
    int mnx, mny;
    int dn, hmin_width, cmin_width;
    int cnx, cny;
    double fx, fy;
    int block_size;
    int n;

    block_size = shape[0] * shape[1];
    n = Mymax(1, (block_size + split_size / 2) / split_size);

    maxnx = Mymax(1, shape[0] / min_width[0]);
    maxnx = Mymin(cpusize, maxnx);
    maxny = Mymax(1, shape[1] / min_width[1]);
    maxny = Mymin(cpusize, maxny);
    fx = (double)shape[0] / (shape[0] + shape[1]);
    fy = (double)shape[1] / (shape[0] + shape[1]);
    nx = mnx = Mymax(1, Mymin(maxnx, (int)(sqrt(double(n)) * fx / fy)));
    ny = mny = Mymax(1, Mymin(maxny, (int)(sqrt(double(n)) * fy / fx)));
    dn = abs(n - nx * ny);
    hmin_width = Mymin(shape[0] / nx, shape[1] / ny);
    for (cny = Mymax(1, mny - SEARCH_SIZE); cny <= (Mymin(mny + SEARCH_SIZE, maxny)); cny++)
        for (cnx = Mymax(1, mnx - SEARCH_SIZE); cnx <= (Mymin(mnx + SEARCH_SIZE, maxnx)); cnx++)
        {
            cmin_width = Mymin(shape[0] / cnx, shape[1] / cny);
            if (dn > abs(n - cnx * cny) || (dn == abs(n - cnx * cny) && cmin_width > hmin_width))
            {
                dn = abs(n - cnx * cny);
                nx = cnx;
                ny = cny;
                hmin_width = cmin_width;
            }
        }

    nxy[0] = nx;
    nxy[1] = ny;

    return nx * ny;
#undef SEARCH_SIZE
}
int Parallel::partition3(int *nxyz, int split_size, int *min_width, int cpusize, int *shape) // special for 3 diemnsions
                                                                                        // algrithsm from Pretorius
{
//	cout<<split_size<<endl<<min_width[0]<<endl<<min_width[1]<<endl<<min_width[2]<<endl
//            <<shape[0]<<endl<<shape[1]<<endl<<shape[2]<<endl<<cpusize<<endl;
#define SEARCH_SIZE 5
    int i, j, k, nx, ny, nz;
    int maxnx, maxny, maxnz;
    int mnx, mny, mnz;
    int dn, hmin_width, cmin_width;
    int cnx, cny, cnz;
    double fx, fy, fz, max_fxfy, max_fxfz, max_fyfz;
    int block_size;
    int n;

    block_size = shape[0] * shape[1] * shape[2];
    n = Mymax(1, (block_size + split_size / 2) / split_size);

    maxnx = Mymax(1, shape[0] / min_width[0]);
    maxnx = Mymin(cpusize, maxnx);
    maxny = Mymax(1, shape[1] / min_width[1]);
    maxny = Mymin(cpusize, maxny);
    maxnz = Mymax(1, shape[2] / min_width[2]);
    maxnz = Mymin(cpusize, maxnz);
    fx = (double)shape[0] / (shape[0] + shape[1] + shape[2]);
    fy = (double)shape[1] / (shape[0] + shape[1] + shape[2]);
    fz = (double)shape[2] / (shape[0] + shape[1] + shape[2]);
    max_fxfy = Mymax(fx, fy);
    max_fxfz = Mymax(fx, fz);
    max_fyfz = Mymax(fy, fz);
    nx = mnx = Mymax(1, Mymin(maxnx, (int)(pow(n, 1.0 / 3.0) * fx / max_fyfz)));
    ny = mny = Mymax(1, Mymin(maxny, (int)(pow(n, 1.0 / 3.0) * fy / max_fxfz)));
    nz = mnz = Mymax(1, Mymin(maxnz, (int)(pow(n, 1.0 / 3.0) * fz / max_fxfy)));
    dn = abs(n - nx * ny * nz);
    hmin_width = Mymin(shape[2] / nz, shape[1] / ny);
    hmin_width = Mymin(hmin_width, shape[0] / nx);
    for (cnz = Mymax(1, mnz - SEARCH_SIZE); cnz <= (Mymin(mnz + SEARCH_SIZE, maxnz)); cnz++)
        for (cny = Mymax(1, mny - SEARCH_SIZE); cny <= (Mymin(mny + SEARCH_SIZE, maxny)); cny++)
            for (cnx = Mymax(1, mnx - SEARCH_SIZE); cnx <= (Mymin(mnx + SEARCH_SIZE, maxnx)); cnx++)
            {
                cmin_width = Mymin(shape[2] / cnz, shape[1] / cny);
                cmin_width = Mymin(cmin_width, shape[0] / cnx);
                if (dn > abs(n - cnx * cny * cnz) || (dn == abs(n - cnx * cny * cnz) && cmin_width > hmin_width))
                {
                    dn = abs(n - cnx * cny * cnz);
                    nx = cnx;
                    ny = cny;
                    nz = cnz;
                    hmin_width = cmin_width;
                }
            }

    nxyz[0] = nx;
    nxyz[1] = ny;
    nxyz[2] = nz;

    return nx * ny * nz;
#undef SEARCH_SIZE
}
// distribute the data to cprocessors
MyList<Block> *Parallel::distribute(
    MyList<Patch> *PatchLIST, int cpusize, int ingfsi, int fngfsi,
    bool periodic, int nodes
) {
    if (nodes == 0)
        nodes = cpusize;

    if (dim != 3)
    {
        cout << "distrivute: now we only support 3-dimension" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MyList<Block> *BlL = 0;

    int split_size, min_size, block_size = 0;

    int min_width = 2 * Mymax(ghost_width, buffer_width);
    int nxyz[dim], mmin_width[dim], min_shape[dim];

    MyList<Patch> *PLi = PatchLIST;
    for (int i = 0; i < dim; i++)
        min_shape[i] = PLi->data->shape[i];
    int lev = PLi->data->lev;
    PLi = PLi->next;
    while (PLi)
    {
        Patch *PP = PLi->data;
        for (int i = 0; i < dim; i++)
            min_shape[i] = Mymin(min_shape[i], PP->shape[i]);
        if (lev != PLi->data->lev)
            cout << "Parallel::distribute CAUSTION: meet Patches for different level: " << lev << " and " << PLi->data->lev << endl;
        PLi = PLi->next;
    }

    for (int i = 0; i < dim; i++)
        mmin_width[i] = Mymin(min_width, min_shape[i]);

    min_size = mmin_width[0];
    for (int i = 1; i < dim; i++)
        min_size = min_size * mmin_width[i];

    PLi = PatchLIST;
    while (PLi)
    {
        Patch *PP = PLi->data;
        //    PP->checkPatch(true);
        int bs = PP->shape[0];
        for (int i = 1; i < dim; i++)
            bs = bs * PP->shape[i];
        block_size = block_size + bs;
        PLi = PLi->next;
    }
    split_size = Mymax(min_size, block_size / nodes);
    split_size = Mymax(1, split_size);

    int n_rank = 0;
    PLi = PatchLIST;
    int reacpu = 0;
    while (PLi)
    {
        Patch *PP = PLi->data;

        reacpu += partition3(nxyz, split_size, mmin_width, nodes, PP->shape);

        Block *ng0, *ng;
        int shape_here[dim], ibbox_here[2 * dim];
        double bbox_here[2 * dim], dd;

        // ibbox : 0,...N-1
        for (int i = 0; i < nxyz[0]; i++)
            for (int j = 0; j < nxyz[1]; j++)
                for (int k = 0; k < nxyz[2]; k++)
                {
                    ibbox_here[0] = (PP->shape[0] * i) / nxyz[0];
                    ibbox_here[3] = (PP->shape[0] * (i + 1)) / nxyz[0] - 1;
                    ibbox_here[1] = (PP->shape[1] * j) / nxyz[1];
                    ibbox_here[4] = (PP->shape[1] * (j + 1)) / nxyz[1] - 1;
                    ibbox_here[2] = (PP->shape[2] * k) / nxyz[2];
                    ibbox_here[5] = (PP->shape[2] * (k + 1)) / nxyz[2] - 1;

                    if (periodic)
                    {
                        ibbox_here[0] = ibbox_here[0] - ghost_width;
                        ibbox_here[3] = ibbox_here[3] + ghost_width;
                        ibbox_here[1] = ibbox_here[1] - ghost_width;
                        ibbox_here[4] = ibbox_here[4] + ghost_width;
                        ibbox_here[2] = ibbox_here[2] - ghost_width;
                        ibbox_here[5] = ibbox_here[5] + ghost_width;
                    }
                    else
                    {
                        ibbox_here[0] = Mymax(0, ibbox_here[0] - ghost_width);
                        ibbox_here[3] = Mymin(PP->shape[0] - 1, ibbox_here[3] + ghost_width);
                        ibbox_here[1] = Mymax(0, ibbox_here[1] - ghost_width);
                        ibbox_here[4] = Mymin(PP->shape[1] - 1, ibbox_here[4] + ghost_width);
                        ibbox_here[2] = Mymax(0, ibbox_here[2] - ghost_width);
                        ibbox_here[5] = Mymin(PP->shape[2] - 1, ibbox_here[5] + ghost_width);
                    }

                    shape_here[0] = ibbox_here[3] - ibbox_here[0] + 1;
                    shape_here[1] = ibbox_here[4] - ibbox_here[1] + 1;
                    shape_here[2] = ibbox_here[5] - ibbox_here[2] + 1;

                    // 0--5, 5--10
                    dd = (PP->bbox[3] - PP->bbox[0]) / PP->shape[0];
                    bbox_here[0] = PP->bbox[0] + (ibbox_here[0]) * dd;
                    bbox_here[3] = PP->bbox[0] + (ibbox_here[3] + 1) * dd;

                    dd = (PP->bbox[4] - PP->bbox[1]) / PP->shape[1];
                    bbox_here[1] = PP->bbox[1] + (ibbox_here[1]) * dd;
                    bbox_here[4] = PP->bbox[1] + (ibbox_here[4] + 1) * dd;

                    dd = (PP->bbox[5] - PP->bbox[2]) / PP->shape[2];
                    bbox_here[2] = PP->bbox[2] + (ibbox_here[2]) * dd;
                    bbox_here[5] = PP->bbox[2] + (ibbox_here[5] + 1) * dd;

                    ng = ng0 = new Block(dim, shape_here, bbox_here, n_rank++, ingfsi, fngfsi, PP->lev); // delete through KillBlocks
                    //	    ng->checkBlock();
                    if (BlL)
                        BlL->insert(ng);
                    else
                        BlL = new MyList<Block>(ng); // delete through KillBlocks

                    if (n_rank == cpusize)
                        n_rank = 0;

                    // set PP->blb
                    if (i == 0 && j == 0 && k == 0)
                    {
                        MyList<Block> *Bp = BlL;
                        while (Bp->data != ng0)
                            Bp = Bp->next; // ng0 is the first of the pices list
                        PP->blb = Bp;
                    }
                }
        // set PP->ble
        {
            MyList<Block> *Bp = BlL;
            while (Bp->data != ng)
                Bp = Bp->next; // ng is the last of the pices list
            PP->ble = Bp;
        }
        PLi = PLi->next;
    }
    if (reacpu < nodes * 2 / 3)
    {
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        if (myrank == 0)
            cout << "Parallel::distribute CAUSTION: level#" << lev << " uses essencially " << reacpu << " processors vs " << nodes << " nodes run, your scientific computation scale is not as large as you estimate." << endl;
    }

    return BlL;
}

void Parallel::setfunction(MyList<Block> *BlL, var *vn, double func(double x, double y, double z))
{
    while (BlL)
    {
        if (BlL->data->X[0])
        {
            int nn = BlL->data->shape[0] * BlL->data->shape[1] * BlL->data->shape[2];
            double *p = BlL->data->fgfs[vn->sgfn];
            for (int i = 0; i < nn; i++)
            {
                int ind[3];
                getarrayindex(3, BlL->data->shape, ind, i);
                p[i] = func(BlL->data->X[0][ind[0]], BlL->data->X[1][ind[1]], BlL->data->X[2][ind[2]]);
            }
        }
        BlL = BlL->next;
    }
}
// set function only for cpu rank
void Parallel::setfunction(int rank, MyList<Block> *BlL, var *vn, double func(double x, double y, double z))
{
    while (BlL)
    {
        if (BlL->data->X[0] && BlL->data->rank == rank)
        {
            int nn = BlL->data->shape[0] * BlL->data->shape[1] * BlL->data->shape[2];
            double *p = BlL->data->fgfs[vn->sgfn];
            for (int i = 0; i < nn; i++)
            {
                int ind[3];
                getarrayindex(3, BlL->data->shape, ind, i);
                p[i] = func(BlL->data->X[0][ind[0]], BlL->data->X[1][ind[1]], BlL->data->X[2][ind[2]]);
            }
        }
        BlL = BlL->next;
    }
}
void Parallel::getarrayindex(int DIM, int *shape, int *index, int n)
{
    // we assume index has already memory space
    int *mu;
    mu = new int[DIM];
    mu[0] = 1;
    for (int i = 1; i < DIM; i++)
        mu[i] = mu[i - 1] * shape[i - 1];
    for (int i = DIM - 1; i >= 0; i--)
    {
        index[i] = n / mu[i];
        n = n - index[i] * mu[i];
    }

    delete[] mu;
}
int Parallel::getarraylocation(int DIM, int *shape, int *index)
{
    int n, mu;
    mu = shape[0];
    n = index[0];
    for (int i = 1; i < DIM; i++)
    {
        n = n + index[i] * mu;
        mu = mu * shape[i];
    }

    return n;
}
void Parallel::copy(int DIM, double *llbout, double *uubout, int *Dshape, double *DD, double *llbin, double *uubin,
                                        int *shape, double *datain, double *llb, double *uub)
{
    // for 3 dimensional case, based on simple test, I found this is half slower than f90 code
    int *illi, *iuui;
    int *illo, *iuuo;
    int *indi, *indo;
    illi = new int[DIM];
    iuui = new int[DIM];
    illo = new int[DIM];
    iuuo = new int[DIM];
    indi = new int[DIM];
    indo = new int[DIM];

    int ial = 1;
    for (int i = 0; i < DIM; i++)
    {
        double ho, hi;
        ho = (uubout[i] - llbout[i]) / Dshape[i];
        hi = (uubin[i] - llbin[i]) / shape[i];
        illo[i] = int((llb[i] - llbout[i]) / ho);
        iuuo[i] = Dshape[i] - 1 - int((uubout[i] - uub[i]) / ho);
        illi[i] = int((llb[i] - llbin[i]) / hi);
        iuui[i] = shape[i] - 1 - int((uubin[i] - uub[i]) / hi);

        if (illo[i] > iuuo[i] || illi[i] > iuui[i] || illo[i] < 0 || illi[i] < 0 ||
                iuui[i] >= shape[i] || iuuo[i] >= Dshape[i])
        {
            cout << "Parallel copy: in direction " << i << ":" << endl;
            cout << "llb = " << llb[i] << ", uub = " << uub[i] << endl;
            cout << " in data : il = " << illi[i] << ", iu = " << iuui[i] << endl;
            cout << "bbox = (" << llbin[i] << "," << uubin[i] << ")" << endl;
            cout << "shape = " << shape[i] << endl;
            cout << "out data : il = " << illo[i] << ", iu = " << iuuo[i] << endl;
            cout << "bbox = (" << llbout[i] << "," << uubout[i] << ")" << endl;
            cout << "shape = " << Dshape[i] << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int ihi = iuui[i] - illi[i] + 1, iho = iuuo[i] - illo[i] + 1;
        if (!(feq(ho, hi, ho / 2)) || ihi != iho)
        {
            cout << "Parallel copy: in direction " << i << ":" << endl;
            cout << "Parallel copy: not the same grid structure." << endl;
            cout << "hi = " << hi << ", bbox = (" << llbin[i] << "," << uubin[i] << "), shape = " << shape[i] << endl;
            cout << "ho = " << ho << ", bbox = (" << llbout[i] << "," << uubout[i] << "), shape = " << Dshape[i] << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ial = ial * ihi;
    }

    for (int i = 0; i < DIM; i++)
    {
        indi[i] = illi[i];
        indo[i] = illo[i];
    }
    /*
    //check start index
         for(int i=0;i<DIM;i++)
         {
             cout << "Parallel copy: in direction " <<i<<":"<< endl;
             cout<<"start : indi = "<<indi[i]<<", indo = "<<indo[i]<<endl;
         }
    */
    int NNi = 1, NNo = 1;
    for (int i = 0; i < DIM; i++)
    {
        NNi = NNi * shape[i];
        NNo = NNo * Dshape[i];
    }
    for (int i = 0; i < ial; i++)
    {
        int ni, no;
        ni = getarraylocation(DIM, shape, indi);
        no = getarraylocation(DIM, Dshape, indo);
        if (no < 0 || no > NNo)
        {
            cout << "Parallel copy: no = " << no << " is out of array range (0," << NNo << ")." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (ni < 0 || ni > NNi)
        {
            cout << "Parallel copy: ni = " << ni << " is out of array range (0," << NNi << ")." << endl;
            cout << "shape = (";
            for (int j = 0; j < DIM; j++)
            {
                cout << shape[j];
                if (j < DIM - 1)
                    cout << ",";
                else
                    cout << ")" << endl;
            }
            cout << "ind = (";
            for (int j = 0; j < DIM; j++)
            {
                cout << indi[j];
                if (j < DIM - 1)
                    cout << ",";
                else
                    cout << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        DD[no] = datain[ni];

        indi[0]++;
        for (int j = 1; j < DIM; j++)
        {
            if (indi[j - 1] == iuui[j - 1] + 1)
            {
                indi[j - 1] = illi[j - 1];
                indi[j]++;
            } // carry 1 to next digital
            else
                break;
        }
        indo[0]++;
        for (int j = 1; j < DIM; j++)
        {
            if (indo[j - 1] == iuuo[j - 1] + 1)
            {
                indo[j - 1] = illo[j - 1];
                indo[j]++;
            }
            else
                break;
        }
    }
    delete[] illi;
    delete[] iuui;
    delete[] illo;
    delete[] iuuo;
    delete[] indi;
    delete[] indo;
}
void Parallel::writefile(double time, int nx, int ny, int nz, double xmin, double xmax, double ymin, double ymax,
                                                 double zmin, double zmax, char *filename, double *data_out)
{
    ofstream outfile;
    outfile.open(filename, ios::out | ios::trunc);
    if (!outfile)
    {
        cout << "Can't open " << filename << " for output." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    outfile.write((char *)&time, sizeof(double));
    outfile.write((char *)&nx, sizeof(int));
    outfile.write((char *)&ny, sizeof(int));
    outfile.write((char *)&nz, sizeof(int));
    outfile.write((char *)&xmin, sizeof(double));
    outfile.write((char *)&xmax, sizeof(double));
    outfile.write((char *)&ymin, sizeof(double));
    outfile.write((char *)&ymax, sizeof(double));
    outfile.write((char *)&zmin, sizeof(double));
    outfile.write((char *)&zmax, sizeof(double));
    outfile.write((char *)data_out, nx * ny * nz * sizeof(double));
    outfile.close();
}
void Parallel::writefile(double time, int nx, int ny, double xmin, double xmax, double ymin, double ymax,
                                                 char *filename, double *datain)
{
    int i, j;
    double *X, *Y;
    X = new double[nx];
    Y = new double[ny];
    double dd;
    dd = (xmax - xmin) / nx;
    for (i = 0; i < nx; i++)
        X[i] = xmin + (i + 0.5) * dd;
    dd = (ymax - ymin) / ny;
    for (j = 0; j < ny; j++)
        Y[j] = ymin + (j + 0.5) * dd;
    ofstream outfile;
    outfile.open(filename, ios::out | ios::trunc);
    if (!outfile)
    {
        cout << "Can't open " << filename << " for output." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    outfile << "# t = " << time << endl;
    for (j = 0; j < ny; j++)
    {
        for (i = 0; i < nx; i++)
        {
            int ind1 = i + j * nx;
            outfile << setw(10) << setprecision(10) << X[i] << " "
                            << setw(10) << setprecision(10) << Y[j] << " "
                            << setw(16) << setprecision(15) << datain[ind1]
                            << endl;
        }
        outfile << "\n"; /* blanck line for gnuplot */
    }
    outfile.close();

    delete[] X;
    delete[] Y;
}
void Parallel::Dump_CPU_Data(MyList<Block> *BlL, MyList<var> *DumpList, char *tag, double time, double dT)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // round at 4 and 5
    int ncount = int(time / dT + 0.5);

    MyList<Block> *Bp;
    while (DumpList)
    {
        Bp = BlL;
        int Bi = 0;
        while (Bp)
        {
            Block *BP = Bp->data;
            var *VP = DumpList->data;
            if (BP->rank == myrank)
            {

                string out_dir;
                map<string, string>::iterator iter;
                iter = parameters::str_par.find("output dir");
                if (iter != parameters::str_par.end())
                {
                    out_dir = iter->second;
                }
                else
                {
                    // read parameter from file
                    const int LEN = 256;
                    char pline[LEN];
                    string str, sgrp, skey, sval;
                    int sind;
                    char pname[50];
                    {
                        map<string, string>::iterator iter = parameters::str_par.find("inputpar");
                        if (iter != parameters::str_par.end())
                        {
                            strcpy(pname, (iter->second).c_str());
                        }
                        else
                        {
                            cout << "Error inputpar" << endl;
                            exit(0);
                        }
                    }
                    ifstream inf(pname, ifstream::in);
                    if (!inf.good())
                    {
                        cout << "Can not open parameter file " << pname << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                    for (int i = 1; inf.good(); i++)
                    {
                        inf.getline(pline, LEN);
                        str = pline;

                        int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                        if (status == -1)
                        {
                            cout << "error reading parameter file " << pname << " in line " << i << endl;
                            MPI_Abort(MPI_COMM_WORLD, 1);
                        }
                        else if (status == 0)
                            continue;

                        if (sgrp == "ABE")
                        {
                            if (skey == "output dir")
                                out_dir = sval;
                        }
                    }
                    inf.close();

                    parameters::str_par.insert(map<string, string>::value_type("output dir", out_dir));
                }

                char filename[100];
                if (tag)
                    sprintf(filename, "%s/%s_Lev%02d-%02d_%02d_%s_%05d.bin", out_dir.c_str(), tag, BP->lev, Bi, myrank, VP->name, ncount);
                else
                    sprintf(filename, "%s/Lev%02d-%02d_%02d_%s_%05d.bin", out_dir.c_str(), BP->lev, Bi, myrank, VP->name, ncount);
                writefile(time, BP->shape[0], BP->shape[1], BP->shape[2], BP->bbox[0], BP->bbox[3], BP->bbox[1], BP->bbox[4],
                                    BP->bbox[2], BP->bbox[5], filename, BP->fgfs[VP->sgfn]);
                cout << "end of dump " << VP->name << " at time " << time << ", on node " << myrank << endl;
            }
            Bp = Bp->next;
            Bi++;
        }
        DumpList = DumpList->next;
    }
}
// Now we dump the data including buffer points
void Parallel::Dump_Data(Patch *PP, MyList<var> *DumpList, char *tag, double time, double dT, int grd)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    //   round at 4 and 5
    int ncount = int(time / dT + 0.5);

    MPI_Status sta;
    int DIM = 3;
    double llb[3], uub[3];
    double DX, DY, DZ;

    double *databuffer = 0;
    if (myrank == 0)
    {
        databuffer = (double *)malloc(sizeof(double) * PP->shape[0] * PP->shape[1] * PP->shape[2]);
        if (!databuffer)
        {
            cout << "Parallel::Dump_Data: out of memory when dumping data." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    while (DumpList)
    {
        var *VP = DumpList->data;

        MyList<Block> *Bp = PP->blb;
        while (Bp)
        {
            Block *BP = Bp->data;
            if (BP->rank == 0 && myrank == 0)
            {
                DX = BP->getdX(0);
                DY = BP->getdX(1);
                DZ = BP->getdX(2);
                llb[0] = (feq(BP->bbox[0], PP->bbox[0], DX / 2)) ? BP->bbox[0] : BP->bbox[0] + ghost_width * DX;
                llb[1] = (feq(BP->bbox[1], PP->bbox[1], DY / 2)) ? BP->bbox[1] : BP->bbox[1] + ghost_width * DY;
                llb[2] = (feq(BP->bbox[2], PP->bbox[2], DZ / 2)) ? BP->bbox[2] : BP->bbox[2] + ghost_width * DZ;
                uub[0] = (feq(BP->bbox[3], PP->bbox[3], DX / 2)) ? BP->bbox[3] : BP->bbox[3] - ghost_width * DX;
                uub[1] = (feq(BP->bbox[4], PP->bbox[4], DY / 2)) ? BP->bbox[4] : BP->bbox[4] - ghost_width * DY;
                uub[2] = (feq(BP->bbox[5], PP->bbox[5], DZ / 2)) ? BP->bbox[5] : BP->bbox[5] - ghost_width * DZ;
                f_copy(DIM, PP->bbox, PP->bbox + DIM, PP->shape, databuffer, BP->bbox, BP->bbox + DIM, BP->shape, BP->fgfs[VP->sgfn], llb, uub);
            }
            else
            {
                int nnn = (BP->shape[0]) * (BP->shape[1]) * (BP->shape[2]);
                if (myrank == 0)
                {
                    double *bufferhere = (double *)malloc(sizeof(double) * nnn);
                    if (!bufferhere)
                    {
                        cout << "on node#" << myrank << ", out of memory when dumping data." << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    MPI_Recv(bufferhere, nnn, MPI_DOUBLE, BP->rank, 0, MPI_COMM_WORLD, &sta);
                    DX = BP->getdX(0);
                    DY = BP->getdX(1);
                    DZ = BP->getdX(2);
                    llb[0] = (feq(BP->bbox[0], PP->bbox[0], DX / 2)) ? BP->bbox[0] : BP->bbox[0] + ghost_width * DX;
                    llb[1] = (feq(BP->bbox[1], PP->bbox[1], DY / 2)) ? BP->bbox[1] : BP->bbox[1] + ghost_width * DY;
                    llb[2] = (feq(BP->bbox[2], PP->bbox[2], DZ / 2)) ? BP->bbox[2] : BP->bbox[2] + ghost_width * DZ;
                    uub[0] = (feq(BP->bbox[3], PP->bbox[3], DX / 2)) ? BP->bbox[3] : BP->bbox[3] - ghost_width * DX;
                    uub[1] = (feq(BP->bbox[4], PP->bbox[4], DY / 2)) ? BP->bbox[4] : BP->bbox[4] - ghost_width * DY;
                    uub[2] = (feq(BP->bbox[5], PP->bbox[5], DZ / 2)) ? BP->bbox[5] : BP->bbox[5] - ghost_width * DZ;
                    f_copy(DIM, PP->bbox, PP->bbox + DIM, PP->shape, databuffer, BP->bbox, BP->bbox + DIM, BP->shape, bufferhere, llb, uub);
                    free(bufferhere);
                }
                else if (myrank == BP->rank)
                {
                    MPI_Send(BP->fgfs[VP->sgfn], nnn, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
            if (Bp == PP->ble)
                break;
            Bp = Bp->next;
        }
        if (myrank == 0)
        {

            string out_dir;
            map<string, string>::iterator iter;
            iter = parameters::str_par.find("output dir");
            if (iter != parameters::str_par.end())
            {
                out_dir = iter->second;
            }
            else
            {
                // read parameter from file
                const int LEN = 256;
                char pline[LEN];
                string str, sgrp, skey, sval;
                int sind;
                char pname[50];
                {
                    map<string, string>::iterator iter = parameters::str_par.find("inputpar");
                    if (iter != parameters::str_par.end())
                    {
                        strcpy(pname, (iter->second).c_str());
                    }
                    else
                    {
                        cout << "Error inputpar" << endl;
                        exit(0);
                    }
                }
                ifstream inf(pname, ifstream::in);
                if (!inf.good())
                {
                    cout << "Can not open parameter file " << pname << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                for (int i = 1; inf.good(); i++)
                {
                    inf.getline(pline, LEN);
                    str = pline;

                    int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                    if (status == -1)
                    {
                        cout << "error reading parameter file " << pname << " in line " << i << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    else if (status == 0)
                        continue;

                    if (sgrp == "ABE")
                    {
                        if (skey == "output dir")
                            out_dir = sval;
                    }
                }
                inf.close();

                parameters::str_par.insert(map<string, string>::value_type("output dir", out_dir));
            }

            char filename[100];
            if (tag)
                sprintf(filename, "%s/%s_Lev%02d-%02d_%s_%05d.bin", out_dir.c_str(), tag, PP->lev, grd, VP->name, ncount);
            else
                sprintf(filename, "%s/Lev%02d-%02d_%s_%05d.bin", out_dir.c_str(), PP->lev, grd, VP->name, ncount);

            writefile(time, PP->shape[0], PP->shape[1], PP->shape[2], PP->bbox[0], PP->bbox[3], PP->bbox[1], PP->bbox[4],
                                PP->bbox[2], PP->bbox[5], filename, databuffer);
        }
        DumpList = DumpList->next;
    }

    if (myrank == 0)
        free(databuffer);
}
void Parallel::Dump_Data(MyList<Patch> *PL, MyList<var> *DumpList, char *tag, double time, double dT)
{
    MyList<Patch> *Pp;
    Pp = PL;
    int grd = 0;
    while (Pp)
    {
        Patch *PP = Pp->data;
        Dump_Data(PP, DumpList, tag, time, dT, grd);
        grd++;
        Pp = Pp->next;
    }
}
// collect the data including buffer points
double *Parallel::Collect_Data(Patch *PP, var *VP)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MPI_Status sta;
    int DIM = 3;
    double llb[3], uub[3];
    double DX, DY, DZ;

    double *databuffer = 0;
    if (myrank == 0)
    {
        databuffer = (double *)malloc(sizeof(double) * PP->shape[0] * PP->shape[1] * PP->shape[2]);
        if (!databuffer)
        {
            cout << "Parallel::Collect_Data: out of memory when dumping data." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MyList<Block> *Bp = PP->blb;
    while (Bp)
    {
        Block *BP = Bp->data;
        if (BP->rank == 0 && myrank == 0)
        {
            DX = BP->getdX(0);
            DY = BP->getdX(1);
            DZ = BP->getdX(2);
            llb[0] = (feq(BP->bbox[0], PP->bbox[0], DX / 2)) ? BP->bbox[0] : BP->bbox[0] + ghost_width * DX;
            llb[1] = (feq(BP->bbox[1], PP->bbox[1], DY / 2)) ? BP->bbox[1] : BP->bbox[1] + ghost_width * DY;
            llb[2] = (feq(BP->bbox[2], PP->bbox[2], DZ / 2)) ? BP->bbox[2] : BP->bbox[2] + ghost_width * DZ;
            uub[0] = (feq(BP->bbox[3], PP->bbox[3], DX / 2)) ? BP->bbox[3] : BP->bbox[3] - ghost_width * DX;
            uub[1] = (feq(BP->bbox[4], PP->bbox[4], DY / 2)) ? BP->bbox[4] : BP->bbox[4] - ghost_width * DY;
            uub[2] = (feq(BP->bbox[5], PP->bbox[5], DZ / 2)) ? BP->bbox[5] : BP->bbox[5] - ghost_width * DZ;
            f_copy(DIM, PP->bbox, PP->bbox + DIM, PP->shape, databuffer, BP->bbox, BP->bbox + DIM, BP->shape, BP->fgfs[VP->sgfn], llb, uub);
        }
        else
        {
            int nnn = (BP->shape[0]) * (BP->shape[1]) * (BP->shape[2]);
            if (myrank == 0)
            {
                double *bufferhere = (double *)malloc(sizeof(double) * nnn);
                if (!bufferhere)
                {
                    cout << "on node#" << myrank << ", out of memory when dumping data." << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                MPI_Recv(bufferhere, nnn, MPI_DOUBLE, BP->rank, 0, MPI_COMM_WORLD, &sta);
                DX = BP->getdX(0);
                DY = BP->getdX(1);
                DZ = BP->getdX(2);
                llb[0] = (feq(BP->bbox[0], PP->bbox[0], DX / 2)) ? BP->bbox[0] : BP->bbox[0] + ghost_width * DX;
                llb[1] = (feq(BP->bbox[1], PP->bbox[1], DY / 2)) ? BP->bbox[1] : BP->bbox[1] + ghost_width * DY;
                llb[2] = (feq(BP->bbox[2], PP->bbox[2], DZ / 2)) ? BP->bbox[2] : BP->bbox[2] + ghost_width * DZ;
                uub[0] = (feq(BP->bbox[3], PP->bbox[3], DX / 2)) ? BP->bbox[3] : BP->bbox[3] - ghost_width * DX;
                uub[1] = (feq(BP->bbox[4], PP->bbox[4], DY / 2)) ? BP->bbox[4] : BP->bbox[4] - ghost_width * DY;
                uub[2] = (feq(BP->bbox[5], PP->bbox[5], DZ / 2)) ? BP->bbox[5] : BP->bbox[5] - ghost_width * DZ;
                f_copy(DIM, PP->bbox, PP->bbox + DIM, PP->shape, databuffer, BP->bbox, BP->bbox + DIM, BP->shape, bufferhere, llb, uub);
                free(bufferhere);
            }
            else if (myrank == BP->rank)
            {
                MPI_Send(BP->fgfs[VP->sgfn], nnn, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
        if (Bp == PP->ble)
            break;
        Bp = Bp->next;
    }

    return databuffer;
}
// Now we dump the data including buffer points
// dump z = 0 plane
void Parallel::d2Dump_Data(Patch *PP, MyList<var> *DumpList, char *tag, double time, double dT, int grd)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    //   round at 4 and 5
    int ncount = int(time / dT + 0.5);

    MPI_Status sta;
    int DIM = 3;
    double llb[3], uub[3];
    double DX, DY, DZ;

    double *databuffer = 0, *databuffer2 = 0;
    if (myrank == 0)
    {
        databuffer = (double *)malloc(sizeof(double) * PP->shape[0] * PP->shape[1] * PP->shape[2]);
        databuffer2 = (double *)malloc(sizeof(double) * PP->shape[0] * PP->shape[1]);
        if (!databuffer || !databuffer2)
        {
            cout << "Parallel::d2Dump_Data: out of memory when dumping data." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    while (DumpList)
    {
        var *VP = DumpList->data;

        MyList<Block> *Bp = PP->blb;
        while (Bp)
        {
            Block *BP = Bp->data;
            if (BP->rank == 0 && myrank == 0)
            {
                DX = BP->getdX(0);
                DY = BP->getdX(1);
                DZ = BP->getdX(2);
                llb[0] = (feq(BP->bbox[0], PP->bbox[0], DX / 2)) ? BP->bbox[0] : BP->bbox[0] + ghost_width * DX;
                llb[1] = (feq(BP->bbox[1], PP->bbox[1], DY / 2)) ? BP->bbox[1] : BP->bbox[1] + ghost_width * DY;
                llb[2] = (feq(BP->bbox[2], PP->bbox[2], DZ / 2)) ? BP->bbox[2] : BP->bbox[2] + ghost_width * DZ;
                uub[0] = (feq(BP->bbox[3], PP->bbox[3], DX / 2)) ? BP->bbox[3] : BP->bbox[3] - ghost_width * DX;
                uub[1] = (feq(BP->bbox[4], PP->bbox[4], DY / 2)) ? BP->bbox[4] : BP->bbox[4] - ghost_width * DY;
                uub[2] = (feq(BP->bbox[5], PP->bbox[5], DZ / 2)) ? BP->bbox[5] : BP->bbox[5] - ghost_width * DZ;
                f_copy(DIM, PP->bbox, PP->bbox + DIM, PP->shape, databuffer, BP->bbox, BP->bbox + DIM, BP->shape, BP->fgfs[VP->sgfn], llb, uub);
            }
            else
            {
                int nnn = (BP->shape[0]) * (BP->shape[1]) * (BP->shape[2]);
                if (myrank == 0)
                {
                    double *bufferhere = (double *)malloc(sizeof(double) * nnn);
                    if (!bufferhere)
                    {
                        cout << "on node#" << myrank << ", out of memory when dumping data." << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    MPI_Recv(bufferhere, nnn, MPI_DOUBLE, BP->rank, 0, MPI_COMM_WORLD, &sta);
                    DX = BP->getdX(0);
                    DY = BP->getdX(1);
                    DZ = BP->getdX(2);
                    llb[0] = (feq(BP->bbox[0], PP->bbox[0], DX / 2)) ? BP->bbox[0] : BP->bbox[0] + ghost_width * DX;
                    llb[1] = (feq(BP->bbox[1], PP->bbox[1], DY / 2)) ? BP->bbox[1] : BP->bbox[1] + ghost_width * DY;
                    llb[2] = (feq(BP->bbox[2], PP->bbox[2], DZ / 2)) ? BP->bbox[2] : BP->bbox[2] + ghost_width * DZ;
                    uub[0] = (feq(BP->bbox[3], PP->bbox[3], DX / 2)) ? BP->bbox[3] : BP->bbox[3] - ghost_width * DX;
                    uub[1] = (feq(BP->bbox[4], PP->bbox[4], DY / 2)) ? BP->bbox[4] : BP->bbox[4] - ghost_width * DY;
                    uub[2] = (feq(BP->bbox[5], PP->bbox[5], DZ / 2)) ? BP->bbox[5] : BP->bbox[5] - ghost_width * DZ;
                    f_copy(DIM, PP->bbox, PP->bbox + DIM, PP->shape, databuffer, BP->bbox, BP->bbox + DIM, BP->shape, bufferhere, llb, uub);
                    free(bufferhere);
                }
                else if (myrank == BP->rank)
                {
                    MPI_Send(BP->fgfs[VP->sgfn], nnn, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
            if (Bp == PP->ble)
                break;
            Bp = Bp->next;
        }
        if (myrank == 0)
        {

            string out_dir;
            map<string, string>::iterator iter;
            iter = parameters::str_par.find("output dir");
            if (iter != parameters::str_par.end())
            {
                out_dir = iter->second;
            }
            else
            {
                // read parameter from file
                const int LEN = 256;
                char pline[LEN];
                string str, sgrp, skey, sval;
                int sind;
                char pname[50];
                {
                    map<string, string>::iterator iter = parameters::str_par.find("inputpar");
                    if (iter != parameters::str_par.end())
                    {
                        strcpy(pname, (iter->second).c_str());
                    }
                    else
                    {
                        cout << "Error inputpar" << endl;
                        exit(0);
                    }
                }
                ifstream inf(pname, ifstream::in);
                if (!inf.good())
                {
                    cout << "Can not open parameter file " << pname << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                for (int i = 1; inf.good(); i++)
                {
                    inf.getline(pline, LEN);
                    str = pline;

                    int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                    if (status == -1)
                    {
                        cout << "error reading parameter file " << pname << " in line " << i << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    else if (status == 0)
                        continue;

                    if (sgrp == "ABE")
                    {
                        if (skey == "output dir")
                            out_dir = sval;
                    }
                }
                inf.close();

                parameters::str_par.insert(map<string, string>::value_type("output dir", out_dir));
            }

            char filename[100];
            if (tag)
                sprintf(filename, "%s/%s_2d_Lev%02d-%02d_%s_%05d.dat", out_dir.c_str(), tag, PP->lev, grd, VP->name, ncount);
            else
                sprintf(filename, "%s/2d_Lev%02d-%02d_%s_%05d.dat", out_dir.c_str(), PP->lev, grd, VP->name, ncount);

            int gord = ghost_width;
            f_d2dump(DIM, PP->bbox, PP->bbox + DIM, PP->shape, databuffer, databuffer2, gord, VP->SoA);
            writefile(time, PP->shape[0], PP->shape[1], PP->bbox[0], PP->bbox[3], PP->bbox[1], PP->bbox[4],
                                filename, databuffer2);
        }
        DumpList = DumpList->next;
    }

    if (myrank == 0)
    {
        free(databuffer);
        free(databuffer2);
    }
}
void Parallel::d2Dump_Data(MyList<Patch> *PL, MyList<var> *DumpList, char *tag, double time, double dT)
{
    MyList<Patch> *Pp;
    Pp = PL;
    int grd = 0;
    while (Pp)
    {
        Patch *PP = Pp->data;
        d2Dump_Data(PP, DumpList, tag, time, dT, grd);
        grd++;
        Pp = Pp->next;
    }
}
// Now we dump the data including buffer points and ghost points of the given patch
void Parallel::Dump_Data0(Patch *PP, MyList<var> *DumpList, char *tag, double time, double dT)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    //   round at 4 and 5
    int ncount = int(time / dT + 0.5);

    MPI_Status sta;
    int DIM = 3;
    double llb[3], uub[3], tllb[3], tuub[3];
    int tshape[3];
    double DX, DY, DZ;

    for (int i = 0; i < 3; i++)
    {
        double DX = PP->blb->data->getdX(i);
        tshape[i] = PP->shape[i] + 2 * ghost_width;
        tllb[i] = PP->bbox[i] - ghost_width * DX;
        tuub[i] = PP->bbox[i + dim] + ghost_width * DX;
    }

    int NN = tshape[0] * tshape[1] * tshape[2];
    double *databuffer = 0;
    if (myrank == 0)
    {
        databuffer = (double *)malloc(sizeof(double) * NN);
        if (!databuffer)
        {
            cout << "on node# " << myrank << ", out of memory when dumping data." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    while (DumpList)
    {
        var *VP = DumpList->data;
        MyList<Block> *Bp = PP->blb;
        while (Bp)
        {
            Block *BP = Bp->data;
            if (BP->rank == 0 && myrank == 0)
            {
                DX = BP->getdX(0);
                DY = BP->getdX(1);
                DZ = BP->getdX(2);
                llb[0] = (feq(BP->bbox[0], tllb[0], DX / 2)) ? BP->bbox[0] : BP->bbox[0] + ghost_width * DX;
                llb[1] = (feq(BP->bbox[1], tllb[1], DY / 2)) ? BP->bbox[1] : BP->bbox[1] + ghost_width * DY;
                llb[2] = (feq(BP->bbox[2], tllb[2], DZ / 2)) ? BP->bbox[2] : BP->bbox[2] + ghost_width * DZ;
                uub[0] = (feq(BP->bbox[3], tuub[0], DX / 2)) ? BP->bbox[3] : BP->bbox[3] - ghost_width * DX;
                uub[1] = (feq(BP->bbox[4], tuub[1], DY / 2)) ? BP->bbox[4] : BP->bbox[4] - ghost_width * DY;
                uub[2] = (feq(BP->bbox[5], tuub[2], DZ / 2)) ? BP->bbox[5] : BP->bbox[5] - ghost_width * DZ;
                f_copy(DIM, tllb, tuub, tshape, databuffer, BP->bbox, BP->bbox + DIM, BP->shape, BP->fgfs[VP->sgfn], llb, uub);
            }
            else
            {
                if (myrank == 0)
                {
                    int nnn = (BP->shape[0]) * (BP->shape[1]) * (BP->shape[2]);
                    double *bufferhere = (double *)malloc(sizeof(double) * nnn);
                    if (!bufferhere)
                    {
                        cout << "on node#" << myrank << ", out of memory when dumping data." << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    MPI_Recv(bufferhere, nnn, MPI_DOUBLE, BP->rank, 0, MPI_COMM_WORLD, &sta);
                    DX = BP->getdX(0);
                    DY = BP->getdX(1);
                    DZ = BP->getdX(2);
                    llb[0] = (feq(BP->bbox[0], tllb[0], DX / 2)) ? BP->bbox[0] : BP->bbox[0] + ghost_width * DX;
                    llb[1] = (feq(BP->bbox[1], tllb[1], DY / 2)) ? BP->bbox[1] : BP->bbox[1] + ghost_width * DY;
                    llb[2] = (feq(BP->bbox[2], tllb[2], DZ / 2)) ? BP->bbox[2] : BP->bbox[2] + ghost_width * DZ;
                    uub[0] = (feq(BP->bbox[3], tuub[0], DX / 2)) ? BP->bbox[3] : BP->bbox[3] - ghost_width * DX;
                    uub[1] = (feq(BP->bbox[4], tuub[1], DY / 2)) ? BP->bbox[4] : BP->bbox[4] - ghost_width * DY;
                    uub[2] = (feq(BP->bbox[5], tuub[2], DZ / 2)) ? BP->bbox[5] : BP->bbox[5] - ghost_width * DZ;
                    f_copy(DIM, tllb, tuub, tshape, databuffer, BP->bbox, BP->bbox + DIM, BP->shape, bufferhere, llb, uub);
                    free(bufferhere);
                }
                else if (myrank == BP->rank)
                {
                    int nnn = (BP->shape[0]) * (BP->shape[1]) * (BP->shape[2]);
                    MPI_Send(BP->fgfs[VP->sgfn], nnn, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
            if (Bp == PP->ble)
                break;
            Bp = Bp->next;
        }
        if (myrank == 0)
        {

            string out_dir;
            map<string, string>::iterator iter;
            iter = parameters::str_par.find("output dir");
            if (iter != parameters::str_par.end())
            {
                out_dir = iter->second;
            }
            else
            {
                // read parameter from file
                const int LEN = 256;
                char pline[LEN];
                string str, sgrp, skey, sval;
                int sind;
                char pname[50];
                {
                    map<string, string>::iterator iter = parameters::str_par.find("inputpar");
                    if (iter != parameters::str_par.end())
                    {
                        strcpy(pname, (iter->second).c_str());
                    }
                    else
                    {
                        cout << "Error inputpar" << endl;
                        exit(0);
                    }
                }
                ifstream inf(pname, ifstream::in);
                if (!inf.good())
                {
                    cout << "Can not open parameter file " << pname << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                for (int i = 1; inf.good(); i++)
                {
                    inf.getline(pline, LEN);
                    str = pline;

                    int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                    if (status == -1)
                    {
                        cout << "error reading parameter file " << pname << " in line " << i << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    else if (status == 0)
                        continue;

                    if (sgrp == "ABE")
                    {
                        if (skey == "output dir")
                            out_dir = sval;
                    }
                }
                inf.close();

                parameters::str_par.insert(map<string, string>::value_type("output dir", out_dir));
            }

            char filename[100];
            if (tag)
                sprintf(filename, "%s/%s_Lev%02d_%s_%05d.bin", out_dir.c_str(), tag, PP->lev, VP->name, ncount);
            else
                sprintf(filename, "%s/Lev%02d_%s_%05d.bin", out_dir.c_str(), PP->lev, VP->name, ncount);

            writefile(time, tshape[0], tshape[1], tshape[2], tllb[0], tuub[0], tllb[1], tuub[2],
                                tllb[2], tuub[2], filename, databuffer);
        }
        DumpList = DumpList->next;
    }

    if (myrank == 0)
        free(databuffer);
}
// Map point is much easier than maping data itself
// But the main problem is about the points near the boundary
// worst case is -ghost -ghost+1 .... 0 * ......
double Parallel::global_interp(int DIM, int *ext, double **CoX, double *datain,
                                                             double *poXb, int ordn, double *SoA, int Symmetry)
{
    if (DIM != 3)
    {
        cout << "Parallel::global_interp does not suport DIM = " << DIM << " for Symmetry." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double resu;
    double poX[3];
    double asgn = 1;

    for (int i = 0; i < 3; i++)
        poX[i] = poXb[i];

    switch (Symmetry)
    {
    case 2:
        for (int i = 0; i < 3; i++)
            if (poX[i] < 0)
            {
                poX[i] = -poX[i];
                asgn = asgn * SoA[i];
            }
        break;
    case 1:
        if (poX[2] < 0)
        {
            poX[2] = -poX[2];
            asgn = asgn * SoA[2];
        }
    }

    int extb[3];

    for (int i = 0; i < 3; i++)
        extb[i] = ext[i];

    switch (Symmetry)
    {
    case 2:
        if (poX[0] < (ghost_width - 0.5) * (CoX[0][1] - CoX[0][0]))
            extb[0] = extb[0] + ghost_width;
        if (poX[1] < (ghost_width - 0.5) * (CoX[1][1] - CoX[1][0]))
            extb[1] = extb[1] + ghost_width;
    case 1:
        if (poX[2] < (ghost_width - 0.5) * (CoX[2][1] - CoX[2][0]))
            extb[2] = extb[2] + ghost_width;
    }

    if (extb[0] > ext[0] || extb[1] > ext[1] || extb[2] > ext[2])
    {
        double *CoXb[3];
        int Nb = extb[0] * extb[1] * extb[2];
        double *datab;
        datab = new double[Nb];
        for (int i = 0; i < 3; i++)
        {
            CoXb[i] = new double[extb[i]];
            double DH = CoX[i][1] - CoX[i][0];
            if (extb[i] > ext[i])
            {
                if (CoX[i][0] > DH)
                {
                    cout << "lower boundary[" << i << "] = " << CoX[i][0] << ", but SYmmetry = " << Symmetry << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                for (int j = 0; j < ghost_width; j++)
                    CoXb[i][j] = -CoX[i][ghost_width - 1 - j];
                for (int j = ghost_width; j < extb[i]; j++)
                    CoXb[i][j] = CoX[i][j - ghost_width];

            }
            else
            {
                for (int j = 0; j < extb[i]; j++)
                    CoXb[i][j] = CoX[i][j];
            }
        }

        for (int i = 0; i < Nb; i++)
        {
            int ind[3], indb[3];
            getarrayindex(3, extb, indb, i);
            double sgn = 1;
            for (int j = 0; j < 3; j++)
            {
                if (extb[j] > ext[j])
                {

                    if (indb[j] < ghost_width)
                    {
                        ind[j] = ghost_width - 1 - indb[j];
                        sgn = sgn * SoA[j];
                    }
                    else
                    {
                        ind[j] = indb[j] - ghost_width;
                    }

                }
                else
                    ind[j] = indb[j];
            }
            int lon = getarraylocation(3, ext, ind);
            datab[i] = datain[lon] * sgn;
        }

        resu = global_interp(DIM, extb, CoXb, datab, poX, ordn);

        for (int i = 0; i < 3; i++)
            delete[] CoXb[i];
        delete[] datab;
    }
    else
    {
        resu = global_interp(DIM, ext, CoX, datain, poX, ordn);
    }

    return resu * asgn;
}
double Parallel::global_interp(int DIM, int *ext, double **CoX, double *datain,
                                                             double *poX, int ordn)
{
    if (ordn > 2 * ghost_width)
    {
        cout << "Parallel::global_interp can not handle ordn = " << ordn << " > 2*ghost_width = " << 2 * ghost_width << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *bbox, *datainbbox;
    bbox = new double[2 * DIM];
    datainbbox = new double[2 * DIM];

    int *NN, *ind, *shape;
    NN = new int[DIM];
    ind = new int[DIM];
    shape = new int[DIM];

    for (int i = 0; i < DIM; i++)
    {
        ind[i] = int((poX[i] - CoX[i][0]) / (CoX[i][1] - CoX[i][0])) - ordn / 2 + 1;
        // poX may exactly locate on the boundary (exclude ghost)
        if (ind[i] == -1 && feq(poX[i], CoX[i][0], (CoX[i][1] - CoX[i][0]) / 2))
            ind[i] = 0;
        /*
                 if(ind[i] < 0)
                 {
                     cout<<"Parallel::global_interp error ind["<<i<<"] = "<<ind[i]<<endl;
                     cout<<"pox = "<<poX[i]<<", CoX[0] = "<<CoX[i][0]<<endl;
                     MPI_Abort(MPI_COMM_WORLD,1);
                 }
        */
        if (ind[i] == ext[i] - ordn + 1 && feq(poX[i], CoX[i][ext[i] - ordn / 2], (CoX[i][1] - CoX[i][0]) / 2))
            ind[i] = ext[i] - ordn - 1;
        /*
                 if(ind[i]+ordn-1 > ext[i]-1)
                 {
                     cout<<"Parallel::global_interp error ind["<<i<<"] = "<<ind[i]<<" + ordn ("<<ordn<<") > ext = "<<ext[i]<<endl;
                     cout<<"pox = "<<poX[i]<<", CoX[ind] = "<<CoX[i][ind[i]]<<", CoX = ("<<CoX[i][0]<<","<<CoX[i][ext[i]-1]<<")"<<endl;
                     MPI_Abort(MPI_COMM_WORLD,1);
                 }
        */
        bbox[i] = CoX[i][ind[i]];
        bbox[DIM + i] = CoX[i][ind[i] + ordn - 1];
        datainbbox[i] = CoX[i][0];
        datainbbox[DIM + i] = CoX[i][ext[i] - 1];
        shape[i] = ordn;
    }

    NN[DIM - 1] = ordn;
    for (int i = DIM - 2; i >= 0; i--)
        NN[i] = NN[i + 1] * ordn;

    double *xpts, *funcvals;
    xpts = new double[ordn];
    funcvals = new double[ordn];
    double *DDd, *DDd1, rr;

    DDd = new double[NN[0]];

    copy(DIM, bbox, bbox + DIM, shape, DDd, datainbbox, datainbbox + DIM, ext, datain, bbox, bbox + DIM);

    for (int i = 0; i < DIM; i++)
    {
        for (int j = ind[i]; j < ind[i] + ordn; j++)
        {
            xpts[j - ind[i]] = CoX[i][j];
        }

        if (i < DIM - 1)
        {
            DDd1 = new double[NN[i + 1]];
            for (int j = 0; j < NN[i + 1]; j++)
            {
                for (int k = 0; k < ordn; k++)
                    funcvals[k] = DDd[k + j * ordn];
                DDd1[j] = Lagrangian_Int(poX[i], ordn, xpts, funcvals);
            }
            delete[] DDd;
            DDd = DDd1;
        }
        else
        {
            for (int j = 0; j < ordn; j++)
                funcvals[j] = DDd[j];
            rr = Lagrangian_Int(poX[i], ordn, xpts, funcvals);
            delete[] DDd1; // since DDd and DDd1 now point to the same stuff, we need delete after above int
        }
    }

    delete[] NN;
    delete[] ind;
    delete[] xpts;
    delete[] funcvals;
    delete[] bbox;
    delete[] datainbbox;
    delete[] shape;

    return rr;
}
double Parallel::Lagrangian_Int(double x, int npts, double *xpts, double *funcvals)
{
    double sum = 0;
    for (int i = 0; i < npts; i++)
    {
        sum = sum + funcvals[i] * LagrangePoly(x, i, npts, xpts);
    }
    return sum;
}
double Parallel::LagrangePoly(double x, int pt, int npts, double *xpts)
{
    double h = 1;
    int i;

    for (i = 0; i < pt; i++)
        h = h * (x - xpts[i]) / (xpts[pt] - xpts[i]);

    for (i = pt + 1; i < npts; i++)
        h = h * (x - xpts[i]) / (xpts[pt] - xpts[i]);

    return h;
}
// collect all grid segments or blocks including ghost and buffer for given patch
MyList<Parallel::gridseg> *Parallel::build_complete_gsl(Patch *Pat)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        if (!cgsl)
        {
            cgsl = gs = new MyList<Parallel::gridseg>; // delete through destroyList();
            gs->data = new Parallel::gridseg;
        }
        else
        {
            gs->next = new MyList<Parallel::gridseg>;
            gs = gs->next;
            gs->data = new Parallel::gridseg;
        }

        for (int i = 0; i < dim; i++)
        {
            gs->data->llb[i] = BP->data->bbox[i];
            gs->data->uub[i] = BP->data->bbox[dim + i];
            gs->data->shape[i] = BP->data->shape[i];
        }
        gs->data->Bg = BP->data;
        gs->next = 0;

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// collect all grid segments or blocks including ghost and buffer for given patch list
MyList<Parallel::gridseg> *Parallel::build_complete_gsl(MyList<Patch> *PatL)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    while (PatL)
    {
        if (!cgsl)
        {
            cgsl = build_complete_gsl(PatL->data);
            gs = cgsl;
            while (gs->next)
                gs = gs->next;
        }
        else
        {
            gs->next = build_complete_gsl(PatL->data);
            gs = gs->next;
            while (gs->next)
                gs = gs->next;
        }
        PatL = PatL->next;
    }

    return cgsl;
}
// cellect the information of Patch list
MyList<Parallel::gridseg> *Parallel::build_complete_gsl_virtual(MyList<Patch> *PatL)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    while (PatL)
    {
        if (cgsl)
        {
            gs->next = new MyList<Parallel::gridseg>;
            gs = gs->next;
            gs->data = new Parallel::gridseg;
        }
        else
        {
            cgsl = gs = new MyList<Parallel::gridseg>;
            gs->data = new Parallel::gridseg;
        }

        for (int i = 0; i < dim; i++)
        {
            gs->data->llb[i] = PatL->data->bbox[i];
            gs->data->uub[i] = PatL->data->bbox[dim + i];
            gs->data->shape[i] = PatL->data->shape[i];
        }
        gs->data->Bg = 0;
        gs->next = 0;

        PatL = PatL->next;
    }

    return cgsl;
}
// cellect the information of Patch list without buffer points
MyList<Parallel::gridseg> *Parallel::build_complete_gsl_virtual2(MyList<Patch> *PatL) // - buffer
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    while (PatL)
    {
        if (cgsl)
        {
            gs->next = new MyList<Parallel::gridseg>;
            gs = gs->next;
            gs->data = new Parallel::gridseg;
        }
        else
        {
            cgsl = gs = new MyList<Parallel::gridseg>;
            gs->data = new Parallel::gridseg;
        }

        for (int i = 0; i < dim; i++)
        {
            double DH = PatL->data->getdX(i);
            gs->data->llb[i] = PatL->data->bbox[i] + PatL->data->lli[i] * DH;
            gs->data->uub[i] = PatL->data->bbox[dim + i] - PatL->data->uui[i] * DH;
            gs->data->shape[i] = PatL->data->shape[i] - PatL->data->lli[i] - PatL->data->uui[i];
        }
        gs->data->Bg = 0;
        gs->next = 0;

        PatL = PatL->next;
    }

    return cgsl;
}
// collect all grid segments or blocks without ghost for given patch, without extension
MyList<Parallel::gridseg> *Parallel::build_bulk_gsl(Patch *Pat)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *bp = BP->data;
        if (!cgsl)
        {
            cgsl = gs = new MyList<Parallel::gridseg>;
            gs->data = new Parallel::gridseg;
        }
        else
        {
            gs->next = new MyList<Parallel::gridseg>;
            gs = gs->next;
            gs->data = new Parallel::gridseg;
        }

        for (int i = 0; i < dim; i++)
        {
            double DH = bp->getdX(i);
            gs->data->uub[i] = (feq(bp->bbox[dim + i], Pat->bbox[dim + i], DH / 2)) ? bp->bbox[dim + i] : bp->bbox[dim + i] - ghost_width * DH;
            gs->data->llb[i] = (feq(bp->bbox[i], Pat->bbox[i], DH / 2)) ? bp->bbox[i] : bp->bbox[i] + ghost_width * DH;

            gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);

        }
        gs->data->Bg = BP->data;
        gs->next = 0;

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// bulk part for given Block within given patch, without extension
MyList<Parallel::gridseg> *Parallel::build_bulk_gsl(Block *bp, Patch *Pat)
{
    MyList<Parallel::gridseg> *gs = 0;

    gs = new MyList<Parallel::gridseg>;
    gs->data = new Parallel::gridseg;

    for (int i = 0; i < dim; i++)
    {
        double DH = bp->getdX(i);
        gs->data->uub[i] = (feq(bp->bbox[dim + i], Pat->bbox[dim + i], DH / 2)) ? bp->bbox[dim + i] : bp->bbox[dim + i] - ghost_width * DH;
        gs->data->llb[i] = (feq(bp->bbox[i], Pat->bbox[i], DH / 2)) ? bp->bbox[i] : bp->bbox[i] + ghost_width * DH;
        gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
    }
    gs->data->Bg = bp;
    gs->next = 0;

    return gs;
}
MyList<Parallel::gridseg> *Parallel::clone_gsl(MyList<Parallel::gridseg> *p, bool first_only)
{
    MyList<Parallel::gridseg> *np = 0, *q = 0, *pq = 0;

    while (p)
    {
        q = new MyList<Parallel::gridseg>;
        q->data = new Parallel::gridseg;
        q->data->Bg = p->data->Bg;
        for (int i = 0; i < dim; i++)
        {
            q->data->llb[i] = p->data->llb[i];
            q->data->uub[i] = p->data->uub[i];
            q->data->shape[i] = p->data->shape[i];
        }
        if (pq)
            pq->next = q;
        else
            np = q;
        if (first_only)
        {
            np->next = 0;
            return np;
        }
        pq = q;
        p = p->next;
    }
    return np;
}
MyList<Parallel::gridseg> *Parallel::gs_subtract(MyList<Parallel::gridseg> *A, MyList<Parallel::gridseg> *B)
{
    if (!A)
        return 0;
    if (!B)
        return clone_gsl(A, true);

    double cut_plane[2 * dim], DH[dim];

    for (int i = 0; i < dim; i++)
    {
        DH[i] = A->data->Bg->getdX(i);
        if (B->data->Bg && !feq(DH[i], B->data->Bg->getdX(i), DH[i] / 2))
        {
            cout << "Parallel::gs_subtract meets different grid segment " << DH[i] << " vs " << B->data->Bg->getdX(i) << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MyList<Parallel::gridseg> *C = 0, *q;
    for (int i = 0; i < dim; i++)
    {
        if (B->data->llb[i] > A->data->uub[i] || B->data->uub[i] < A->data->llb[i])
            return clone_gsl(A, true);
        cut_plane[i] = A->data->llb[i];
        cut_plane[i + dim] = A->data->uub[i];
    }

    for (int i = 0; i < dim; i++)
    {
        cut_plane[i] = Mymax(A->data->llb[i], B->data->llb[i]);
        if (cut_plane[i] - A->data->llb[i] > DH[i] / 2)
        {
            q = clone_gsl(A, true);
            // prolong the list from head
            if (C)
                q->next = C;
            C = q;
            for (int j = 0; j < dim; j++)
            {
                if (i == j)
                {
                    C->data->llb[i] = A->data->llb[i];
                    C->data->uub[i] = Mymax(C->data->llb[i], cut_plane[i]);
                }
                else
                {
                    C->data->llb[j] = cut_plane[j];
                    C->data->uub[j] = cut_plane[j + dim];
                }
                C->data->shape[j] = int((C->data->uub[j] - C->data->llb[j]) / DH[j] + 0.4);
            }
        }

        cut_plane[i + dim] = Mymin(A->data->uub[i], B->data->uub[i]);
        if (A->data->uub[i] - cut_plane[i + dim] > DH[i] / 2)
        {
            q = clone_gsl(A, true);
            if (C)
                q->next = C;
            C = q;
            for (int j = 0; j < dim; j++)
            {
                if (i == j)
                {
                    C->data->uub[i] = A->data->uub[i];
                    C->data->llb[i] = Mymin(C->data->uub[i], cut_plane[i + dim]);
                }
                else
                {
                    C->data->llb[j] = cut_plane[j];
                    C->data->uub[j] = cut_plane[j + dim];
                }
                C->data->shape[j] = int((C->data->uub[j] - C->data->llb[j]) / DH[j] + 0.4);
            }
        }
    }
    return C;
}

// more clever method
MyList<Parallel::gridseg> *Parallel::gsl_subtract(MyList<Parallel::gridseg> *A, MyList<Parallel::gridseg> *B) // A subtract B but with A's information
{
    // always make return and A, B distinct
    if (!A)
        return 0;

    MyList<Parallel::gridseg> *C = 0, *C0, *C1;

    C = clone_gsl(A, 0);

    while (B)
    {
        C0 = 0;
        C1 = C;
        while (C1)
        {
            if (C0)
                C0->catList(gs_subtract(C1, B));
            else
                C0 = gs_subtract(C1, B);
            C1 = C1->next;
        }
        if (C)
            C->destroyList();
        else
        {
            if (C0)
                C0->destroyList();
            return 0;
        }

        C = C0;
        B = B->next;
    }

    return C;
}
MyList<Parallel::gridseg> *Parallel::gs_and(MyList<Parallel::gridseg> *A, MyList<Parallel::gridseg> *B)
{
    if (!A || !B)
        return 0;

    double llb[dim], uub[dim];
    bool flag = false;
    for (int i = 0; i < dim; i++)
    {
        llb[i] = Mymax(A->data->llb[i], B->data->llb[i]);
        uub[i] = Mymin(A->data->uub[i], B->data->uub[i]);
        if (llb[i] > uub[i])
        {
            flag = true;
            break;
        }
    }
    if (flag)
        return 0;

    MyList<Parallel::gridseg> *C;
    C = clone_gsl(A, true);
    for (int i = 0; i < dim; i++)
    {
        C->data->llb[i] = llb[i];
        C->data->uub[i] = uub[i];

        C->data->shape[i] = int((C->data->uub[i] - C->data->llb[i]) / C->data->Bg->getdX(i) + 0.4);

    }

    return C;
}
// overlap of A_i and (union of all j of B_j)
MyList<Parallel::gridseg> *Parallel::gsl_and(MyList<Parallel::gridseg> *A, MyList<Parallel::gridseg> *B) // A and B but with A's information
{
    MyList<Parallel::gridseg> *C = 0, *C1;

    while (A)
    {
        C1 = B;
        while (C1)
        {
            if (C)
                C->catList(gs_and(A, C1));
            else
                C = gs_and(A, C1);
            C1 = C1->next;
        }
        A = A->next;
    }
    return C;
}
// collect all ghost grid segments or blocks for given patch
MyList<Parallel::gridseg> *Parallel::build_ghost_gsl(Patch *Pat)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs, *gsb;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        gs = new MyList<Parallel::gridseg>;
        gs->data = new Parallel::gridseg;

        for (int i = 0; i < dim; i++)
        {
            gs->data->llb[i] = BP->data->bbox[i];
            gs->data->uub[i] = BP->data->bbox[dim + i];
            gs->data->shape[i] = BP->data->shape[i];
        }
        gs->data->Bg = BP->data;
        gs->next = 0;

        gsb = build_bulk_gsl(BP->data, Pat);

        if (!cgsl)
            cgsl = gs_subtract(gs, gsb);
        else
            cgsl->catList(gs_subtract(gs, gsb));

        gsb->destroyList();
        gs->destroyList();

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// collect all ghost grid segments or blocks for given patch list
MyList<Parallel::gridseg> *Parallel::build_ghost_gsl(MyList<Patch> *PatL)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    while (PatL)
    {
        if (!cgsl)
        {
            cgsl = build_ghost_gsl(PatL->data);
            gs = cgsl;
            while (gs->next)
                gs = gs->next;
        }
        else
        {
            gs->next = build_ghost_gsl(PatL->data);
            gs = gs->next;
            while (gs->next)
                gs = gs->next;
        }
        PatL = PatL->next;
    }

    return cgsl;
}
// collect all grid segments or blocks without ghost for given patch
// special for Sync usage, so we do not need consider missing points
MyList<Parallel::gridseg> *Parallel::build_owned_gsl0(Patch *Pat, int rank_in)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *bp = BP->data;
        if (bp->rank == rank_in)
        {
            if (!cgsl)
            {
                cgsl = gs = new MyList<Parallel::gridseg>;
                gs->data = new Parallel::gridseg;
            }
            else
            {
                gs->next = new MyList<Parallel::gridseg>;
                gs = gs->next;
                gs->data = new Parallel::gridseg;
            }

            for (int i = 0; i < dim; i++)
            {
                double DH = bp->getdX(i);
                gs->data->uub[i] = (feq(bp->bbox[dim + i], Pat->bbox[dim + i], DH / 2)) ? bp->bbox[dim + i] : bp->bbox[dim + i] - ghost_width * DH;
                gs->data->llb[i] = (feq(bp->bbox[i], Pat->bbox[i], DH / 2)) ? bp->bbox[i] : bp->bbox[i] + ghost_width * DH;
                gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
            }
            gs->data->Bg = BP->data;
            gs->next = 0;
        }

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// collect all grid segments or blocks without ghost for given patch
MyList<Parallel::gridseg> *Parallel::build_owned_gsl1(Patch *Pat, int rank_in)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *bp = BP->data;
        if (bp->rank == rank_in)
        {
            if (!cgsl)
            {
                cgsl = gs = new MyList<Parallel::gridseg>;
                gs->data = new Parallel::gridseg;
            }
            else
            {
                gs->next = new MyList<Parallel::gridseg>;
                gs = gs->next;
                gs->data = new Parallel::gridseg;
            }

            for (int i = 0; i < dim; i++)
            {
                double DH = bp->getdX(i);
                gs->data->uub[i] = (feq(bp->bbox[dim + i], Pat->bbox[dim + i], DH / 2)) ? bp->bbox[dim + i] : bp->bbox[dim + i] - ghost_width * DH;
                gs->data->llb[i] = (feq(bp->bbox[i], Pat->bbox[i], DH / 2)) ? bp->bbox[i] : bp->bbox[i] + ghost_width * DH;
                gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
            }
            gs->data->Bg = BP->data;
            gs->next = 0;
        }

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// collect all grid segments or blocks without ghost nor buffer for given patch
MyList<Parallel::gridseg> *Parallel::build_owned_gsl2(Patch *Pat, int rank_in)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *bp = BP->data;
        if (bp->rank == rank_in)
        {
            if (!cgsl)
            {
                cgsl = gs = new MyList<Parallel::gridseg>;
                gs->data = new Parallel::gridseg;
            }
            else
            {
                gs->next = new MyList<Parallel::gridseg>;
                gs = gs->next;
                gs->data = new Parallel::gridseg;
            }

            for (int i = 0; i < dim; i++)
            {
                double DH = bp->getdX(i);
                gs->data->uub[i] = (feq(bp->bbox[dim + i], Pat->bbox[dim + i], DH / 2)) ? bp->bbox[dim + i] - Pat->uui[i] * DH : bp->bbox[dim + i] - ghost_width * DH;
                gs->data->llb[i] = (feq(bp->bbox[i], Pat->bbox[i], DH / 2)) ? bp->bbox[i] + Pat->lli[i] * DH : bp->bbox[i] + ghost_width * DH;
                gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
            }
            gs->data->Bg = BP->data;
            gs->next = 0;
        }

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// collect all grid segments or blocks without ghost for given patch, and delete the ghost_width for interpolation consideration on the patch boundary
MyList<Parallel::gridseg> *Parallel::build_owned_gsl3(Patch *Pat, int rank_in, int Symmetry)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *bp = BP->data;
        if (bp->rank == rank_in)
        {
            if (!cgsl)
            {
                cgsl = gs = new MyList<Parallel::gridseg>;
                gs->data = new Parallel::gridseg;
            }
            else
            {
                gs->next = new MyList<Parallel::gridseg>;
                gs = gs->next;
                gs->data = new Parallel::gridseg;
            }

            for (int i = 0; i < dim; i++)
            {
                double DH = bp->getdX(i);
                gs->data->uub[i] = bp->bbox[dim + i] - ghost_width * DH;
                gs->data->llb[i] = bp->bbox[i] + ghost_width * DH;
                gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
            }
            // Symmetry consideration
            if (Symmetry > 0)
            {
                double DH = bp->getdX(2);
                if (feq(bp->bbox[2], 0, DH / 2))
                {
                    gs->data->llb[2] = bp->bbox[2];
                    gs->data->shape[2] = int((gs->data->uub[2] - gs->data->llb[2]) / DH + 0.4);
                }
                if (Symmetry > 1)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        DH = bp->getdX(i);
                        if (feq(bp->bbox[i], 0, DH / 2))
                        {
                            gs->data->llb[i] = bp->bbox[i];
                            gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
                        }
                    }
                }
            }

            gs->data->Bg = BP->data;
            gs->next = 0;
        }

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// collect all grid segments or blocks without ghost nor buffer for given patch,
// and delete the ghost_width for interpolation consideration on the patch boundary
MyList<Parallel::gridseg> *Parallel::build_owned_gsl4(Patch *Pat, int rank_in, int Symmetry)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *bp = BP->data;
        if (bp->rank == rank_in)
        {
            if (!cgsl)
            {
                cgsl = gs = new MyList<Parallel::gridseg>;
                gs->data = new Parallel::gridseg;
            }
            else
            {
                gs->next = new MyList<Parallel::gridseg>;
                gs = gs->next;
                gs->data = new Parallel::gridseg;
            }

            for (int i = 0; i < dim; i++)
            {
                double DH = bp->getdX(i);
                gs->data->uub[i] = (feq(bp->bbox[dim + i], Pat->bbox[dim + i], DH / 2)) ? bp->bbox[dim + i] - Pat->uui[i] * DH : bp->bbox[dim + i];
                gs->data->uub[i] -= ghost_width * DH;
                gs->data->llb[i] = (feq(bp->bbox[i], Pat->bbox[i], DH / 2)) ? bp->bbox[i] + Pat->lli[i] * DH : bp->bbox[i];
                gs->data->llb[i] += ghost_width * DH;
                gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
            }
            // Symmetry consideration
            if (Symmetry > 0)
            {
                double DH = bp->getdX(2);
                if (feq(bp->bbox[2], 0, DH / 2))
                {
                    gs->data->llb[2] = bp->bbox[2];
                    gs->data->shape[2] = int((gs->data->uub[2] - gs->data->llb[2]) / DH + 0.4);
                }
                if (Symmetry > 1)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        DH = bp->getdX(i);
                        if (feq(bp->bbox[i], 0, DH / 2))
                        {
                            gs->data->llb[i] = bp->bbox[i];
                            gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
                        }
                    }
                }
            }

            gs->data->Bg = BP->data;
            gs->next = 0;
        }

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// collect all grid segments or blocks without ghost nor buffer for given patch, no extention
MyList<Parallel::gridseg> *Parallel::build_owned_gsl5(Patch *Pat, int rank_in)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *bp = BP->data;
        if (bp->rank == rank_in)
        {
            if (!cgsl)
            {
                cgsl = gs = new MyList<Parallel::gridseg>;
                gs->data = new Parallel::gridseg;
            }
            else
            {
                gs->next = new MyList<Parallel::gridseg>;
                gs = gs->next;
                gs->data = new Parallel::gridseg;
            }

            for (int i = 0; i < dim; i++)
            {
                double DH = bp->getdX(i);
                gs->data->uub[i] = (feq(bp->bbox[dim + i], Pat->bbox[dim + i], DH / 2)) ? bp->bbox[dim + i] - Pat->uui[i] * DH : bp->bbox[dim + i] - ghost_width * DH;
                gs->data->llb[i] = (feq(bp->bbox[i], Pat->bbox[i], DH / 2)) ? bp->bbox[i] + Pat->lli[i] * DH : bp->bbox[i] + ghost_width * DH;
                gs->data->shape[i] = int((gs->data->uub[i] - gs->data->llb[i]) / DH + 0.4);
            }
            gs->data->Bg = BP->data;
            gs->next = 0;
        }

        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    return cgsl;
}
// collect all grid segments or blocks without ghost for given patch list
// more clever method
MyList<Parallel::gridseg> *Parallel::build_owned_gsl(MyList<Patch> *PatL, int rank_in, int type, int Symmetry)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    while (PatL)
    {
        switch (type)
        {
        case 0:
            gs = build_owned_gsl0(PatL->data, rank_in);
            break;
        case 1:
            gs = build_owned_gsl1(PatL->data, rank_in);
            break;
        case 2:
            gs = build_owned_gsl2(PatL->data, rank_in);
            break;
        case 3:
            gs = build_owned_gsl3(PatL->data, rank_in, Symmetry);
            break;
        case 4:
            gs = build_owned_gsl4(PatL->data, rank_in, Symmetry);
            break;
        case 5:
            gs = build_owned_gsl5(PatL->data, rank_in);
            break;
        default:
            cout << "Parallel::build_owned_gsl : unknown type = " << type << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (cgsl)
            cgsl->catList(gs);
        else
            cgsl = gs;
        PatL = PatL->next;
    }

    return cgsl;
}
// according to overlape to determine real grid segments
void Parallel::build_gstl(MyList<Parallel::gridseg> *srci, MyList<Parallel::gridseg> *dsti,
                                                    MyList<Parallel::gridseg> **out_src, MyList<Parallel::gridseg> **out_dst)
{
    *out_src = *out_dst = 0;

    if (!srci || !dsti)
        return;

    MyList<Parallel::gridseg> *s, *d;
    MyList<Parallel::gridseg> *s2, *d2;

    double llb[dim], uub[dim];

    s = srci;
    while (s)
    {
        Parallel::gridseg *sd = s->data;
        d = dsti;
        while (d)
        {
            Parallel::gridseg *dd = d->data;
            bool flag = true;
            for (int i = 0; i < dim; i++)
            {
                double SH = sd->Bg->getdX(i), DH = dd->Bg->getdX(i);
                llb[i] = Mymax(sd->llb[i], dd->llb[i]);
                uub[i] = Mymin(sd->uub[i], dd->uub[i]);
                // make sure the region boundary is consistent to the grids
                // here we only judge if the domain is empty, so do not need to adjust the align
                double lb = llb[i], ub = uub[i];
                // |------|
                // |-------------|
                //		if     (int(2*(sd->uub[i]-uub[i])/SH+0.4)%2 == 1) ub = uub[i]+SH/2;
                //		else if(int(2*(dd->uub[i]-uub[i])/DH+0.4)%2 == 1) ub = uub[i]+DH/2;
                //        |------|
                // |-------------|
                //		if     (int(2*(llb[i]-sd->llb[i])/SH+0.4)%2 == 1) lb = llb[i]-SH/2;
                //		else if(int(2*(llb[i]-dd->llb[i])/DH+0.4)%2 == 1) lb = llb[i]-DH/2;
                if (ub - lb < Mymin(SH, DH) / 2)
                {
                    flag = false;
                    break;
                } // even for isolated point, it has a cell belong to it
            }

            if (flag)
            {
                if (!(*out_src))
                {
                    *out_src = s2 = new MyList<Parallel::gridseg>;
                    *out_dst = d2 = new MyList<Parallel::gridseg>;
                    s2->data = new Parallel::gridseg;
                    d2->data = new Parallel::gridseg;
                }
                else
                {
                    s2->next = new MyList<Parallel::gridseg>;
                    s2 = s2->next;
                    d2->next = new MyList<Parallel::gridseg>;
                    d2 = d2->next;
                    s2->data = new Parallel::gridseg;
                    d2->data = new Parallel::gridseg;
                }

                for (int i = 0; i < dim; i++)
                {
                    double SH = sd->Bg->getdX(i), DH = dd->Bg->getdX(i);
                    s2->data->llb[i] = d2->data->llb[i] = llb[i];
                    s2->data->uub[i] = d2->data->uub[i] = uub[i];
// using float method to count point, we do not need following consideration (2012 nov 17)
                    if (int(2 * (sd->uub[i] - uub[i]) / SH + 0.4) % 2 == 1)
                        s2->data->uub[i] = uub[i] + SH / 2;
                    else if (int(2 * (dd->uub[i] - uub[i]) / DH + 0.4) % 2 == 1)
                        d2->data->uub[i] = uub[i] + DH / 2;
                    if (int(2 * (llb[i] - sd->llb[i]) / SH + 0.4) % 2 == 1)
                        s2->data->llb[i] = llb[i] - SH / 2;
                    else if (int(2 * (llb[i] - dd->llb[i]) / DH + 0.4) % 2 == 1)
                        d2->data->llb[i] = llb[i] - DH / 2;
                    s2->data->shape[i] = int((s2->data->uub[i] - s2->data->llb[i]) / SH + 0.4);
                    d2->data->shape[i] = int((d2->data->uub[i] - d2->data->llb[i]) / DH + 0.4);
                    s2->data->illb[i] = sd->illb[i];
                    d2->data->illb[i] = dd->illb[i];
                    s2->data->iuub[i] = sd->iuub[i];
                    d2->data->iuub[i] = dd->iuub[i];
                }
                s2->data->Bg = sd->Bg;
                s2->next = 0;
                d2->data->Bg = dd->Bg;
                d2->next = 0;
            }
            d = d->next;
        }
        s = s->next;
    }
}

//   PACK: prepare target data in 'data'
// UNPACK: copy target data from 'data' to corresponding numerical grids
int Parallel::data_packer(double *data, MyList<Parallel::gridseg> *src, MyList<Parallel::gridseg> *dst, int rank_in, int dir,
                          MyList<var> *VarLists /* source */, MyList<var> *VarListd /* target */, int Symmetry)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int DIM = dim;

  if (dir != PACK && dir != UNPACK)
  {
    cout << "error dir " << dir << " for data_packer " << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int size_out = 0;

  if (!src || !dst)
    return size_out;

  MyList<var> *varls, *varld;

  varls = VarLists;
  varld = VarListd;
  while (varls && varld)
  {
    varls = varls->next;
    varld = varld->next;
  }

  if (varls || varld)
  {
    cout << "error in short data packer, var lists does not match." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int type; /* 1 copy, 2 restrict, 3 prolong */
  if (src->data->Bg->lev == dst->data->Bg->lev)
    type = 1;
  else if (src->data->Bg->lev > dst->data->Bg->lev)
    type = 2;
  else
    type = 3;

  while (src && dst)
  {
    if ((dir == PACK && dst->data->Bg->rank == rank_in && src->data->Bg->rank == myrank) ||
        (dir == UNPACK && src->data->Bg->rank == rank_in && dst->data->Bg->rank == myrank))
    {
      varls = VarLists;
      varld = VarListd;
      while (varls && varld)
      {
        if (data)
        {
          if (dir == PACK)
            switch (type)
            {
              // attention must be paied to the difference between src's llb,uub and dst's llb,uub
            case 1:
              f_copy(DIM, dst->data->llb, dst->data->uub, dst->data->shape, data + size_out,
                     src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, src->data->Bg->fgfs[varls->data->sgfn],
                     dst->data->llb, dst->data->uub);
              break;
            case 2:
              f_restrict3(DIM, dst->data->llb, dst->data->uub, dst->data->shape, data + size_out,
                          src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, src->data->Bg->fgfs[varls->data->sgfn],
                          dst->data->llb, dst->data->uub, varls->data->SoA, Symmetry);
              break;
            case 3:
              f_prolong3(DIM, src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, src->data->Bg->fgfs[varls->data->sgfn],
                         dst->data->llb, dst->data->uub, dst->data->shape, data + size_out,
                         dst->data->llb, dst->data->uub, varls->data->SoA, Symmetry);
            }
          if (dir == UNPACK) // from target data to corresponding grid
            f_copy(DIM, dst->data->Bg->bbox, dst->data->Bg->bbox + dim, dst->data->Bg->shape, dst->data->Bg->fgfs[varld->data->sgfn],
                   dst->data->llb, dst->data->uub, dst->data->shape, data + size_out,
                   dst->data->llb, dst->data->uub);
        }
        size_out += dst->data->shape[0] * dst->data->shape[1] * dst->data->shape[2];
        varls = varls->next;
        varld = varld->next;
      }
    }
    dst = dst->next;
    src = src->next;
  }

  return size_out;
}

int Parallel::data_packermix(double *data, MyList<Parallel::gridseg> *src, MyList<Parallel::gridseg> *dst, int rank_in, int dir,
                                                         MyList<var> *VarLists /* source */, MyList<var> *VarListd /* target */, int Symmetry)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int DIM = dim;

    if (dir != PACK && dir != UNPACK)
    {
        cout << "Parallel::data_packermix: error dir " << dir << " for data_packermix." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int size_out = 0;

    if (!src || !dst)
        return size_out;

    MyList<var> *varls, *varld;

    varls = VarLists;
    varld = VarListd;
    while (varls && varld)
    {
        varls = varls->next;
        varld = varld->next;
    }

    if (varls || varld)
    {
        cout << "error in short data packer, var lists does not match." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int type; /* 1 copy, 2 restrict, 3 prolong */
    if (src->data->Bg->lev == dst->data->Bg->lev)
        type = 1;
    else if (src->data->Bg->lev > dst->data->Bg->lev)
        type = 2;
    else
        type = 3;

    if (type != 3)
    {
        cout << "Parallel::data_packermix: error type " << type << " for data_packermix." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    while (src && dst)
    {
        if ((dir == PACK && dst->data->Bg->rank == rank_in && src->data->Bg->rank == myrank) ||
                (dir == UNPACK && src->data->Bg->rank == rank_in && dst->data->Bg->rank == myrank))
        {
            varls = VarLists;
            varld = VarListd;
            while (varls && varld)
            {
                if (data)
                {
                    if (dir == PACK)
                        f_prolongcopy3(DIM, src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, src->data->Bg->fgfs[varls->data->sgfn],
                                                     dst->data->llb, dst->data->uub, src->data->shape, data + size_out,
                                                     src->data->llb, src->data->uub, varls->data->SoA, Symmetry);
                    if (dir == UNPACK) // from target data to corresponding grid
                        f_prolongmix3(DIM, dst->data->Bg->bbox, dst->data->Bg->bbox + dim, dst->data->Bg->shape, dst->data->Bg->fgfs[varld->data->sgfn],
                                                    src->data->llb, src->data->uub, src->data->shape, data + size_out,
                                                    dst->data->llb, dst->data->uub, varls->data->SoA, Symmetry, dst->data->illb, dst->data->iuub);
                }
                // the symmetry problem should be dealt in prolongcopy3,
                // so we always have ghost_width for both sides
                size_out += (src->data->shape[0] + 2 * ghost_width) * (src->data->shape[1] + 2 * ghost_width) * (src->data->shape[2] + 2 * ghost_width);
                varls = varls->next;
                varld = varld->next;
            }
        }
        dst = dst->next;
        src = src->next;
    }

    return size_out;
}

void Parallel::transfer(MyList<Parallel::gridseg> **src, MyList<Parallel::gridseg> **dst,
                                                MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /*target */,
                                                int Symmetry)
{
    int myrank, cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int node;

    MPI_Request *reqs;
    MPI_Status *stats;
    reqs = new MPI_Request[2 * cpusize];
    stats = new MPI_Status[2 * cpusize];
    int req_no = 0;

    double **send_data, **rec_data;
    send_data = new double *[cpusize];
    rec_data = new double *[cpusize];
    int length;

    for (node = 0; node < cpusize; node++)
    {
        send_data[node] = rec_data[node] = 0;
        if (node == myrank)
        {
            if (length = data_packer(0, src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry))
            {
                rec_data[node] = new double[length];
                if (!rec_data[node])
                {
                    cout << "out of memory when new in short transfer, place 1" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                data_packer(rec_data[node], src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry);
            }
        }
        else
        {
            // send from this cpu to cpu#node
            if (length = data_packer(0, src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry))
            {
                send_data[node] = new double[length];
                if (!send_data[node])
                {
                    cout << "out of memory when new in short transfer, place 2" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                data_packer(send_data[node], src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry);
                MPI_Isend((void *)send_data[node], length, MPI_DOUBLE, node, 1, MPI_COMM_WORLD, reqs + req_no++);
            }
            // receive from cpu#node to this cpu
            if (length = data_packer(0, src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry))
            {
                rec_data[node] = new double[length];
                if (!rec_data[node])
                {
                    cout << "out of memory when new in short transfer, place 3" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                MPI_Irecv((void *)rec_data[node], length, MPI_DOUBLE, node, 1, MPI_COMM_WORLD, reqs + req_no++);
            }
        }
    }
    // wait for all requests to complete
    MPI_Waitall(req_no, reqs, stats);

    for (node = 0; node < cpusize; node++)
        if (rec_data[node])
            data_packer(rec_data[node], src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry);

    for (node = 0; node < cpusize; node++)
    {
        if (send_data[node])
            delete[] send_data[node];
        if (rec_data[node])
            delete[] rec_data[node];
    }

    delete[] reqs;
    delete[] stats;
    delete[] send_data;
    delete[] rec_data;
}
//
void Parallel::transfermix(MyList<Parallel::gridseg> **src, MyList<Parallel::gridseg> **dst,
                                                     MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /*target */,
                                                     int Symmetry)
{
    int myrank, cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int node;

    MPI_Request *reqs;
    MPI_Status *stats;
    reqs = new MPI_Request[2 * cpusize];
    stats = new MPI_Status[2 * cpusize];
    int req_no = 0;

    double **send_data, **rec_data;
    send_data = new double *[cpusize];
    rec_data = new double *[cpusize];
    int length;

    for (node = 0; node < cpusize; node++)
    {
        send_data[node] = rec_data[node] = 0;
        if (node == myrank)
        {
            if (length = data_packermix(0, src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry))
            {
                rec_data[node] = new double[length];
                if (!rec_data[node])
                {
                    cout << "out of memory when new in short transfer, place 1" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                data_packermix(rec_data[node], src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry);
            }
        }
        else
        {
            // send from this cpu to cpu#node
            if (length = data_packermix(0, src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry))
            {
                send_data[node] = new double[length];
                if (!send_data[node])
                {
                    cout << "out of memory when new in short transfer, place 2" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                data_packermix(send_data[node], src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry);
                MPI_Isend((void *)send_data[node], length, MPI_DOUBLE, node, 1, MPI_COMM_WORLD, reqs + req_no++);
            }
            // receive from cpu#node to this cpu
            if (length = data_packermix(0, src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry))
            {
                rec_data[node] = new double[length];
                if (!rec_data[node])
                {
                    cout << "out of memory when new in short transfer, place 3" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                MPI_Irecv((void *)rec_data[node], length, MPI_DOUBLE, node, 1, MPI_COMM_WORLD, reqs + req_no++);
            }
        }
    }
    // wait for all requests to complete
    MPI_Waitall(req_no, reqs, stats);

    for (node = 0; node < cpusize; node++)
        if (rec_data[node])
            data_packermix(rec_data[node], src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry);

    for (node = 0; node < cpusize; node++)
    {
        if (send_data[node])
            delete[] send_data[node];
        if (rec_data[node])
            delete[] rec_data[node];
    }

    delete[] reqs;
    delete[] stats;
    delete[] send_data;
    delete[] rec_data;
}

void Parallel::Sync(Patch *Pat, MyList<var> *VarList, int Symmetry)
{
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_ghost_gsl(Pat); // ghost region only
    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl0(Pat, node);                              // for the part without ghost points and do not extend
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer_src[node], data locate on cpu#node;
                                                                                                                                                    // but for transfer_dst[node] the data may locate on any node
    }

    transfer(transfer_src, transfer_dst, VarList, VarList, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
void Parallel::Sync(MyList<Patch> *PatL, MyList<var> *VarList, int Symmetry)
{
    // Patch inner Synch
    MyList<Patch> *Pp = PatL;
    while (Pp)
    {
        Sync(Pp->data, VarList, Symmetry);
        Pp = Pp->next;
    }

    // Patch inter Synch
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_buffer_gsl(PatL); // buffer region only
    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl(PatL, node, 5, Symmetry);                 // for the part without ghost nor buffer points and do not extend
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfer(transfer_src, transfer_dst, VarList, VarList, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
// collect buffer grid segments or blocks for the periodic boundary condition of given patch
// ---------------------------------------------------
// |con |                                       |con |
// |ner |                PhysBD                 |ner |
// |-------------------------------------------------|
// |    |                                       |    |
// |Phy |                                       |Phy |
// |sBD |                                       |BD  |
// |    |                                       |    |
// |    |                                       |    |
// |    |                                       |    |
// |-------------------------------------------------|
// |con |               PhysBD                  |con |
// |ner |                                       |ner |
// ---------------------------------------------------
// first order derivetive does not need conner information,
// but second order derivative needs!
// the following code includes conner part
MyList<Parallel::gridseg> *Parallel::build_PhysBD_gsl(Patch *Pat)
{
    MyList<Parallel::gridseg> *cgsl, *gsc, *gsb = 0, *p;

    gsc = build_complete_gsl(Pat);

    gsb = new MyList<Parallel::gridseg>;
    gsb->data = new Parallel::gridseg;
    gsb->next = 0;
    gsb->data->Bg = 0;

    for (int j = 0; j < dim; j++)
    {
        gsb->data->llb[j] = Pat->bbox[j];
        gsb->data->uub[j] = Pat->bbox[j + dim];
        gsb->data->shape[j] = Pat->shape[j];
    }

    p = gsl_subtract(gsc, gsb);

    gsc->destroyList();
    gsb->destroyList();

    cgsl = divide_gsl(p, Pat);

    p->destroyList();

    return cgsl;
}
MyList<Parallel::gridseg> *Parallel::divide_gsl(MyList<Parallel::gridseg> *p, Patch *Pat)
{
    MyList<Parallel::gridseg> *cgsl = 0;
    while (p)
    {
        if (cgsl)
            cgsl->catList(divide_gs(p, Pat));
        else
            cgsl = divide_gs(p, Pat);
        p = p->next;
    }

    return cgsl;
}
// divide the gs into pices which locate either totally outside of the given Patch coordinate range
// or totally inside it. It's usefull for periodic boundary condition
MyList<Parallel::gridseg> *Parallel::divide_gs(MyList<Parallel::gridseg> *p, Patch *Pat)
{
    double DH[dim];
    for (int i = 0; i < dim; i++)
    {
        DH[i] = p->data->Bg->getdX(i);
    }

    int num[dim];
    double llb[3][dim], uub[3][dim];
    for (int i = 0; i < dim; i++)
    {
        if (p->data->llb[i] < Pat->bbox[i] - DH[i] / 2)
        {
            if (p->data->uub[i] > Pat->bbox[i + dim] + DH[i] / 2)
            {
                num[i] = 3;
                llb[0][i] = p->data->llb[i];
                llb[1][i] = Pat->bbox[i];
                uub[1][i] = Pat->bbox[i + dim];
                uub[2][i] = p->data->uub[i];
                uub[0][i] = Pat->bbox[i];
                llb[2][i] = Pat->bbox[i + dim];
            }
            else if (p->data->uub[i] > Pat->bbox[i] + DH[i] / 2)
            {
                num[i] = 2;
                llb[0][i] = p->data->llb[i];
                llb[1][i] = Pat->bbox[i];
                uub[1][i] = p->data->uub[i];

                uub[0][i] = Pat->bbox[i];
            }
            else
            {
                num[i] = 1;
                llb[0][i] = p->data->llb[i];
                uub[0][i] = p->data->uub[i];
            }
        }
        else if (p->data->llb[i] < Pat->bbox[i + dim] - DH[i] / 2)
        {
            if (p->data->uub[i] > Pat->bbox[i + dim] + DH[i] / 2)
            {
                num[i] = 2;
                llb[0][i] = p->data->llb[i];
                uub[0][i] = Pat->bbox[i + dim];
                uub[1][i] = p->data->uub[i];

                llb[1][i] = Pat->bbox[i + dim];
            }
            else
            {
                num[i] = 1;
                llb[0][i] = p->data->llb[i];
                uub[0][i] = p->data->uub[i];
            }
        }
        else
        {
            num[i] = 1;
            llb[0][i] = p->data->llb[i];
            uub[0][i] = p->data->uub[i];
        }
    }
    MyList<Parallel::gridseg> *cgsl = 0, *gg;
    int NN = 1;
    for (int i = 0; i < dim; i++)
        NN = NN * num[i];

    for (int i = 0; i < NN; i++)
    {
        int ind[dim];
        getarrayindex(dim, num, ind, i);
        gg = clone_gsl(p, true);
        for (int k = 0; k < dim; k++)
        {
            gg->data->llb[k] = llb[ind[k]][k];
            gg->data->uub[k] = uub[ind[k]][k];

            gg->data->shape[k] = int((uub[ind[k]][k] - llb[ind[k]][k]) / DH[k] + 0.4);

        }

        if (cgsl)
            cgsl->catList(gg);
        else
            cgsl = gg;
    }

    return cgsl;
}
// after mod operation, according to overlape to determine real grid segments
void Parallel::build_PhysBD_gstl(Patch *Pat, MyList<Parallel::gridseg> *srci, MyList<Parallel::gridseg> *dsti,
                                                                 MyList<Parallel::gridseg> **out_src, MyList<Parallel::gridseg> **out_dst)
{
    *out_src = *out_dst = 0;

    if (!srci || !dsti)
        return;

    MyList<Parallel::gridseg> *s, *d;
    MyList<Parallel::gridseg> *s2, *d2;

    double llb[dim], uub[dim];

    s = srci;
    while (s)
    {
        Parallel::gridseg *sd = s->data;
        d = dsti;
        while (d)
        {
            Parallel::gridseg *dd = d->data;
            bool flag = true;
            for (int i = 0; i < dim; i++)
            {
                double SH = sd->Bg->getdX(i), DH = dd->Bg->getdX(i);
                if (!feq(SH, DH, SH / 2))
                {
                    cout << "Parallel::build_PhysBD_gstl meets different grid space SH = " << SH << ", DH = " << DH << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                // we assume dst and src locate on the same Patch
                if (dd->llb[i] < Pat->bbox[i])
                    llb[i] = Mymax(sd->llb[i], dd->llb[i] + Pat->bbox[dim + i] - Pat->bbox[i]);
                else if (dd->llb[i] > Pat->bbox[i + dim])
                    llb[i] = Mymax(sd->llb[i], dd->llb[i] - Pat->bbox[dim + i] + Pat->bbox[i]);
                else
                    llb[i] = Mymax(sd->llb[i], dd->llb[i]);

                if (dd->uub[i] < Pat->bbox[i])
                    uub[i] = Mymin(sd->uub[i], dd->uub[i] + Pat->bbox[dim + i] - Pat->bbox[i]);
                else if (dd->uub[i] > Pat->bbox[dim + i])
                    uub[i] = Mymin(sd->uub[i], dd->uub[i] - Pat->bbox[dim + i] + Pat->bbox[i]);
                else
                    uub[i] = Mymin(sd->uub[i], dd->uub[i]);

                if (llb[i] > uub[i])
                {
                    flag = false;
                    break;
                }
            }

            if (flag)
            {
                if (!(*out_src))
                {
                    *out_src = s2 = new MyList<Parallel::gridseg>;
                    *out_dst = d2 = new MyList<Parallel::gridseg>;
                    s2->data = new Parallel::gridseg;
                    d2->data = new Parallel::gridseg;
                }
                else
                {
                    s2->next = new MyList<Parallel::gridseg>;
                    s2 = s2->next;
                    d2->next = new MyList<Parallel::gridseg>;
                    d2 = d2->next;
                    s2->data = new Parallel::gridseg;
                    d2->data = new Parallel::gridseg;
                }

                for (int i = 0; i < dim; i++)
                {
                    double SH = sd->Bg->getdX(i), DH = dd->Bg->getdX(i);
                    s2->data->llb[i] = llb[i];
                    s2->data->uub[i] = uub[i];

                    if (dd->llb[i] < Pat->bbox[i])
                        d2->data->llb[i] = llb[i] - Pat->bbox[dim + i] + Pat->bbox[i];
                    else if (dd->llb[i] > Pat->bbox[i + dim])
                        d2->data->llb[i] = llb[i] + Pat->bbox[dim + i] - Pat->bbox[i];
                    else
                        d2->data->llb[i] = llb[i];

                    if (dd->uub[i] < Pat->bbox[i])
                        d2->data->uub[i] = uub[i] - Pat->bbox[dim + i] + Pat->bbox[i];
                    else if (dd->uub[i] > Pat->bbox[dim + i])
                        d2->data->uub[i] = uub[i] + Pat->bbox[dim + i] - Pat->bbox[i];
                    else
                        d2->data->uub[i] = uub[i];

                    s2->data->shape[i] = int((s2->data->uub[i] - s2->data->llb[i]) / SH + 0.4);
                    d2->data->shape[i] = int((d2->data->uub[i] - d2->data->llb[i]) / DH + 0.4);

                }
                s2->data->Bg = sd->Bg;
                s2->next = 0;
                d2->data->Bg = dd->Bg;
                d2->next = 0;
            }
            d = d->next;
        }
        s = s->next;
    }
}
void Parallel::PeriodicBD(Patch *Pat, MyList<var> *VarList, int Symmetry)
{
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_PhysBD_gsl(Pat);
    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl0(Pat, node);                                          // for the part without ghost points and do not extend
        build_PhysBD_gstl(Pat, src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfer(transfer_src, transfer_dst, VarList, VarList, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
double Parallel::L2Norm(Patch *Pat, var *vf)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double tvf, dtvf = 0;
    int BDW = ghost_width;

    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *cg = BP->data;
        if (myrank == cg->rank)
        {
            f_l2normhelper(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                         Pat->bbox[0], Pat->bbox[1], Pat->bbox[2],
                                         Pat->bbox[3], Pat->bbox[4], Pat->bbox[5],
                                         cg->fgfs[vf->sgfn], tvf, BDW);
            dtvf += tvf;
        }
        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    MPI_Allreduce(&dtvf, &tvf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    tvf = sqrt(tvf);

    return tvf;
}
double Parallel::L2Norm(Patch *Pat, var *vf, MPI_Comm Comm_here)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double tvf, dtvf = 0;
    int BDW = ghost_width;

    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *cg = BP->data;
        if (myrank == cg->rank)
        {
            f_l2normhelper(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                         Pat->bbox[0], Pat->bbox[1], Pat->bbox[2],
                                         Pat->bbox[3], Pat->bbox[4], Pat->bbox[5],
                                         cg->fgfs[vf->sgfn], tvf, BDW);
            dtvf += tvf;
        }
        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }

    MPI_Allreduce(&dtvf, &tvf, 1, MPI_DOUBLE, MPI_SUM, Comm_here);

    tvf = sqrt(tvf);

    return tvf;
}
void Parallel::checkgsl(MyList<Parallel::gridseg> *pp, bool first_only)
{
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0)
    {
        if (!pp)
            cout << " Parallel::checkgsl meets empty gsl" << endl;
        while (pp)
        {
            if (pp->data->Bg)
                cout << " on node#" << pp->data->Bg->rank << endl;
            else
                cout << " virtual grid segment" << endl;
            cout << " shape: (";
            for (int i = 0; i < dim; i++)
            {
                if (i < dim - 1)
                    cout << pp->data->shape[i] << ",";
                else
                    cout << pp->data->shape[i] << ")" << endl;
            }
            cout << " range: (";
            for (int i = 0; i < dim; i++)
            {
                if (i < dim - 1)
                    cout << pp->data->llb[i] << ":" << pp->data->uub[i] << ",";
                else
                    cout << pp->data->llb[i] << ":" << pp->data->uub[i] << ")" << endl;
            }
            if (first_only)
                return;
            pp = pp->next;
        }
    }
}
void Parallel::checkvarl(MyList<var> *pp, bool first_only)
{
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0)
    {
        while (pp)
        {
            cout << "name: " << pp->data->name << endl;
            cout << "SoA = (" << pp->data->SoA[0] << "," << pp->data->SoA[1] << "," << pp->data->SoA[2] << ")" << endl;
            cout << "sgfn = " << pp->data->sgfn << endl;
            if (first_only)
                return;
            pp = pp->next;
        }
    }
}
void Parallel::prepare_inter_time_level(MyList<Patch> *PatL,
                                                                                MyList<var> *VarList1 /* source (t+dt) */, MyList<var> *VarList2 /* source (t) */,
                                                                                MyList<var> *VarList3 /* target (t+a*dt) */, int tindex)
{
    while (PatL)
    {
        prepare_inter_time_level(PatL->data, VarList1, VarList2, VarList3, tindex);
        PatL = PatL->next;
    }
}
void Parallel::prepare_inter_time_level(Patch *Pat,
                                                                                MyList<var> *VarList1 /* source (t+dt) */, MyList<var> *VarList2 /* source (t) */,
                                                                                MyList<var> *VarList3 /* target (t+a*dt) */, int tindex)
{
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MyList<var> *varl1;
    MyList<var> *varl2;
    MyList<var> *varl3;

    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *cg = BP->data;
        if (myrank == cg->rank)
        {
            varl1 = VarList1;
            varl2 = VarList2;
            varl3 = VarList3;
            while (varl1)
            {
                if (tindex == 0)
                    f_average(cg->shape, cg->fgfs[varl1->data->sgfn], cg->fgfs[varl2->data->sgfn], cg->fgfs[varl3->data->sgfn]);
                else if (tindex == 1)
                    f_average3(cg->shape, cg->fgfs[varl1->data->sgfn], cg->fgfs[varl2->data->sgfn], cg->fgfs[varl3->data->sgfn]);
                else if (tindex == -1)
                    // just change data order to use average3
                    f_average3(cg->shape, cg->fgfs[varl2->data->sgfn], cg->fgfs[varl1->data->sgfn], cg->fgfs[varl3->data->sgfn]);
                else
                {
                    cout << "error tindex in Parallel::prepare_inter_time_level" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                varl1 = varl1->next;
                varl2 = varl2->next;
                varl3 = varl3->next;
            }
        }
        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }
}
void Parallel::prepare_inter_time_level(MyList<Patch> *PatL,
                                                                                MyList<var> *VarList1 /* source (t+dt) */, MyList<var> *VarList2 /* source (t) */,
                                                                                MyList<var> *VarList3 /* source (t-dt) */, MyList<var> *VarList4 /* target (t+a*dt) */, int tindex)
{
    while (PatL)
    {
        prepare_inter_time_level(PatL->data, VarList1, VarList2, VarList3, VarList4, tindex);
        PatL = PatL->next;
    }
}
void Parallel::prepare_inter_time_level(Patch *Pat,
                                                                                MyList<var> *VarList1 /* source (t+dt) */, MyList<var> *VarList2 /* source (t) */,
                                                                                MyList<var> *VarList3 /* source (t-dt) */, MyList<var> *VarList4 /* target (t+a*dt) */, int tindex)
{
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MyList<var> *varl1;
    MyList<var> *varl2;
    MyList<var> *varl3;
    MyList<var> *varl4;

    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *cg = BP->data;
        if (myrank == cg->rank)
        {
            varl1 = VarList1;
            varl2 = VarList2;
            varl3 = VarList3;
            varl4 = VarList4;
            while (varl1)
            {
                if (tindex == 0)
                    f_average2(cg->shape, cg->fgfs[varl1->data->sgfn], cg->fgfs[varl2->data->sgfn],
                                         cg->fgfs[varl3->data->sgfn], cg->fgfs[varl4->data->sgfn]);
                else if (tindex == 1)
                    f_average2p(cg->shape, cg->fgfs[varl1->data->sgfn], cg->fgfs[varl2->data->sgfn],
                                            cg->fgfs[varl3->data->sgfn], cg->fgfs[varl4->data->sgfn]);
                else if (tindex == -1)
                    f_average2m(cg->shape, cg->fgfs[varl1->data->sgfn], cg->fgfs[varl2->data->sgfn],
                                            cg->fgfs[varl3->data->sgfn], cg->fgfs[varl4->data->sgfn]);
                else
                {
                    cout << "error tindex in long cgh::prepare_inter_time_level" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                varl1 = varl1->next;
                varl2 = varl2->next;
                varl3 = varl3->next;
                varl4 = varl4->next;
            }
        }
        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }
}
void Parallel::Prolong(Patch *Patc, Patch *Patf,
                                             MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
                                             int Symmetry)
{
    if (Patc->lev >= Patf->lev)
    {
        cout << "Parallel::Prolong: meet requst of Prolong from lev#" << Patc->lev << " to lev#" << Patf->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_complete_gsl(Patf); // including ghost
    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl4(Patc, node, Symmetry);                   // - buffer - ghost - BD ghost
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfer(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
void Parallel::Restrict(MyList<Patch> *PatcL, MyList<Patch> *PatfL,
                                                MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
                                                int Symmetry)
{
    if (PatcL->data->lev >= PatfL->data->lev)
    {
        cout << "Parallel::Restrict: meet requst of Restrict from lev#" << PatfL->data->lev << " to lev#" << PatcL->data->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_complete_gsl(PatcL); // including ghost
    for (int node = 0; node < cpusize; node++)
    {
        // it seems bam always use this
        src[node] = build_owned_gsl(PatfL, node, 2, Symmetry); // - buffer - ghost
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfer(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
void Parallel::Restrict_after(MyList<Patch> *PatcL, MyList<Patch> *PatfL,
                                                            MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
                                                            int Symmetry)
{
    if (PatcL->data->lev >= PatfL->data->lev)
    {
        cout << "Parallel::Restrict: meet requst of Restrict from lev#" << PatfL->data->lev << " to lev#" << PatcL->data->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_complete_gsl(PatcL); // including ghost
    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl(PatfL, node, 3, Symmetry); // - ghost - BD ghost

        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfer(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
// for the same time level
void Parallel::OutBdLow2Hi(Patch *Patc, Patch *Patf,
                                                     MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
                                                     int Symmetry)
{
    if (Patc->lev >= Patf->lev)
    {
        cout << "Parallel::OutBdLow2Hi: meet requst of Prolong from lev#" << Patc->lev << " to lev#" << Patf->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_buffer_gsl(Patf); // buffer region only

    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl4(Patc, node, Symmetry);                   // - buffer - ghost - BD ghost
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfer(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
void Parallel::OutBdLow2Hi(MyList<Patch> *PatcL, MyList<Patch> *PatfL,
                                                     MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
                                                     int Symmetry)
{
    MyList<Patch> *Pp, *Ppc;
    Ppc = PatcL;
    while (Ppc)
    {
        Pp = PatfL;
        while (Pp)
        {
            if (Ppc->data->lev >= Pp->data->lev)
            {
                cout << "Parallel::OutBdLow2Hi(list): meet requst of Prolong from lev#" << Ppc->data->lev << " to lev#" << Pp->data->lev << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            Pp = Pp->next;
        }
        Ppc = Ppc->next;
    }

    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_buffer_gsl(PatfL); // buffer region only

    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl(PatcL, node, 4, Symmetry);                // - buffer - ghost - BD ghost
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfer(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
// for the same time level
void Parallel::OutBdLow2Himix(Patch *Patc, Patch *Patf,
                                                            MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
                                                            int Symmetry)
{
    if (Patc->lev >= Patf->lev)
    {
        cout << "Parallel::OutBdLow2Himix: meet requst of Prolong from lev#" << Patc->lev << " to lev#" << Patf->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_buffer_gsl(Patf); // buffer region only

    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl4(Patc, node, Symmetry);                   // - buffer - ghost - BD ghost
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfermix(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;

    // do not need this, we have done after calling of this routine in ProlongRestrict or RestrictProlong
    //    Sync(Patf,VarList2,Symmetry);  // fine level points may be not enough for interpolation
}
void Parallel::OutBdLow2Himix(MyList<Patch> *PatcL, MyList<Patch> *PatfL,
                                                            MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
                                                            int Symmetry)
{
    MyList<Patch> *Pp, *Ppc;
    Ppc = PatcL;
    while (Ppc)
    {
        Pp = PatfL;
        while (Pp)
        {
            if (Ppc->data->lev >= Pp->data->lev)
            {
                cout << "Parallel::OutBdLow2Himix(list): meet requst of Prolong from lev#" << Ppc->data->lev << " to lev#" << Pp->data->lev << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            Pp = Pp->next;
        }
        Ppc = Ppc->next;
    }

    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_buffer_gsl(PatfL); // buffer region only

    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl(PatcL, node, 4, Symmetry);                // - buffer - ghost - BD ghost
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfermix(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}
// collect all buffer grid segments or blocks for given patch
MyList<Parallel::gridseg> *Parallel::build_buffer_gsl(Patch *Pat)
{
    MyList<Parallel::gridseg> *cgsl, *gsc, *gsb;

    gsc = build_complete_gsl(Pat); // including ghost

    gsb = new MyList<Parallel::gridseg>;
    gsb->data = new Parallel::gridseg;

    for (int i = 0; i < dim; i++)
    {
        double DH = Pat->blb->data->getdX(i);
        gsb->data->uub[i] = Pat->bbox[dim + i] - Pat->uui[i] * DH;
        gsb->data->llb[i] = Pat->bbox[i] + Pat->lli[i] * DH;

        gsb->data->shape[i] = int((gsb->data->uub[i] - gsb->data->llb[i]) / DH + 0.4);

    }
    gsb->data->Bg = 0;
    gsb->next = 0;

    cgsl = gsl_subtract(gsc, gsb);

    gsc->destroyList();
    gsb->destroyList();

    //  set illb and iuub
    gsb = cgsl;
    while (gsb)
    {
        for (int i = 0; i < dim; i++)
        {
            double DH = Pat->blb->data->getdX(i);
            gsb->data->iuub[i] = Pat->bbox[dim + i] - Pat->uui[i] * DH;
            gsb->data->illb[i] = Pat->bbox[i] + Pat->lli[i] * DH;
        }
        gsb = gsb->next;
    }

    return cgsl;
}
MyList<Parallel::gridseg> *Parallel::build_buffer_gsl(MyList<Patch> *PatL)
{
    MyList<Parallel::gridseg> *cgsl = 0, *gs;
    while (PatL)
    {
        if (cgsl)
        {
            gs->next = build_buffer_gsl(PatL->data);
            gs = gs->next;
            if (gs)
                while (gs->next)
                    gs = gs->next;
        }
        else
        {
            cgsl = build_buffer_gsl(PatL->data);
            gs = cgsl;
            if (gs)
                while (gs->next)
                    gs = gs->next;
        }
        PatL = PatL->next;
    }

    return cgsl;
}
void Parallel::Prolongint(Patch *Patc, Patch *Patf,
                                                    MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
                                                    int Symmetry)
{
    if (Patc->lev >= Patf->lev)
    {
        cout << "Parallel::Prolong: meet requst of Prolong from lev#" << Patc->lev << " to lev#" << Patf->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int num_var = 0;
    MyList<var> *varl;
    varl = VarList1;
    while (varl)
    {
        num_var++;
        varl = varl->next;
    }

    MyList<Block> *BP = Patf->blb;
    while (BP)
    {
        int Npts;
        if (myrank == BP->data->rank)
            Npts = BP->data->shape[0] * BP->data->shape[1] * BP->data->shape[2];
        MPI_Bcast(&Npts, 1, MPI_INT, BP->data->rank, MPI_COMM_WORLD);
        double *pox[3];
        for (int i = 0; i < 3; i++)
            pox[i] = new double[Npts];
        if (myrank == BP->data->rank)
        {
            for (int i = 0; i < Npts; i++)
            {
                int ind[3];
                Parallel::getarrayindex(3, BP->data->shape, ind, i);
                pox[0][i] = BP->data->X[0][ind[0]];
                pox[1][i] = BP->data->X[1][ind[1]];
                pox[2][i] = BP->data->X[2][ind[2]];
            }
        }
        for (int i = 0; i < 3; i++)
            MPI_Bcast(pox[i], Npts, MPI_DOUBLE, BP->data->rank, MPI_COMM_WORLD);
        double *res;
        res = new double[num_var * Npts];
        Patc->Interp_Points(VarList1, Npts, pox, res, Symmetry); // because this operation is a global operation (for all processors)
                                                                                                                         // we have to isolate it out of myrank==BP->data->rank
        if (myrank == BP->data->rank)
        {
            for (int i = 0; i < Npts; i++)
            {
                varl = VarList2;
                int j = 0;
                while (varl)
                {
                    (BP->data->fgfs[varl->data->sgfn])[i] = res[j + i * num_var];
                    j++;
                    varl = varl->next;
                }
            }
        }
        delete[] pox[0];
        delete[] pox[1];
        delete[] pox[2];
        delete[] res;
        BP = BP->next;
    }
}
//
void Parallel::merge_gsl(MyList<gridseg> *&A, const double ratio)
{
    if (!A)
        return;

    MyList<gridseg> *B, *C, *D = A;
    bool flag = false;
    while (D->next)
    {
        B = D->next;
        while (B)
        {
            flag = merge_gs(D, B, C, ratio);
            if (flag)
                break;
            B = B->next;
        }
        if (flag)
            break;
        D = D->next;
    }

    if (flag)
    {
        // delete D and B from A
        MyList<gridseg> *E = A;
        while (E->next)
        {
            MyList<gridseg> *tp = E->next;
            if (D == tp || B == tp)
            {
                E->next = (tp->next) ? tp->next : 0;
                delete tp->data;
                delete tp;
            }
            if (E->next)
                E = E->next;
        }

        if (D == A)
        {
            MyList<gridseg> *tp = A;
            A = (A->next) ? A->next : 0;
            delete tp->data;
            delete tp;
        }
        // cat C to A
        if (A)
            A->catList(C);
        else
            A = C;

        merge_gsl(A, ratio);
    }
}
//
bool Parallel::merge_gs(MyList<gridseg> *D, MyList<gridseg> *B, MyList<gridseg> *&C, const double ratio)
{
    if (!B || !D)
        return false;

    C = 0;
    double llb[dim], uub[dim], DH[dim];
    for (int i = 0; i < dim; i++)
    {
        double tdh;
        DH[i] = (D->data->uub[i] - D->data->llb[i]) / D->data->shape[i];
        tdh = (B->data->uub[i] - B->data->llb[i]) / B->data->shape[i];
        if (!feq(DH[i], tdh, DH[i] / 2))
        {
            cout << "Parallel::merge_gs meets different grid segment " << DH[i] << " vs " << tdh << endl;
            checkgsl(B, true);
            checkgsl(D, true);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        llb[i] = Mymax(D->data->llb[i], B->data->llb[i]);
        uub[i] = Mymin(D->data->uub[i], B->data->uub[i]);
        //    if(uub[i]-llb[i] < DH[i]/2) return false;  //here this is valid for both vertex and cell

        // use 0 instead of DH[i]/2, we consider contact case, 2012 Aug 8
        if (uub[i] - llb[i] < 0)
            return false; // here this is valid for both vertex and cell
    }

    // vb: volume of B
    // vd: volume of D
    // vo: volume of overlap
    // vt: volume of smallest common box (virtual merged box)
    double vd = 1, vb = 1, vt = 1, vo = 1;
    for (int i = 0; i < dim; i++)
    {
        vt = vt * (Mymax(D->data->uub[i], B->data->uub[i]) - Mymin(D->data->llb[i], B->data->llb[i]));
        vo = vo * (uub[i] - llb[i]);
        vd = vd * (D->data->uub[i] - D->data->llb[i]);
        vb = vb * (B->data->uub[i] - B->data->llb[i]);
    }

    // smller ratio, more possible to merge
    if ((vd + vb - vo) / vt > ratio)
    {
        C = new MyList<gridseg>;
        C->data = new gridseg;
        for (int i = 0; i < dim; i++)
        {
            C->data->uub[i] = Mymax(D->data->uub[i], B->data->uub[i]);
            C->data->llb[i] = Mymin(D->data->llb[i], B->data->llb[i]);
            C->data->shape[i] = int((C->data->uub[i] - C->data->llb[i]) / DH[i] + 0.4);
        }
        if (D->data->Bg == B->data->Bg)
            C->data->Bg = D->data->Bg;
        else
            C->data->Bg = 0;

        C->next = 0;

        return true;
    }
    else
    {
        return false;
    }
}
// Add ghost region to tangent plane
// we assume the grids have the same resolution
void Parallel::add_ghost_touch(MyList<gridseg> *&A)
{
    if (!A || !(A->next))
        return;

    double DH[dim];
    for (int i = 0; i < dim; i++)
        DH[i] = (A->data->uub[i] - A->data->llb[i]) / A->data->shape[i] / 2;

    MyList<gridseg> *C1, *C2, *A1 = A, *A2, *dc;
    dc = C1 = clone_gsl(A, false);
    while (C1)
    {
        C2 = C1->next;
        A2 = A1->next;
        while (C2)
        {
            for (int i = 0; i < dim; i++)
            {
                if (feq(C1->data->llb[i], C2->data->uub[i], DH[i]))
                {
                    // direction i touch, other directions overlap
                    bool flag = true;
                    for (int j = 0; j < i; j++)
                        if ((C1->data->llb[j] - C2->data->llb[j]) * (C1->data->uub[j] - C2->data->llb[j]) > 0 &&
                                (C2->data->llb[j] - C1->data->llb[j]) * (C2->data->uub[j] - C1->data->llb[j]) > 0)
                            flag = false;
                    for (int j = i + 1; j < dim; j++)
                        if ((C1->data->llb[j] - C2->data->llb[j]) * (C1->data->uub[j] - C2->data->llb[j]) > 0 &&
                                (C2->data->llb[j] - C1->data->llb[j]) * (C2->data->uub[j] - C1->data->llb[j]) > 0)
                            flag = false;

                    if (flag)
                    {
                        // only add one ghost region
                        if (feq(A1->data->llb[i], C1->data->llb[i], DH[i]))
                        {
                            A1->data->llb[i] -= ghost_width * 2 * DH[i];
                            A1->data->shape[i] += ghost_width;
                        }
                        if (feq(A2->data->uub[i], C2->data->uub[i], DH[i]))
                        {
                            A2->data->uub[i] += ghost_width * 2 * DH[i];
                            A2->data->shape[i] += ghost_width;
                        }
                    }
                }
                if (feq(C1->data->uub[i], C2->data->llb[i], DH[i]))
                {
                    // direction i touch, other directions overlap
                    bool flag = true;
                    for (int j = 0; j < i; j++)
                        if ((C1->data->llb[j] - C2->data->llb[j]) * (C1->data->uub[j] - C2->data->llb[j]) > 0 &&
                                (C2->data->llb[j] - C1->data->llb[j]) * (C2->data->uub[j] - C1->data->llb[j]) > 0)
                            flag = false;
                    for (int j = i + 1; j < dim; j++)
                        if ((C1->data->llb[j] - C2->data->llb[j]) * (C1->data->uub[j] - C2->data->llb[j]) > 0 &&
                                (C2->data->llb[j] - C1->data->llb[j]) * (C2->data->uub[j] - C1->data->llb[j]) > 0)
                            flag = false;

                    if (flag)
                    {
                        // only add one ghost region
                        if (feq(A1->data->uub[i], C1->data->uub[i], DH[i]))
                        {
                            A1->data->uub[i] += ghost_width * 2 * DH[i];
                            A1->data->shape[i] += ghost_width;
                        }
                        if (feq(A2->data->llb[i], C2->data->llb[i], DH[i]))
                        {
                            A2->data->llb[i] -= ghost_width * 2 * DH[i];
                            A2->data->shape[i] += ghost_width;
                        }
                    }
                }
            }
            C2 = C2->next;
            A2 = A2->next;
        }
        C1 = C1->next;
        A1 = A1->next;
    }

    if (dc)
        dc->destroyList();
}
// According to overlap to cut the gsl into recular pices
void Parallel::cut_gsl(MyList<gridseg> *&A)
{
    if (!A)
        return;

    MyList<gridseg> *B, *C, *D = A;
    bool flag = false;
    while (D->next)
    {
        B = D->next;
        while (B)
        {
            flag = cut_gs(D, B, C);
            if (flag)
                break;
            B = B->next;
        }
        if (flag)
            break;
        D = D->next;
    }

    if (flag)
    {
        // delete D and B from A
        MyList<gridseg> *E = A;
        while (E->next)
        {
            MyList<gridseg> *tp = E->next;
            if (D == tp || B == tp)
            {
                E->next = (tp->next) ? tp->next : 0;
                delete tp->data;
                delete tp;
            }
            if (E->next)
                E = E->next;
        }

        if (D == A)
        {
            MyList<gridseg> *tp = A;
            A = (A->next) ? A->next : 0;
            delete tp->data;
            delete tp;
        }
        // cat C to A
        if (A)
            A->catList(C);
        else
            A = C;

        cut_gsl(A);
    }
}
// when D and B have overlap, cut them into C and return true
// otherwise return false and C=0
bool Parallel::cut_gs(MyList<gridseg> *D, MyList<gridseg> *B, MyList<gridseg> *&C)
{
    C = 0;
    double llb[dim], uub[dim], DH[dim];
    for (int i = 0; i < dim; i++)
    {
        double tdh;

        DH[i] = (D->data->uub[i] - D->data->llb[i]) / D->data->shape[i];
        tdh = (B->data->uub[i] - B->data->llb[i]) / B->data->shape[i];

        if (!feq(DH[i], tdh, DH[i] / 2))
        {
            cout << "Parallel::cut_gs meets different grid segment " << DH[i] << " vs " << tdh << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        llb[i] = Mymax(D->data->llb[i], B->data->llb[i]);
        uub[i] = Mymin(D->data->uub[i], B->data->uub[i]);
        // for efficiency we ask the width of the patch at least 2(buffer+ghost+BD ghost)
        if (uub[i] - llb[i] < DH[i] * 2 * (buffer_width + 2 * ghost_width))
            return false; // here this is valid for both vertex and cell
    }

    // this part code results in 5 patches generally

    C = new MyList<gridseg>;
    C->data = new gridseg;
    for (int i = 0; i < dim; i++)
    {
        C->data->llb[i] = llb[i];
        C->data->uub[i] = uub[i];
        C->data->shape[i] = int((C->data->uub[i] - C->data->llb[i]) / DH[i] + 0.4);
    }
    if (D->data->Bg == B->data->Bg)
        C->data->Bg = D->data->Bg;
    else
        C->data->Bg = 0;

    C->next = gs_subtract_virtual(D, C);

    MyList<gridseg> *E = C;

    while (E->next)
        E = E->next;

    E->next = gs_subtract_virtual(B, C);

    // this part code results in 3 patches generally
    /*
             C = clone_gsl(D,true);
             C->next = gs_subtract_virtual(B,C);
    */

    return true;
}
// note here it is different to real cut, we need leave the cutting edge for both vertex center and cell center
MyList<Parallel::gridseg> *Parallel::gs_subtract_virtual(MyList<Parallel::gridseg> *A, MyList<Parallel::gridseg> *B)
{
    if (!A)
        return 0;
    if (!B)
        return clone_gsl(A, true);

    double cut_plane[2 * dim], DH[dim];

    for (int i = 0; i < dim; i++)
    {
        double tdh;
        DH[i] = (A->data->uub[i] - A->data->llb[i]) / A->data->shape[i];
        tdh = (B->data->uub[i] - B->data->llb[i]) / B->data->shape[i];
        if (!feq(DH[i], tdh, DH[i] / 2))
        {
            cout << "Parallel::gs_subtract_virtual meets different grid segment " << DH[i] << " vs " << tdh << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MyList<Parallel::gridseg> *C = 0, *q;
    for (int i = 0; i < dim; i++)
    {
        if (B->data->llb[i] > A->data->uub[i] || B->data->uub[i] < A->data->llb[i])
            return clone_gsl(A, true);
        cut_plane[i] = A->data->llb[i];
        cut_plane[i + dim] = A->data->uub[i];
    }

    for (int i = 0; i < dim; i++)
    {
        cut_plane[i] = Mymax(A->data->llb[i], B->data->llb[i]);
        if (cut_plane[i] > A->data->llb[i])
        {
            q = clone_gsl(A, true);
            // prolong the list from head
            if (C)
                q->next = C;
            C = q;
            for (int j = 0; j < dim; j++)
            {
                if (i == j)
                {
                    C->data->llb[i] = A->data->llb[i];
                    // **note here it is different to real cut, we need leave the cutting edge for both vertex center and cell center**
                    C->data->uub[i] = Mymax(C->data->llb[i], cut_plane[i]);
                }
                else
                {
                    C->data->llb[j] = cut_plane[j];
                    C->data->uub[j] = cut_plane[j + dim];
                }
                C->data->shape[j] = int((C->data->uub[j] - C->data->llb[j]) / DH[j] + 0.4);
            }
        }

        cut_plane[i + dim] = Mymin(A->data->uub[i], B->data->uub[i]);
        if (cut_plane[i + dim] < A->data->uub[i])
        {
            q = clone_gsl(A, true);
            if (C)
                q->next = C;
            C = q;
            for (int j = 0; j < dim; j++)
            {
                if (i == j)
                {
                    C->data->uub[i] = A->data->uub[i];
                    // note here it is different to real cut, we need leave the cutting edge for both vertex center and cell center
                    C->data->llb[i] = Mymin(C->data->uub[i], cut_plane[i + dim]);
                }
                else
                {
                    C->data->llb[j] = cut_plane[j];
                    C->data->uub[j] = cut_plane[j + dim];
                }

                C->data->shape[j] = int((C->data->uub[j] - C->data->llb[j]) / DH[j] + 0.4);
            }
        }
    }
    return C;
}
// note the data structure
// if CC is true
// 1   -----------  1   ------  ^
//                  0   ------  |  t
// 0   -----------  old ------  |
//
// old -----------
// if CC is false
// 1   -----------  1   ------  ^
// 0   -----------  0   ------  |  t
// old -----------  old ------  |
void Parallel::fill_level_data(MyList<Patch> *PatLd, MyList<Patch> *PatLs, MyList<Patch> *PatcL,
                                                             MyList<var> *OldList, MyList<var> *StateList, MyList<var> *FutureList,
                                                             MyList<var> *tmList, int Symmetry, bool BB, bool CC)
{
    if (PatLd->data->lev != PatLs->data->lev)
    {
        cout << "Parallel::fill_level_data: meet requst from lev#" << PatLs->data->lev << " to lev#" << PatLd->data->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (PatLd->data->lev <= PatcL->data->lev)
    {
        cout << "Parallel::fill_level_data: meet prolong requst from lev#" << PatcL->data->lev << " to lev#" << PatLd->data->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<var> *VarList = 0;
    MyList<var> *p;
    p = StateList;
    while (p)
    {
        if (VarList)
            VarList->insert(p->data);
        else
            VarList = new MyList<var>(p->data);
        p = p->next;
    }
    p = FutureList;
    while (p)
    {
        if (VarList)
            VarList->insert(p->data);
        else
            VarList = new MyList<var>(p->data);
        p = p->next;
    }

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_complete_gsl(PatLd); // including ghost
    // copy part
    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl(PatLs, node, 0, Symmetry);                // similar to Sync
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    transfer(transfer_src, transfer_dst, VarList, VarList, Symmetry);

    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    MyList<Parallel::gridseg> *dsts, *dstd;
    dsts = build_complete_gsl_virtual(PatLs);
    dstd = dst;
    dst = gsl_subtract(dstd, dsts);
    if (dstd)
        dstd->destroyList();
    if (dsts)
        dsts->destroyList();

    if (dst)
    {
        // prolongation part
        for (int node = 0; node < cpusize; node++)
        {
            src[node] = build_owned_gsl(PatcL, node, 4, Symmetry);                // - buffer - ghost - BD ghost
            build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
        }

        if (CC)
        {
            // for FutureList
            // restrict first~~~>
            {
                Restrict(PatcL, PatLs, FutureList, FutureList, Symmetry);
                Sync(PatcL, FutureList, Symmetry);
            }
            //<~~~prolong then
            transfer(transfer_src, transfer_dst, FutureList, FutureList, Symmetry);

            // for StateList
            // time interpolation part
            if (BB)
                prepare_inter_time_level(PatcL, FutureList, StateList, OldList,
                                                                 tmList, 0); // use SynchList_pre as temporal storage space
            else
                prepare_inter_time_level(PatcL, FutureList, StateList,
                                                                 tmList, 0); // use SynchList_pre as temporal storage space
                                                                                         // restrict first~~~>
            {
                Restrict(PatcL, PatLs, StateList, tmList, Symmetry);
                Sync(PatcL, tmList, Symmetry);
            }
            //<~~~prolong then
            transfer(transfer_src, transfer_dst, tmList, StateList, Symmetry);
        }
        else
        {
            // for both FutureList and StateList
            // restrict first~~~>
            {
                Restrict(PatcL, PatLs, VarList, VarList, Symmetry);
                Sync(PatcL, VarList, Symmetry);
            }
            //<~~~prolong then
            transfer(transfer_src, transfer_dst, VarList, VarList, Symmetry);
        }

        for (int node = 0; node < cpusize; node++)
        {
            if (src[node])
                src[node]->destroyList();
            if (transfer_src[node])
                transfer_src[node]->destroyList();
            if (transfer_dst[node])
                transfer_dst[node]->destroyList();
        }

        dst->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;

    VarList->clearList();
}
void Parallel::KillBlocks(MyList<Patch> *PatchLIST)
{
    while (PatchLIST)
    {
        Patch *Pp = PatchLIST->data;
        MyList<Block> *bg;
        while (Pp->blb)
        {
            if (Pp->blb == Pp->ble)
                break;
            bg = (Pp->blb->next) ? Pp->blb->next : 0;
            delete Pp->blb->data;
            delete Pp->blb;
            Pp->blb = bg;
        }
        if (Pp->ble)
        {
            delete Pp->ble->data;
            delete Pp->ble;
        }
        Pp->blb = Pp->ble = 0;
        PatchLIST = PatchLIST->next;
    }
}
bool Parallel::PatList_Interp_Points(MyList<Patch> *PatL, MyList<var> *VarList,
                                                                         int NN, double **XX,
                                                                         double *Shellf, int Symmetry)
{
    MyList<var> *varl;
    int num_var = 0;
    varl = VarList;
    while (varl)
    {
        num_var++;
        varl = varl->next;
    }

    double lld[dim], uud[dim];
    double **pox;
    pox = new double *[dim];
    for (int j = 0; j < dim; j++)
        pox[j] = new double[1];
    for (int i = 0; i < NN; i++)
    {
        MyList<Patch> *PL = PatL;
        while (PL)
        {
            bool flag = true;
            for (int j = 0; j < dim; j++)
            {
                double h = PL->data->getdX(j);
                lld[j] = PL->data->lli[j] * h;
                uud[j] = PL->data->uui[j] * h;
                if (XX[j][i] < PL->data->bbox[j] + lld[j] || XX[j][i] > PL->data->bbox[j + dim] - uud[j])
                {
                    flag = false;
                    break;
                }
                pox[j][0] = XX[j][i];
            }
            if (flag)
            {
                PL->data->Interp_Points(VarList, 1, pox, Shellf + i * num_var, Symmetry);
                break;
            }
            PL = PL->next;
        }
        if (!PL)
        {
            checkpatchlist(PatL, false);
            return false;
        }
    }
    for (int j = 0; j < dim; j++)
        delete[] pox[j];
    delete[] pox;

    return true;
}
bool Parallel::PatList_Interp_Points(MyList<Patch> *PatL, MyList<var> *VarList,
                                                                         int NN, double **XX,
                                                                         double *Shellf, int Symmetry, MPI_Comm Comm_here)
{
    MyList<var> *varl;
    int num_var = 0;
    varl = VarList;
    while (varl)
    {
        num_var++;
        varl = varl->next;
    }

    double lld[dim], uud[dim];
    double **pox;
    pox = new double *[dim];
    for (int j = 0; j < dim; j++)
        pox[j] = new double[1];
    for (int i = 0; i < NN; i++)
    {
        MyList<Patch> *PL = PatL;
        while (PL)
        {
            bool flag = true;
            for (int j = 0; j < dim; j++)
            {
                double h = PL->data->getdX(j);
                lld[j] = PL->data->lli[j] * h;
                uud[j] = PL->data->uui[j] * h;
                if (XX[j][i] < PL->data->bbox[j] + lld[j] || XX[j][i] > PL->data->bbox[j + dim] - uud[j])
                {
                    flag = false;
                    break;
                }
                pox[j][0] = XX[j][i];
            }
            if (flag)
            {
                PL->data->Interp_Points(VarList, 1, pox, Shellf + i * num_var, Symmetry, Comm_here);
                break;
            }
            PL = PL->next;
        }
        if (!PL)
        {
            checkpatchlist(PatL, false);
            return false;
        }
    }
    for (int j = 0; j < dim; j++)
        delete[] pox[j];
    delete[] pox;

    return true;
}
void Parallel::aligncheck(double *bbox0, double *bboxl, int lev, double *DH0, int *shape)
{
    const double aligntiny = 0.1;
    double DHl, rr;
    int NN;
    for (int i = 0; i < dim; i++)
    {
        DHl = DH0[i] * pow(0.5, lev);
        rr = bboxl[i] - bbox0[i];
        bboxl[i] = bbox0[i] + int(rr / DHl + 0.4) * DHl;
        rr = bbox0[i + dim] - bboxl[i + dim];
        bboxl[i + dim] = bbox0[i + dim] - int(rr / DHl + 0.4) * DHl;

        NN = int((bboxl[i + dim] - bboxl[i]) / DHl + 0.4);

        if (NN != shape[i])
        {
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            if (myrank == 0)
            {
                cout << "Parallel::aligncheck want shape " << NN << " for lev#" << lev << ", but " << shape[i] << endl;
                cout << "i = " << i << ", low = " << bboxl[i] << ", up = " << bboxl[i + dim] << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
}
bool Parallel::point_locat_gsl(double *pox, MyList<Parallel::gridseg> *gsl)
{
    bool flag = false;
    while (gsl)
    {
        for (int i = 0; i < dim; i++)
        {
            if (pox[i] > gsl->data->llb[i] && pox[i] < gsl->data->uub[i])
                flag = true;
            else
            {
                flag = false;
                break;
            }
        }
        if (flag)
            break;
        gsl = gsl->next;
    }

    return flag;
}
void Parallel::checkpatchlist(MyList<Patch> *PatL, bool buflog)
{
    MyList<Patch> *PL = PatL;
    while (PL)
    {
        PL->data->checkPatch(buflog);
        PL = PL->next;
    }
}
