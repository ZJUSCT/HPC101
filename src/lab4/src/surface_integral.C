
//----------------------------------------------------------------
// Using Gauss-Legendre quadrature in theta direction
// and   trapezoidal rule in phi direction (from Second Euler-Maclaurin summation formula, we can see that
// this method gives expolential convergence for periodic function)
//----------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <fstream>
#include <strstream>
#include <cmath>
#include <map>
using namespace std;
#include <mpi.h>

#include "misc.h"
#include "cgh.h"
#include "Parallel.h"
#include "surface_integral.h"
#include "fadmquantites_bssn.h"
#include "getnp4.h"
#include "parameters.h"

#ifdef USE_GPU
#include "gpu_manager.h"
#include "helper.h"
#endif

#define PI M_PI
//|============================================================================
//| Constructor
//|============================================================================

surface_integral::surface_integral(int iSymmetry) : Symmetry(iSymmetry) {
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
    int N = 40;
    // read parameter from file
    {
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
        if (!inf.good() && myrank == 0)
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

            if (sgrp == "SurfaceIntegral")
            {
                if (skey == "number of points for quarter sphere")
                    N = atoi(sval.c_str());
            }
        }
        inf.close();
    }
    //|-----number of points for whole [0,pi] x [0,2pi]
    N_phi   = 4 * N;   // for simplicity, we require this number must be 4*N
    N_theta = 2 * N;   //                                                2*N

    if (myrank == 0)
    {
        cout << "-----------------------------------------------------------------------" << endl;
        cout << " spherical integration for wave form extraction with Gauss method      " << endl;
        cout << " N_phi   = " << N_phi   << endl;
        cout << " N_theta = " << N_theta << endl;
        cout << "-----------------------------------------------------------------------" << endl;
    }

    //  weight function cover all of [0,pi]
    arcostheta = new double[N_theta];
    wtcostheta = new double[N_theta];

#ifdef USE_GPU
    stream = GPUManager::getInstance().get_stream();

    d_arcostheta = GPUManager::getInstance().allocate_device_memory(N_theta);
    d_wtcostheta = GPUManager::getInstance().allocate_device_memory(N_theta);
#endif

    // note: theta in [0,pi/2], upper half sphere, corresponds to 1 < costheta < 0
    misc::gaulegf(-1.0, 1.0, arcostheta, wtcostheta, N_theta);
    // due to symmetry, I need first half array corresponds to upper sphere, note these two arrays must match each other
    misc::inversearray(arcostheta, N_theta);
    misc::inversearray(wtcostheta, N_theta);
#ifdef USE_GPU
    GPUManager::getInstance().sync_to_gpu(arcostheta, d_arcostheta, N_theta);
    GPUManager::getInstance().sync_to_gpu(wtcostheta, d_wtcostheta, N_theta);
#endif

    if (Symmetry == 2)
    {
        N_phi = N_phi / 4;
        N_theta = N_theta / 2;
        dphi = PI / (2.0 * N_phi);
        dcostheta = 1.0 / N_theta;
        factor = 8;
    }
    else if (Symmetry == 1)
    {
        N_theta = N_theta / 2;
        dphi = 2.0 * PI / N_phi;
        dcostheta = 1.0 / N_theta;
        factor = 2;
    }
    else if (Symmetry == 0)
    {
        dphi = 2.0 * PI / N_phi;
        dcostheta = 2.0 / N_theta;
        factor = 1;
    }
    else if (myrank == 0)
    {
        cout << "surface_integral::surface_integral: not supported Symmetry setting!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    n_tot = N_theta * N_phi;
    nx_g = new double[n_tot];
    ny_g = new double[n_tot];
    nz_g = new double[n_tot];
#ifdef USE_GPU
    d_nx_g = GPUManager::getInstance().allocate_device_memory(n_tot);
    d_ny_g = GPUManager::getInstance().allocate_device_memory(n_tot);
    d_nz_g = GPUManager::getInstance().allocate_device_memory(n_tot);
#endif

    int n = 0;
    double costheta, sintheta, ph;

    for (int i = 0; i < N_theta; ++i) {
        costheta = arcostheta[i];
        sintheta = sqrt(1.0 - costheta * costheta);

        for (int j = 0; j < N_phi; ++j) {
            ph = (j + 0.5) * dphi;
            // normal vector respect to the constant R sphere
            nx_g[n] = sintheta * cos(ph);
            ny_g[n] = sintheta * sin(ph);
            nz_g[n] = costheta;
            n++;
        }
    }
#ifdef USE_GPU
    GPUManager::getInstance().sync_to_gpu(nx_g, d_nx_g, n_tot);
    GPUManager::getInstance().sync_to_gpu(ny_g, d_ny_g, n_tot);
    GPUManager::getInstance().sync_to_gpu(nz_g, d_nz_g, n_tot);
#endif
}

//|============================================================================
//| Destructor
//|============================================================================
surface_integral::~surface_integral() {
    delete[] nx_g;
    delete[] ny_g;
    delete[] nz_g;
    delete[] arcostheta;
    delete[] wtcostheta;
#ifdef USE_GPU
    GPUManager::getInstance().free_device_memory(d_arcostheta, N_theta);
    GPUManager::getInstance().free_device_memory(d_wtcostheta, N_theta);
    GPUManager::getInstance().free_device_memory(d_nx_g, n_tot);
    GPUManager::getInstance().free_device_memory(d_ny_g, n_tot);
    GPUManager::getInstance().free_device_memory(d_nz_g, n_tot);
#endif
}
//|----------------------------------------------------------------
//  spin weighted spinw component of psi4, general routine
//  l takes from spinw to maxl; m takes from -l to l
//|----------------------------------------------------------------
void surface_integral::surf_Wave(double rex, int lev, cgh *GH, var *Rpsi4, var *Ipsi4,
                                                                 int spinw, int maxl, int NN, double *RP, double *IP,
                                                                 monitor *Monitor) // NN is the length of RP and IP
{
    if (myrank == 0 && GH->grids[lev] != 1)
        if (Monitor->outfile)
            Monitor->outfile << "WARNING: surface integral on multipatches" << endl;
        else
            cout << "WARNING: surface integral on multipatches" << endl;

    const int InList = 2;

    MyList<var> *DG_List = new MyList<var>(Rpsi4);
    DG_List->insert(Ipsi4);

    int n;
    double *pox[3];
    for (int i = 0; i < 3; i++)
        pox[i] = new double[n_tot];
    for (n = 0; n < n_tot; n++)
    {
        pox[0][n] = rex * nx_g[n];
        pox[1][n] = rex * ny_g[n];
        pox[2][n] = rex * nz_g[n];
    }

    double *shellf;
    shellf = new double[n_tot * InList];

#ifdef USE_GPU
    Helper::move_to_gpu_whole(GH->PatL[lev], myrank, DG_List);
#endif
    GH->PatL[lev]->data->Interp_Points(DG_List, n_tot, pox, shellf, Symmetry);

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

    //|~~~~~> Integrate the dot product of Dphi with the surface normal.

    double *RP_out, *IP_out;
    RP_out = new double[NN];
    IP_out = new double[NN];

    for (int ii = 0; ii < NN; ii++)
    {
        RP_out[ii] = 0;
        IP_out[ii] = 0;
    }
    // theta part
    double costheta, thetap;
    double cosmphi, sinmphi;

    int i, j;
    int lpsy = 0;
    if (Symmetry == 0)
        lpsy = 1;
    else if (Symmetry == 1)
        lpsy = 2;
    else if (Symmetry == 2)
        lpsy = 8;

    double psi4RR, psi4II;
    for (n = Nmin; n <= Nmax; n++)
    {
        //       need round off always
        i = int(n / N_phi); // int(1.723) = 1, int(-1.732) = -1
        j = n - i * N_phi;

        int countlm = 0;
        for (int pl = spinw; pl < maxl + 1; pl++)
            for (int pm = -pl; pm < pl + 1; pm++)
            {
                for (int lp = 0; lp < lpsy; lp++)
                {
                    switch (lp)
                    {
                    case 0: //+++ (theta, phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = sin(pm * (j + 0.5) * dphi);
                        psi4RR = shellf[InList * n];
                        psi4II = shellf[InList * n + 1];
                        break;
                    case 1: //++- (pi-theta, phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = sin(pm * (j + 0.5) * dphi);
                        psi4RR = Rpsi4->SoA[2] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[2] * shellf[InList * n + 1];
                        break;
                    case 2: //+-+ (theta, 2*pi-phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = -sin(pm * (j + 0.5) * dphi);
                        psi4RR = Rpsi4->SoA[1] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[1] * shellf[InList * n + 1];
                        break;
                    case 3: //+-- (pi-theta, 2*pi-phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = -sin(pm * (j + 0.5) * dphi);
                        psi4RR = Rpsi4->SoA[2] * Rpsi4->SoA[1] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[2] * Ipsi4->SoA[1] * shellf[InList * n + 1];
                        break;
                    case 4: //-++ (theta, pi-phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (PI - (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI - (j + 0.5) * dphi));
                        psi4RR = Rpsi4->SoA[0] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[0] * shellf[InList * n + 1];
                        break;
                    case 5: //-+- (pi-theta, pi-phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (PI - (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI - (j + 0.5) * dphi));
                        psi4RR = Rpsi4->SoA[2] * Rpsi4->SoA[0] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[2] * Ipsi4->SoA[0] * shellf[InList * n + 1];
                        break;
                    case 6: //--+ (theta, pi+phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (PI + (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI + (j + 0.5) * dphi));
                        psi4RR = Rpsi4->SoA[1] * Rpsi4->SoA[0] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[1] * Ipsi4->SoA[0] * shellf[InList * n + 1];
                        break;
                    case 7: //--- (pi-theta, pi+phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (PI + (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI + (j + 0.5) * dphi));
                        psi4RR = Rpsi4->SoA[2] * Rpsi4->SoA[1] * Rpsi4->SoA[0] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[2] * Ipsi4->SoA[1] * Ipsi4->SoA[0] * shellf[InList * n + 1];
                    }

                    thetap = sqrt((2 * pl + 1.0) / 4.0 / PI) * misc::Wigner_d_function(pl, pm, spinw, costheta); // note the variation from -2 to 2
                    // wtcostheta is even function respect costheta
                    RP_out[countlm] = RP_out[countlm] + thetap * (psi4RR * cosmphi + psi4II * sinmphi) * wtcostheta[i];
                    IP_out[countlm] = IP_out[countlm] + thetap * (psi4II * cosmphi - psi4RR * sinmphi) * wtcostheta[i];
                }
                countlm++; // no sanity check for countlm and NN which should be noted in the input parameters
            }
    }

    for (int ii = 0; ii < NN; ii++)
    {
        RP_out[ii] = RP_out[ii] * rex * dphi;
        IP_out[ii] = IP_out[ii] * rex * dphi;
    }
    //|------+  Communicate and sum the results from each processor.

    MPI_Allreduce(RP_out, RP, NN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(IP_out, IP, NN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //|------= Free memory.

    delete[] pox[0];
    delete[] pox[1];
    delete[] pox[2];
    delete[] shellf;
    delete[] RP_out;
    delete[] IP_out;
    DG_List->clearList();
}
void surface_integral::surf_Wave(double rex, int lev, cgh *GH, var *Rpsi4, var *Ipsi4,
                                                                 int spinw, int maxl, int NN, double *RP, double *IP,
                                                                 monitor *Monitor, MPI_Comm Comm_here) // NN is the length of RP and IP
{
    //   misc::tillherecheck(GH->Commlev[lev],GH->start_rank[lev],"start surface_integral::surf_Wave");

    int lmyrank;
    MPI_Comm_rank(Comm_here, &lmyrank);
    if (lmyrank == 0 && GH->grids[lev] != 1)
        if (Monitor->outfile)
            Monitor->outfile << "WARNING: surface integral on multipatches" << endl;
        else
            cout << "WARNING: surface integral on multipatches" << endl;

    const int InList = 2;

    MyList<var> *DG_List = new MyList<var>(Rpsi4);
    DG_List->insert(Ipsi4);

    int n;
    double *pox[3];
    for (int i = 0; i < 3; i++)
        pox[i] = new double[n_tot];
    for (n = 0; n < n_tot; n++)
    {
        pox[0][n] = rex * nx_g[n];
        pox[1][n] = rex * ny_g[n];
        pox[2][n] = rex * nz_g[n];
    }

    double *shellf;
    shellf = new double[n_tot * InList];

    //    misc::tillherecheck(GH->Commlev[lev],GH->start_rank[lev],"before Interp_Points");

    GH->PatL[lev]->data->Interp_Points(DG_List, n_tot, pox, shellf, Symmetry, Comm_here);

    //    misc::tillherecheck(GH->Commlev[lev],GH->start_rank[lev],"after Interp_Points");

    int mp, Lp, Nmin, Nmax;

    int cpusize_here;
    MPI_Comm_size(Comm_here, &cpusize_here);

    mp = n_tot / cpusize_here;
    Lp = n_tot - cpusize_here * mp;

    if (Lp > lmyrank)
    {
        Nmin = lmyrank * mp + lmyrank;
        Nmax = Nmin + mp;
    }
    else
    {
        Nmin = lmyrank * mp + Lp;
        Nmax = Nmin + mp - 1;
    }

    //|~~~~~> Integrate the dot product of Dphi with the surface normal.

    double *RP_out, *IP_out;
    RP_out = new double[NN];
    IP_out = new double[NN];

    for (int ii = 0; ii < NN; ii++)
    {
        RP_out[ii] = 0;
        IP_out[ii] = 0;
    }
    // theta part
    double costheta, thetap;
    double cosmphi, sinmphi;

    int i, j;
    int lpsy = 0;
    if (Symmetry == 0)
        lpsy = 1;
    else if (Symmetry == 1)
        lpsy = 2;
    else if (Symmetry == 2)
        lpsy = 8;

    double psi4RR, psi4II;
    for (n = Nmin; n <= Nmax; n++)
    {
        //       need round off always
        i = int(n / N_phi); // int(1.723) = 1, int(-1.732) = -1
        j = n - i * N_phi;

        int countlm = 0;
        for (int pl = spinw; pl < maxl + 1; pl++)
            for (int pm = -pl; pm < pl + 1; pm++)
            {
                for (int lp = 0; lp < lpsy; lp++)
                {
                    switch (lp)
                    {
                    case 0: //+++ (theta, phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = sin(pm * (j + 0.5) * dphi);
                        psi4RR = shellf[InList * n];
                        psi4II = shellf[InList * n + 1];
                        break;
                    case 1: //++- (pi-theta, phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = sin(pm * (j + 0.5) * dphi);
                        psi4RR = Rpsi4->SoA[2] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[2] * shellf[InList * n + 1];
                        break;
                    case 2: //+-+ (theta, 2*pi-phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = -sin(pm * (j + 0.5) * dphi);
                        psi4RR = Rpsi4->SoA[1] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[1] * shellf[InList * n + 1];
                        break;
                    case 3: //+-- (pi-theta, 2*pi-phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = -sin(pm * (j + 0.5) * dphi);
                        psi4RR = Rpsi4->SoA[2] * Rpsi4->SoA[1] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[2] * Ipsi4->SoA[1] * shellf[InList * n + 1];
                        break;
                    case 4: //-++ (theta, pi-phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (PI - (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI - (j + 0.5) * dphi));
                        psi4RR = Rpsi4->SoA[0] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[0] * shellf[InList * n + 1];
                        break;
                    case 5: //-+- (pi-theta, pi-phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (PI - (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI - (j + 0.5) * dphi));
                        psi4RR = Rpsi4->SoA[2] * Rpsi4->SoA[0] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[2] * Ipsi4->SoA[0] * shellf[InList * n + 1];
                        break;
                    case 6: //--+ (theta, pi+phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (PI + (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI + (j + 0.5) * dphi));
                        psi4RR = Rpsi4->SoA[1] * Rpsi4->SoA[0] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[1] * Ipsi4->SoA[0] * shellf[InList * n + 1];
                        break;
                    case 7: //--- (pi-theta, pi+phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (PI + (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI + (j + 0.5) * dphi));
                        psi4RR = Rpsi4->SoA[2] * Rpsi4->SoA[1] * Rpsi4->SoA[0] * shellf[InList * n];
                        psi4II = Ipsi4->SoA[2] * Ipsi4->SoA[1] * Ipsi4->SoA[0] * shellf[InList * n + 1];
                    }

                    thetap = sqrt((2 * pl + 1.0) / 4.0 / PI) * misc::Wigner_d_function(pl, pm, spinw, costheta); // note the variation from -2 to 2
#ifdef GaussInt
                    // wtcostheta is even function respect costheta
                    RP_out[countlm] = RP_out[countlm] + thetap * (psi4RR * cosmphi + psi4II * sinmphi) * wtcostheta[i];
                    IP_out[countlm] = IP_out[countlm] + thetap * (psi4II * cosmphi - psi4RR * sinmphi) * wtcostheta[i];
#else
                    RP_out[countlm] = RP_out[countlm] + thetap * (psi4RR * cosmphi + psi4II * sinmphi);
                    IP_out[countlm] = IP_out[countlm] + thetap * (psi4II * cosmphi - psi4RR * sinmphi);
#endif
                }
                countlm++; // no sanity check for countlm and NN which should be noted in the input parameters
            }
    }

    for (int ii = 0; ii < NN; ii++)
    {
#ifdef GaussInt
        RP_out[ii] = RP_out[ii] * rex * dphi;
        IP_out[ii] = IP_out[ii] * rex * dphi;
#else
        RP_out[ii] = RP_out[ii] * rex * dphi * dcostheta;
        IP_out[ii] = IP_out[ii] * rex * dphi * dcostheta;
#endif
    }
    //|------+  Communicate and sum the results from each processor.

    MPI_Allreduce(RP_out, RP, NN, MPI_DOUBLE, MPI_SUM, Comm_here);
    MPI_Allreduce(IP_out, IP, NN, MPI_DOUBLE, MPI_SUM, Comm_here);

    //|------= Free memory.

    delete[] pox[0];
    delete[] pox[1];
    delete[] pox[2];
    delete[] shellf;
    delete[] RP_out;
    delete[] IP_out;
    DG_List->clearList();
}
void surface_integral::surf_Wave(double rex, int lev, cgh *GH,
                                                                 var *Ex, var *Ey, var *Ez, var *Bx, var *By, var *Bz,
                                                                 var *chi, var *gxx, var *gxy, var *gxz, var *gyy, var *gyz, var *gzz,
                                                                 int spinw, int maxl, int NN, double *RP, double *IP,
                                                                 monitor *Monitor,
                                                                 void (*funcs)(double &, double &, double &,
                                                                                             double &, double &, double &, double &, double &, double &, double &,
                                                                                             double &, double &, double &, double &, double &, double &,
                                                                                             double &, double &)) // NN is the length of RP and IP
{
    const int InList = 13;

    MyList<var> *DG_List = new MyList<var>(Ex);
    DG_List->insert(Ey);
    DG_List->insert(Ez);
    DG_List->insert(Bx);
    DG_List->insert(By);
    DG_List->insert(Bz);
    DG_List->insert(chi);
    DG_List->insert(gxx);
    DG_List->insert(gxy);
    DG_List->insert(gxz);
    DG_List->insert(gyy);
    DG_List->insert(gyz);
    DG_List->insert(gzz);

    int n;
    double *pox[3];
    for (int i = 0; i < 3; i++)
        pox[i] = new double[n_tot];
    for (n = 0; n < n_tot; n++)
    {
        pox[0][n] = rex * nx_g[n];
        pox[1][n] = rex * ny_g[n];
        pox[2][n] = rex * nz_g[n];
    }

    double *shellf;
    shellf = new double[n_tot * InList];

    GH->PatL[lev]->data->Interp_Points(DG_List, n_tot, pox, shellf, Symmetry);

    double *RP_out, *IP_out;
    RP_out = new double[NN];
    IP_out = new double[NN];

    for (int ii = 0; ii < NN; ii++)
    {
        RP_out[ii] = 0;
        IP_out[ii] = 0;
    }

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

    // theta part
    double costheta, thetap;
    double cosmphi, sinmphi;

    int i, j;
    int lpsy = 0;
    if (Symmetry == 0)
        lpsy = 1;
    else if (Symmetry == 1)
        lpsy = 2;
    else if (Symmetry == 2)
        lpsy = 8;

    double psi4RR, psi4II;
    double px, py, pz;
    double pEx, pEy, pEz, pBx, pBy, pBz;
    double pchi, pgxx, pgxy, pgxz, pgyy, pgyz, pgzz;
    for (n = Nmin; n <= Nmax; n++)
    {
        //       need round off always
        i = int(n / N_phi); // int(1.723) = 1, int(-1.732) = -1
        j = n - i * N_phi;

        int countlm = 0;
        for (int pl = spinw; pl < maxl + 1; pl++)
            for (int pm = -pl; pm < pl + 1; pm++)
            {
                for (int lp = 0; lp < lpsy; lp++)
                {
                    px = pox[0][n];
                    py = pox[1][n];
                    pz = pox[2][n];
                    pEx = shellf[InList * n];
                    pEy = shellf[InList * n + 1];
                    pEz = shellf[InList * n + 2];
                    pBx = shellf[InList * n + 3];
                    pBy = shellf[InList * n + 4];
                    pBz = shellf[InList * n + 5];
                    pchi = shellf[InList * n + 6];
                    pgxx = shellf[InList * n + 7];
                    pgxy = shellf[InList * n + 8];
                    pgxz = shellf[InList * n + 9];
                    pgyy = shellf[InList * n + 10];
                    pgyz = shellf[InList * n + 11];
                    pgzz = shellf[InList * n + 12];
                    switch (lp)
                    {
                    case 0: //+++ (theta, phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = sin(pm * (j + 0.5) * dphi);
                        break;
                    case 1: //++- (pi-theta, phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = sin(pm * (j + 0.5) * dphi);
                        pz = -pz;
                        pEz = -pEz;
                        pBx = -pBx;
                        pBy = -pBy;
                        pgxz = -pgxz;
                        pgyz = -pgyz;
                        break;
                    case 2: //+-+ (theta, 2*pi-phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = -sin(pm * (j + 0.5) * dphi);
                        py = -py;
                        pEy = -pEy;
                        pBx = -pBx;
                        pBz = -pBz;
                        pgxy = -pgxy;
                        pgyz = -pgyz;
                        break;
                    case 3: //+-- (pi-theta, 2*pi-phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (j + 0.5) * dphi);
                        sinmphi = -sin(pm * (j + 0.5) * dphi);
                        py = -py;
                        pz = -pz;
                        pEz = -pEz;
                        pBz = -pBz;
                        pgxz = -pgxz;
                        pEy = -pEy;
                        pBy = -pBy;
                        pgxy = -pgxy;
                        break;
                    case 4: //-++ (theta, pi-phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (PI - (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI - (j + 0.5) * dphi));
                        px = -px;
                        pEx = -pEx;
                        pBy = -pBy;
                        pBz = -pBz;
                        pgxy = -pgxy;
                        pgxz = -pgxz;
                        break;
                    case 5: //-+- (pi-theta, pi-phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (PI - (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI - (j + 0.5) * dphi));
                        pz = -pz;
                        px = -px;
                        pEz = -pEz;
                        pBz = -pBz;
                        pgyz = -pgyz;
                        pEx = -pEx;
                        pBx = -pBx;
                        pgxy = -pgxy;
                        break;
                    case 6: //--+ (theta, pi+phi)
                        costheta = arcostheta[i];
                        cosmphi = cos(pm * (PI + (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI + (j + 0.5) * dphi));
                        px = -px;
                        py = -py;
                        pEx = -pEx;
                        pBx = -pBx;
                        pgxz = -pgxz;
                        pEy = -pEy;
                        pBy = -pBy;
                        pgyz = -pgyz;
                        break;
                    case 7: //--- (pi-theta, pi+phi)
                        costheta = -arcostheta[i];
                        cosmphi = cos(pm * (PI + (j + 0.5) * dphi));
                        sinmphi = sin(pm * (PI + (j + 0.5) * dphi));
                        px = -px;
                        py = -py;
                        pz = -pz;
                        pEx = -pEx;
                        pEy = -pEy;
                        pEz = -pEz;
                    }

                    funcs(px, py, pz, pchi, pgxx, pgxy, pgxz, pgyy, pgyz, pgzz, pEx, pEy, pEz, pBx, pBy, pBz,
                                psi4RR, psi4II);
                    thetap = sqrt((2 * pl + 1.0) / 4.0 / PI) * misc::Wigner_d_function(pl, pm, spinw, costheta); // note the variation from -2 to 2

                    //	 find back the one
                    pchi = pchi + 1;

                    // wtcostheta is even function respect costheta
                    RP_out[countlm] = RP_out[countlm] + thetap / pchi / pchi * (psi4RR * cosmphi + psi4II * sinmphi) * wtcostheta[i];
                    IP_out[countlm] = IP_out[countlm] + thetap / pchi / pchi * (psi4II * cosmphi - psi4RR * sinmphi) * wtcostheta[i];

                }
                countlm++; // no sanity check for countlm and NN which should be noted in the input parameters
            }
    }

    for (int ii = 0; ii < NN; ii++)
    {
        RP_out[ii] = RP_out[ii] * rex * dphi;
        IP_out[ii] = IP_out[ii] * rex * dphi;

    }
    //|------+  Communicate and sum the results from each processor.

    MPI_Allreduce(RP_out, RP, NN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(IP_out, IP, NN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //|------= Free memory.

    delete[] pox[0];
    delete[] pox[1];
    delete[] pox[2];
    delete[] shellf;
    delete[] RP_out;
    delete[] IP_out;
    DG_List->clearList();
}
//|----------------------------------------------------
//|
//| ADM mass, linear momentum and angular momentum
//|
//|----------------------------------------------------
void surface_integral::surf_MassPAng(double rex, int lev, cgh *GH, var *chi, var *trK,
                                     var *gxx, var *gxy, var *gxz, var *gyy, var *gyz, var *gzz,
                                     var *Axx, var *Axy, var *Axz, var *Ayy, var *Ayz, var *Azz,
                                     var *Gmx, var *Gmy, var *Gmz,
                                     var *Sfx_rhs, var *Sfy_rhs, var *Sfz_rhs, // temparay memory for mass^i
                                     double *Rout, monitor *Monitor)
{
  if (myrank == 0 && GH->grids[lev] != 1)
    if (Monitor && Monitor->outfile)
      Monitor->outfile << "WARNING: surface integral on multipatches" << endl;
    else
      cout << "WARNING: surface integral on multipatches" << endl;

  double mass, px, py, pz, sx, sy, sz;

  MyList<Patch> *Pp = GH->PatL[lev];
  while (Pp)
  {
    MyList<Block> *BP = Pp->data->blb;
    while (BP)
    {
      Block *cg = BP->data;
      if (myrank == cg->rank)
      {
        f_admmass_bssn(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                       cg->fgfs[chi->sgfn], cg->fgfs[trK->sgfn],
                       cg->fgfs[gxx->sgfn], cg->fgfs[gxy->sgfn], cg->fgfs[gxz->sgfn], cg->fgfs[gyy->sgfn], cg->fgfs[gyz->sgfn], cg->fgfs[gzz->sgfn],
                       cg->fgfs[Axx->sgfn], cg->fgfs[Axy->sgfn], cg->fgfs[Axz->sgfn], cg->fgfs[Ayy->sgfn], cg->fgfs[Ayz->sgfn], cg->fgfs[Azz->sgfn],
                       cg->fgfs[Gmx->sgfn], cg->fgfs[Gmy->sgfn], cg->fgfs[Gmz->sgfn],
                       cg->fgfs[Sfx_rhs->sgfn], cg->fgfs[Sfy_rhs->sgfn], cg->fgfs[Sfz_rhs->sgfn],
                       Symmetry);
      }
      if (BP == Pp->data->ble)
        break;
      BP = BP->next;
    }
    Pp = Pp->next;
  }

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

  int n;
  double *pox[3];
  for (int i = 0; i < 3; i++)
    pox[i] = new double[n_tot];
  for (n = 0; n < n_tot; n++)
  {
    pox[0][n] = rex * nx_g[n];
    pox[1][n] = rex * ny_g[n];
    pox[2][n] = rex * nz_g[n];
    if (isnan(nx_g[n]) || isnan(ny_g[n]) || isnan(nz_g[n]))
    {
      cout << "ERROR: surface integral with NaN coordinates" << endl;
      exit(1);
    }
  }

  double *shellf;
  shellf = new double[n_tot * InList];

  // we have assumed there is only one box on this level,
  // so we do not need loop boxes
  GH->PatL[lev]->data->Interp_Points(DG_List, n_tot, pox, shellf, Symmetry);

  double Mass_out = 0;
  double ang_outx, ang_outy, ang_outz;
  double p_outx, p_outy, p_outz;
  ang_outx = ang_outy = ang_outz = 0.0;
  p_outx = p_outy = p_outz = 0.0;
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

  double Chi, Psi;
  double Gxx, Gxy, Gxz, Gyy, Gyz, Gzz;
  double gupxx, gupxy, gupxz, gupyy, gupyz, gupzz;
  double TRK, axx, axy, axz, ayy, ayz, azz;
  double aupxx, aupxy, aupxz, aupyx, aupyy, aupyz, aupzx, aupzy, aupzz;
  int i;
  for (n = Nmin; n <= Nmax; n++)
  {
    //       need round off always
    i = int(n / N_phi); // int(1.723) = 1, int(-1.732) = -1

    Chi = shellf[InList * n + 3]; // chi in fact
    TRK = shellf[InList * n + 4];
    Gxx = shellf[InList * n + 5] + 1.0;
    Gxy = shellf[InList * n + 6];
    Gxz = shellf[InList * n + 7];
    Gyy = shellf[InList * n + 8] + 1.0;
    Gyz = shellf[InList * n + 9];
    Gzz = shellf[InList * n + 10] + 1.0;
    axx = shellf[InList * n + 11];
    axy = shellf[InList * n + 12];
    axz = shellf[InList * n + 13];
    ayy = shellf[InList * n + 14];
    ayz = shellf[InList * n + 15];
    azz = shellf[InList * n + 16];

    Chi = 1.0 / (1.0 + Chi); // exp(4*phi)
    Psi = Chi * sqrt(Chi);   // Psi^6

// Chi^2 corresponds to metric determinant
// but this factor has been considered in f_admmass_bssn
    // wtcostheta is even function respect costheta
    Mass_out = Mass_out + (shellf[InList * n] * nx_g[n] + shellf[InList * n + 1] * ny_g[n] + shellf[InList * n + 2] * nz_g[n]) * wtcostheta[i];

    gupzz = Gxx * Gyy * Gzz + Gxy * Gyz * Gxz + Gxz * Gxy * Gyz -
            Gxz * Gyy * Gxz - Gxy * Gxy * Gzz - Gxx * Gyz * Gyz;
    gupxx = (Gyy * Gzz - Gyz * Gyz) / gupzz;
    gupxy = -(Gxy * Gzz - Gyz * Gxz) / gupzz;
    gupxz = (Gxy * Gyz - Gyy * Gxz) / gupzz;
    gupyy = (Gxx * Gzz - Gxz * Gxz) / gupzz;
    gupyz = -(Gxx * Gyz - Gxy * Gxz) / gupzz;
    gupzz = (Gxx * Gyy - Gxy * Gxy) / gupzz;

    aupxx = gupxx * axx + gupxy * axy + gupxz * axz;
    aupxy = gupxx * axy + gupxy * ayy + gupxz * ayz;
    aupxz = gupxx * axz + gupxy * ayz + gupxz * azz;
    aupyx = gupxy * axx + gupyy * axy + gupyz * axz;
    aupyy = gupxy * axy + gupyy * ayy + gupyz * ayz;
    aupyz = gupxy * axz + gupyy * ayz + gupyz * azz;
    aupzx = gupxz * axx + gupyz * axy + gupzz * axz;
    aupzy = gupxz * axy + gupyz * ayy + gupzz * ayz;
    aupzz = gupxz * axz + gupyz * ayz + gupzz * azz;
    if (Symmetry == 0)
    {
      // wtcostheta is even function respect costheta
      //  1/8\pi \int \psi^6 (y A^m_z - zA^m_y) dS_m
      ang_outx = ang_outx + f1o8 * Psi * (nx_g[n] * (pox[1][n] * aupxz - pox[2][n] * aupxy) + ny_g[n] * (pox[1][n] * aupyz - pox[2][n] * aupyy) + nz_g[n] * (pox[1][n] * aupzz - pox[2][n] * aupzy)) * wtcostheta[i];
      //  1/8\pi \int \psi^6 (z A^m_x - xA^m_z) dS_m
      ang_outy = ang_outy + f1o8 * Psi * (nx_g[n] * (pox[2][n] * aupxx - pox[0][n] * aupxz) + ny_g[n] * (pox[2][n] * aupyx - pox[0][n] * aupyz) + nz_g[n] * (pox[2][n] * aupzx - pox[0][n] * aupzz)) * wtcostheta[i];
      // 1/8\pi \int \psi^6 (x A^m_y - yA^m_x) dS_m
      ang_outz = ang_outz + f1o8 * Psi * (nx_g[n] * (pox[0][n] * aupxy - pox[1][n] * aupxx) + ny_g[n] * (pox[0][n] * aupyy - pox[1][n] * aupyx) + nz_g[n] * (pox[0][n] * aupzy - pox[1][n] * aupzx)) * wtcostheta[i];
    }
    else if (Symmetry == 1)
    {
      ang_outz = ang_outz + f1o8 * Psi * (nx_g[n] * (pox[0][n] * aupxy - pox[1][n] * aupxx) + ny_g[n] * (pox[0][n] * aupyy - pox[1][n] * aupyx) + nz_g[n] * (pox[0][n] * aupzy - pox[1][n] * aupzx)) * wtcostheta[i];
    }

    axx = Chi * (axx + Gxx * TRK / 3.0);
    axy = Chi * (axy + Gxy * TRK / 3.0);
    axz = Chi * (axz + Gxz * TRK / 3.0);
    ayy = Chi * (ayy + Gyy * TRK / 3.0);
    ayz = Chi * (ayz + Gyz * TRK / 3.0);
    azz = Chi * (azz + Gzz * TRK / 3.0);

    axx = axx - TRK;
    ayy = ayy - TRK;
    azz = azz - TRK;

    // 1/8\pi \int \psi^6 (K_mi - \delta_mi trK) dS^m: lower index linear momentum
    if (Symmetry == 0)
    {
      p_outx = p_outx + f1o8 * Psi * (nx_g[n] * axx + ny_g[n] * axy + nz_g[n] * axz) * wtcostheta[i];
      p_outy = p_outy + f1o8 * Psi * (nx_g[n] * axy + ny_g[n] * ayy + nz_g[n] * ayz) * wtcostheta[i];
      p_outz = p_outz + f1o8 * Psi * (nx_g[n] * axz + ny_g[n] * ayz + nz_g[n] * azz) * wtcostheta[i];
    }
    else if (Symmetry == 1)
    {
      p_outx = p_outx + f1o8 * Psi * (nx_g[n] * axx + ny_g[n] * axy + nz_g[n] * axz) * wtcostheta[i];
      p_outy = p_outy + f1o8 * Psi * (nx_g[n] * axy + ny_g[n] * ayy + nz_g[n] * ayz) * wtcostheta[i];
    }
  }

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

  delete[] pox[0];
  delete[] pox[1];
  delete[] pox[2];
  delete[] shellf;
  DG_List->clearList();
}

void surface_integral::surf_MassPAng(double rex, int lev, cgh *GH, var *chi, var *trK,
                                                                         var *gxx, var *gxy, var *gxz, var *gyy, var *gyz, var *gzz,
                                                                         var *Axx, var *Axy, var *Axz, var *Ayy, var *Ayz, var *Azz,
                                                                         var *Gmx, var *Gmy, var *Gmz,
                                                                         var *Sfx_rhs, var *Sfy_rhs, var *Sfz_rhs, // temparay memory for mass^i
                                                                         double *Rout, monitor *Monitor, MPI_Comm Comm_here)
{
    int lmyrank;
    MPI_Comm_rank(Comm_here, &lmyrank);
    if (lmyrank == 0 && GH->grids[lev] != 1)
        if (Monitor && Monitor->outfile)
            Monitor->outfile << "WARNING: surface integral on multipatches" << endl;
        else
            cout << "WARNING: surface integral on multipatches" << endl;

    double mass, px, py, pz, sx, sy, sz;

    MyList<Patch> *Pp = GH->PatL[lev];
    while (Pp)
    {
        MyList<Block> *BP = Pp->data->blb;
        while (BP)
        {
            Block *cg = BP->data;
            if (myrank == cg->rank)
            {
                f_admmass_bssn(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                             cg->fgfs[chi->sgfn], cg->fgfs[trK->sgfn],
                                             cg->fgfs[gxx->sgfn], cg->fgfs[gxy->sgfn], cg->fgfs[gxz->sgfn], cg->fgfs[gyy->sgfn], cg->fgfs[gyz->sgfn], cg->fgfs[gzz->sgfn],
                                             cg->fgfs[Axx->sgfn], cg->fgfs[Axy->sgfn], cg->fgfs[Axz->sgfn], cg->fgfs[Ayy->sgfn], cg->fgfs[Ayz->sgfn], cg->fgfs[Azz->sgfn],
                                             cg->fgfs[Gmx->sgfn], cg->fgfs[Gmy->sgfn], cg->fgfs[Gmz->sgfn],
                                             cg->fgfs[Sfx_rhs->sgfn], cg->fgfs[Sfy_rhs->sgfn], cg->fgfs[Sfz_rhs->sgfn],
                                             Symmetry);
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }

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

    int n;
    double *pox[3];
    for (int i = 0; i < 3; i++)
        pox[i] = new double[n_tot];
    for (n = 0; n < n_tot; n++)
    {
        pox[0][n] = rex * nx_g[n];
        pox[1][n] = rex * ny_g[n];
        pox[2][n] = rex * nz_g[n];
    }

    double *shellf;
    shellf = new double[n_tot * InList];

    // we have assumed there is only one box on this level,
    // so we do not need loop boxes
    GH->PatL[lev]->data->Interp_Points(DG_List, n_tot, pox, shellf, Symmetry, Comm_here);

    double Mass_out = 0;
    double ang_outx, ang_outy, ang_outz;
    double p_outx, p_outy, p_outz;
    ang_outx = ang_outy = ang_outz = 0.0;
    p_outx = p_outy = p_outz = 0.0;
    const double f1o8 = 0.125;

    int mp, Lp, Nmin, Nmax;

    int cpusize_here;
    MPI_Comm_size(Comm_here, &cpusize_here);

    mp = n_tot / cpusize_here;
    Lp = n_tot - cpusize_here * mp;

    if (Lp > lmyrank)
    {
        Nmin = lmyrank * mp + lmyrank;
        Nmax = Nmin + mp;
    }
    else
    {
        Nmin = lmyrank * mp + Lp;
        Nmax = Nmin + mp - 1;
    }

    double Chi, Psi;
    double Gxx, Gxy, Gxz, Gyy, Gyz, Gzz;
    double gupxx, gupxy, gupxz, gupyy, gupyz, gupzz;
    double TRK, axx, axy, axz, ayy, ayz, azz;
    double aupxx, aupxy, aupxz, aupyx, aupyy, aupyz, aupzx, aupzy, aupzz;
    int i;
    for (n = Nmin; n <= Nmax; n++)
    {
        //       need round off always
        i = int(n / N_phi); // int(1.723) = 1, int(-1.732) = -1

        Chi = shellf[InList * n + 3]; // chi in fact
        TRK = shellf[InList * n + 4];
        Gxx = shellf[InList * n + 5] + 1.0;
        Gxy = shellf[InList * n + 6];
        Gxz = shellf[InList * n + 7];
        Gyy = shellf[InList * n + 8] + 1.0;
        Gyz = shellf[InList * n + 9];
        Gzz = shellf[InList * n + 10] + 1.0;
        axx = shellf[InList * n + 11];
        axy = shellf[InList * n + 12];
        axz = shellf[InList * n + 13];
        ayy = shellf[InList * n + 14];
        ayz = shellf[InList * n + 15];
        azz = shellf[InList * n + 16];

        Chi = 1.0 / (1.0 + Chi); // exp(4*phi)
        Psi = Chi * sqrt(Chi);   // Psi^6

// Chi^2 corresponds to metric determinant
// but this factor has been considered in f_admmass_bssn
        // wtcostheta is even function respect costheta
        Mass_out = Mass_out + (shellf[InList * n] * nx_g[n] + shellf[InList * n + 1] * ny_g[n] + shellf[InList * n + 2] * nz_g[n]) * wtcostheta[i];

        gupzz = Gxx * Gyy * Gzz + Gxy * Gyz * Gxz + Gxz * Gxy * Gyz -
                        Gxz * Gyy * Gxz - Gxy * Gxy * Gzz - Gxx * Gyz * Gyz;
        gupxx = (Gyy * Gzz - Gyz * Gyz) / gupzz;
        gupxy = -(Gxy * Gzz - Gyz * Gxz) / gupzz;
        gupxz = (Gxy * Gyz - Gyy * Gxz) / gupzz;
        gupyy = (Gxx * Gzz - Gxz * Gxz) / gupzz;
        gupyz = -(Gxx * Gyz - Gxy * Gxz) / gupzz;
        gupzz = (Gxx * Gyy - Gxy * Gxy) / gupzz;

        aupxx = gupxx * axx + gupxy * axy + gupxz * axz;
        aupxy = gupxx * axy + gupxy * ayy + gupxz * ayz;
        aupxz = gupxx * axz + gupxy * ayz + gupxz * azz;
        aupyx = gupxy * axx + gupyy * axy + gupyz * axz;
        aupyy = gupxy * axy + gupyy * ayy + gupyz * ayz;
        aupyz = gupxy * axz + gupyy * ayz + gupyz * azz;
        aupzx = gupxz * axx + gupyz * axy + gupzz * axz;
        aupzy = gupxz * axy + gupyz * ayy + gupzz * ayz;
        aupzz = gupxz * axz + gupyz * ayz + gupzz * azz;
        if (Symmetry == 0)
        {
            // wtcostheta is even function respect costheta
            //  1/8\pi \int \psi^6 (y A^m_z - zA^m_y) dS_m
            ang_outx = ang_outx + f1o8 * Psi * (nx_g[n] * (pox[1][n] * aupxz - pox[2][n] * aupxy) + ny_g[n] * (pox[1][n] * aupyz - pox[2][n] * aupyy) + nz_g[n] * (pox[1][n] * aupzz - pox[2][n] * aupzy)) * wtcostheta[i];
            //  1/8\pi \int \psi^6 (z A^m_x - xA^m_z) dS_m
            ang_outy = ang_outy + f1o8 * Psi * (nx_g[n] * (pox[2][n] * aupxx - pox[0][n] * aupxz) + ny_g[n] * (pox[2][n] * aupyx - pox[0][n] * aupyz) + nz_g[n] * (pox[2][n] * aupzx - pox[0][n] * aupzz)) * wtcostheta[i];
            // 1/8\pi \int \psi^6 (x A^m_y - yA^m_x) dS_m
            ang_outz = ang_outz + f1o8 * Psi * (nx_g[n] * (pox[0][n] * aupxy - pox[1][n] * aupxx) + ny_g[n] * (pox[0][n] * aupyy - pox[1][n] * aupyx) + nz_g[n] * (pox[0][n] * aupzy - pox[1][n] * aupzx)) * wtcostheta[i];
        }
        else if (Symmetry == 1)
        {
            ang_outz = ang_outz + f1o8 * Psi * (nx_g[n] * (pox[0][n] * aupxy - pox[1][n] * aupxx) + ny_g[n] * (pox[0][n] * aupyy - pox[1][n] * aupyx) + nz_g[n] * (pox[0][n] * aupzy - pox[1][n] * aupzx)) * wtcostheta[i];
        }

        axx = Chi * (axx + Gxx * TRK / 3.0);
        axy = Chi * (axy + Gxy * TRK / 3.0);
        axz = Chi * (axz + Gxz * TRK / 3.0);
        ayy = Chi * (ayy + Gyy * TRK / 3.0);
        ayz = Chi * (ayz + Gyz * TRK / 3.0);
        azz = Chi * (azz + Gzz * TRK / 3.0);

        axx = axx - TRK;
        ayy = ayy - TRK;
        azz = azz - TRK;

        // 1/8\pi \int \psi^6 (K_mi - \delta_mi trK) dS^m: lower index linear momentum
        if (Symmetry == 0)
        {
            p_outx = p_outx + f1o8 * Psi * (nx_g[n] * axx + ny_g[n] * axy + nz_g[n] * axz) * wtcostheta[i];
            p_outy = p_outy + f1o8 * Psi * (nx_g[n] * axy + ny_g[n] * ayy + nz_g[n] * ayz) * wtcostheta[i];
            p_outz = p_outz + f1o8 * Psi * (nx_g[n] * axz + ny_g[n] * ayz + nz_g[n] * azz) * wtcostheta[i];
        }
        else if (Symmetry == 1)
        {
            p_outx = p_outx + f1o8 * Psi * (nx_g[n] * axx + ny_g[n] * axy + nz_g[n] * axz) * wtcostheta[i];
            p_outy = p_outy + f1o8 * Psi * (nx_g[n] * axy + ny_g[n] * ayy + nz_g[n] * ayz) * wtcostheta[i];
            p_outz = p_outz + f1o8 * Psi * (nx_g[n] * axz + ny_g[n] * ayz + nz_g[n] * azz) * wtcostheta[i];
        }
    }

    MPI_Allreduce(&Mass_out, &mass, 1, MPI_DOUBLE, MPI_SUM, Comm_here);

    MPI_Allreduce(&ang_outx, &sx, 1, MPI_DOUBLE, MPI_SUM, Comm_here);
    MPI_Allreduce(&ang_outy, &sy, 1, MPI_DOUBLE, MPI_SUM, Comm_here);
    MPI_Allreduce(&ang_outz, &sz, 1, MPI_DOUBLE, MPI_SUM, Comm_here);

    MPI_Allreduce(&p_outx, &px, 1, MPI_DOUBLE, MPI_SUM, Comm_here);
    MPI_Allreduce(&p_outy, &py, 1, MPI_DOUBLE, MPI_SUM, Comm_here);
    MPI_Allreduce(&p_outz, &pz, 1, MPI_DOUBLE, MPI_SUM, Comm_here);

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

    delete[] pox[0];
    delete[] pox[1];
    delete[] pox[2];
    delete[] shellf;
    DG_List->clearList();
}
