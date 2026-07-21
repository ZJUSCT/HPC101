
#include <sstream>
#include <cstdio>
using namespace std;

#include "macrodef.h"
#include "misc.h"
#include <time.h>
#include "Ansorg.h"
#include "fmisc.h"
#include "Parallel.h"
#include "bssn_gpu_class.h"
#include "bssn_rhs.h"
#include "initial_puncture.h"
#include "enforce_algebra.h"
#include "rungekutta4_rout.h"
#include "sommerfeld_rout.h"
#include "getnp4.h"

#include "perf.h"
#include "derivatives.h"

// include GPU files
#include "gpu_manager.h"
#include "helper.h"
#include "macrodef.h"

//================================================================================================

// Define bssn_gpu_class

//================================================================================================

bssn_class::bssn_class(double Couranti, double StartTimei, double TotalTimei, 
                                             double DumpTimei, double d2DumpTimei, double CheckTimei, double AnasTimei,
                                             int Symmetryi, int checkruni, char *checkfilenamei, 
                                             double numepssi, double numepsbi, double numepshi,
                                             int a_levi, int maxli, int decni, double maxrexi, double drexi) 
                                             : Courant(Couranti), StartTime(StartTimei), TotalTime(TotalTimei), 
                                                 DumpTime(DumpTimei), d2DumpTime(d2DumpTimei), CheckTime(CheckTimei), AnasTime(AnasTimei),
                                                 Symmetry(Symmetryi), checkrun(checkruni), numepss(numepssi), numepsb(numepsbi), numepsh(numepshi),
                                                 a_lev(a_levi), maxl(maxli), decn(decni), maxrex(maxrexi), drex(drexi),
                                                 CheckPoint(0)
{
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // setup Monitors
    {
        stringstream a_stream;
        a_stream.setf(ios::left);
        a_stream << "# Error log information";
        ErrorMonitor = new monitor("Error.log", myrank, a_stream.str());
        ErrorMonitor->print_message("Warning: we always assume intput parameter in cell center style.");

        a_stream.clear();
        a_stream.str("");
        a_stream << setw(15) << "# time";
        char str[50];
        for (int pl = 2; pl < maxl + 1; pl++)
            for (int pm = -pl; pm < pl + 1; pm++)
            {
                sprintf(str, "R%02dm%03d", pl, pm);
                a_stream << setw(16) << str;
                sprintf(str, "I%02dm%03d", pl, pm);
                a_stream << setw(16) << str;
            }
        Psi4Monitor = new monitor("bssn_psi4.dat", myrank, a_stream.str());

        a_stream.clear();
        a_stream.str("");
        a_stream << setw(15) << "# time";
        BHMonitor = new monitor("bssn_BH.dat", myrank, a_stream.str());

        a_stream.clear();
        a_stream.str("");
        a_stream << setw(15) << "# time ADMmass ADMPx ADMPy ADMPz ADMSx ADMSy ADMSz";
        MAPMonitor = new monitor("bssn_ADMQs.dat", myrank, a_stream.str());

        a_stream.clear();
        a_stream.str("");
        a_stream << setw(15) << "# time Ham Px Py Pz Gx Gy Gz";
        ConVMonitor = new monitor("bssn_constraint.dat", myrank, a_stream.str());
    }
    // setup sphere integration engine
    Waveshell = new surface_integral(Symmetry);

    trfls = 0;
    chitiny = 0;
    // read parameter from file
    {
        char filename[50];
        strcpy(filename, "input.par");
        const int LEN = 256;
        char pline[LEN];
        string str, sgrp, skey, sval;
        int sind;
        ifstream inf(filename, ifstream::in);
        if (!inf.good() && myrank == 0)
        {
            if (ErrorMonitor->outfile)
                ErrorMonitor->outfile << "Can not open parameter file " << filename 
                                                            << " for inputing information of black holes" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 1; inf.good(); i++)
        {
            inf.getline(pline, LEN);
            str = pline;

            int status = misc::parse_parts(str, sgrp, skey, sval, sind);
            if (status == -1)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "error reading parameter file " << filename 
                                                                << " in line " << i << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            else if (status == 0)
                continue;

            if (sgrp == "BSSN" && skey == "chitiny")
                chitiny = atof(sval.c_str());
            else if (sgrp == "BSSN" && skey == "time refinement start from level")
                trfls = atoi(sval.c_str());
        }
        inf.close();
    }
    if (myrank == 0)
    {
        // echo information of lower bound of chi
        cout << "chitiny = " << chitiny << endl;
        cout << "time refinement start from level #" << trfls << endl;
    }

    chitiny = chitiny - 1; // because we have subtracted one from chi

    strcpy(checkfilename, checkfilenamei);

    ngfs = 0;
    phio = new var("phio", ngfs++, 1, 1, 1);
    trKo = new var("trKo", ngfs++, 1, 1, 1);
    gxxo = new var("gxxo", ngfs++, 1, 1, 1);
    gxyo = new var("gxyo", ngfs++, -1, -1, 1);
    gxzo = new var("gxzo", ngfs++, -1, 1, -1);
    gyyo = new var("gyyo", ngfs++, 1, 1, 1);
    gyzo = new var("gyzo", ngfs++, 1, -1, -1);
    gzzo = new var("gzzo", ngfs++, 1, 1, 1);
    Axxo = new var("Axxo", ngfs++, 1, 1, 1);
    Axyo = new var("Axyo", ngfs++, -1, -1, 1);
    Axzo = new var("Axzo", ngfs++, -1, 1, -1);
    Ayyo = new var("Ayyo", ngfs++, 1, 1, 1);
    Ayzo = new var("Ayzo", ngfs++, 1, -1, -1);
    Azzo = new var("Azzo", ngfs++, 1, 1, 1);
    Gmxo = new var("Gmxo", ngfs++, -1, 1, 1);
    Gmyo = new var("Gmyo", ngfs++, 1, -1, 1);
    Gmzo = new var("Gmzo", ngfs++, 1, 1, -1);
    Lapo = new var("Lapo", ngfs++, 1, 1, 1);
    Sfxo = new var("Sfxo", ngfs++, -1, 1, 1);
    Sfyo = new var("Sfyo", ngfs++, 1, -1, 1);
    Sfzo = new var("Sfzo", ngfs++, 1, 1, -1);
    dtSfxo = new var("dtSfxo", ngfs++, -1, 1, 1);
    dtSfyo = new var("dtSfyo", ngfs++, 1, -1, 1);
    dtSfzo = new var("dtSfzo", ngfs++, 1, 1, -1);

    phi0 = new var("phi0", ngfs++, 1, 1, 1);
    trK0 = new var("trK0", ngfs++, 1, 1, 1);
    gxx0 = new var("gxx0", ngfs++, 1, 1, 1);
    gxy0 = new var("gxy0", ngfs++, -1, -1, 1);
    gxz0 = new var("gxz0", ngfs++, -1, 1, -1);
    gyy0 = new var("gyy0", ngfs++, 1, 1, 1);
    gyz0 = new var("gyz0", ngfs++, 1, -1, -1);
    gzz0 = new var("gzz0", ngfs++, 1, 1, 1);
    Axx0 = new var("Axx0", ngfs++, 1, 1, 1);
    Axy0 = new var("Axy0", ngfs++, -1, -1, 1);
    Axz0 = new var("Axz0", ngfs++, -1, 1, -1);
    Ayy0 = new var("Ayy0", ngfs++, 1, 1, 1);
    Ayz0 = new var("Ayz0", ngfs++, 1, -1, -1);
    Azz0 = new var("Azz0", ngfs++, 1, 1, 1);
    Gmx0 = new var("Gmx0", ngfs++, -1, 1, 1);
    Gmy0 = new var("Gmy0", ngfs++, 1, -1, 1);
    Gmz0 = new var("Gmz0", ngfs++, 1, 1, -1);
    Lap0 = new var("Lap0", ngfs++, 1, 1, 1);
    Sfx0 = new var("Sfx0", ngfs++, -1, 1, 1);
    Sfy0 = new var("Sfy0", ngfs++, 1, -1, 1);
    Sfz0 = new var("Sfz0", ngfs++, 1, 1, -1);
    dtSfx0 = new var("dtSfx0", ngfs++, -1, 1, 1);
    dtSfy0 = new var("dtSfy0", ngfs++, 1, -1, 1);
    dtSfz0 = new var("dtSfz0", ngfs++, 1, 1, -1);

    phi = new var("phi", ngfs++, 1, 1, 1);
    trK = new var("trK", ngfs++, 1, 1, 1);
    gxx = new var("gxx", ngfs++, 1, 1, 1);
    gxy = new var("gxy", ngfs++, -1, -1, 1);
    gxz = new var("gxz", ngfs++, -1, 1, -1);
    gyy = new var("gyy", ngfs++, 1, 1, 1);
    gyz = new var("gyz", ngfs++, 1, -1, -1);
    gzz = new var("gzz", ngfs++, 1, 1, 1);
    Axx = new var("Axx", ngfs++, 1, 1, 1);
    Axy = new var("Axy", ngfs++, -1, -1, 1);
    Axz = new var("Axz", ngfs++, -1, 1, -1);
    Ayy = new var("Ayy", ngfs++, 1, 1, 1);
    Ayz = new var("Ayz", ngfs++, 1, -1, -1);
    Azz = new var("Azz", ngfs++, 1, 1, 1);
    Gmx = new var("Gmx", ngfs++, -1, 1, 1);
    Gmy = new var("Gmy", ngfs++, 1, -1, 1);
    Gmz = new var("Gmz", ngfs++, 1, 1, -1);
    Lap = new var("Lap", ngfs++, 1, 1, 1);
    Sfx = new var("Sfx", ngfs++, -1, 1, 1);
    Sfy = new var("Sfy", ngfs++, 1, -1, 1);
    Sfz = new var("Sfz", ngfs++, 1, 1, -1);
    dtSfx = new var("dtSfx", ngfs++, -1, 1, 1);
    dtSfy = new var("dtSfy", ngfs++, 1, -1, 1);
    dtSfz = new var("dtSfz", ngfs++, 1, 1, -1);

    phi1 = new var("phi1", ngfs++, 1, 1, 1);
    trK1 = new var("trK1", ngfs++, 1, 1, 1);
    gxx1 = new var("gxx1", ngfs++, 1, 1, 1);
    gxy1 = new var("gxy1", ngfs++, -1, -1, 1);
    gxz1 = new var("gxz1", ngfs++, -1, 1, -1);
    gyy1 = new var("gyy1", ngfs++, 1, 1, 1);
    gyz1 = new var("gyz1", ngfs++, 1, -1, -1);
    gzz1 = new var("gzz1", ngfs++, 1, 1, 1);
    Axx1 = new var("Axx1", ngfs++, 1, 1, 1);
    Axy1 = new var("Axy1", ngfs++, -1, -1, 1);
    Axz1 = new var("Axz1", ngfs++, -1, 1, -1);
    Ayy1 = new var("Ayy1", ngfs++, 1, 1, 1);
    Ayz1 = new var("Ayz1", ngfs++, 1, -1, -1);
    Azz1 = new var("Azz1", ngfs++, 1, 1, 1);
    Gmx1 = new var("Gmx1", ngfs++, -1, 1, 1);
    Gmy1 = new var("Gmy1", ngfs++, 1, -1, 1);
    Gmz1 = new var("Gmz1", ngfs++, 1, 1, -1);
    Lap1 = new var("Lap1", ngfs++, 1, 1, 1);
    Sfx1 = new var("Sfx1", ngfs++, -1, 1, 1);
    Sfy1 = new var("Sfy1", ngfs++, 1, -1, 1);
    Sfz1 = new var("Sfz1", ngfs++, 1, 1, -1);
    dtSfx1 = new var("dtSfx1", ngfs++, -1, 1, 1);
    dtSfy1 = new var("dtSfy1", ngfs++, 1, -1, 1);
    dtSfz1 = new var("dtSfz1", ngfs++, 1, 1, -1);

    phi_rhs = new var("phi_rhs", ngfs++, 1, 1, 1);
    trK_rhs = new var("trK_rhs", ngfs++, 1, 1, 1);
    gxx_rhs = new var("gxx_rhs", ngfs++, 1, 1, 1);
    gxy_rhs = new var("gxy_rhs", ngfs++, -1, -1, 1);
    gxz_rhs = new var("gxz_rhs", ngfs++, -1, 1, -1);
    gyy_rhs = new var("gyy_rhs", ngfs++, 1, 1, 1);
    gyz_rhs = new var("gyz_rhs", ngfs++, 1, -1, -1);
    gzz_rhs = new var("gzz_rhs", ngfs++, 1, 1, 1);
    Axx_rhs = new var("Axx_rhs", ngfs++, 1, 1, 1);
    Axy_rhs = new var("Axy_rhs", ngfs++, -1, -1, 1);
    Axz_rhs = new var("Axz_rhs", ngfs++, -1, 1, -1);
    Ayy_rhs = new var("Ayy_rhs", ngfs++, 1, 1, 1);
    Ayz_rhs = new var("Ayz_rhs", ngfs++, 1, -1, -1);
    Azz_rhs = new var("Azz_rhs", ngfs++, 1, 1, 1);
    Gmx_rhs = new var("Gmx_rhs", ngfs++, -1, 1, 1);
    Gmy_rhs = new var("Gmy_rhs", ngfs++, 1, -1, 1);
    Gmz_rhs = new var("Gmz_rhs", ngfs++, 1, 1, -1);
    Lap_rhs = new var("Lap_rhs", ngfs++, 1, 1, 1);
    Sfx_rhs = new var("Sfx_rhs", ngfs++, -1, 1, 1);
    Sfy_rhs = new var("Sfy_rhs", ngfs++, 1, -1, 1);
    Sfz_rhs = new var("Sfz_rhs", ngfs++, 1, 1, -1);
    dtSfx_rhs = new var("dtSfx_rhs", ngfs++, -1, 1, 1);
    dtSfy_rhs = new var("dtSfy_rhs", ngfs++, 1, -1, 1);
    dtSfz_rhs = new var("dtSfz_rhs", ngfs++, 1, 1, -1);

    rho = new var("rho", ngfs++, 1, 1, 1);
    Sx = new var("Sx", ngfs++, -1, 1, 1);
    Sy = new var("Sy", ngfs++, 1, -1, 1);
    Sz = new var("Sz", ngfs++, 1, 1, -1);
    Sxx = new var("Sxx", ngfs++, 1, 1, 1);
    Sxy = new var("Sxy", ngfs++, -1, -1, 1);
    Sxz = new var("Sxz", ngfs++, -1, 1, -1);
    Syy = new var("Syy", ngfs++, 1, 1, 1);
    Syz = new var("Syz", ngfs++, 1, -1, -1);
    Szz = new var("Szz", ngfs++, 1, 1, 1);

    Gamxxx = new var("Gamxxx", ngfs++, -1, 1, 1);
    Gamxxy = new var("Gamxxy", ngfs++, 1, -1, 1);
    Gamxxz = new var("Gamxxz", ngfs++, 1, 1, -1);
    Gamxyy = new var("Gamxyy", ngfs++, -1, 1, 1);
    Gamxyz = new var("Gamxyz", ngfs++, -1, -1, -1);
    Gamxzz = new var("Gamxzz", ngfs++, -1, 1, 1);
    Gamyxx = new var("Gamyxx", ngfs++, 1, -1, 1);
    Gamyxy = new var("Gamyxy", ngfs++, -1, 1, 1);
    Gamyxz = new var("Gamyxz", ngfs++, -1, -1, -1);
    Gamyyy = new var("Gamyyy", ngfs++, 1, -1, 1);
    Gamyyz = new var("Gamyyz", ngfs++, 1, 1, -1);
    Gamyzz = new var("Gamyzz", ngfs++, 1, -1, 1);
    Gamzxx = new var("Gamzxx", ngfs++, 1, 1, -1);
    Gamzxy = new var("Gamzxy", ngfs++, -1, -1, -1);
    Gamzxz = new var("Gamzxz", ngfs++, -1, 1, 1);
    Gamzyy = new var("Gamzyy", ngfs++, 1, 1, -1);
    Gamzyz = new var("Gamzyz", ngfs++, 1, -1, 1);
    Gamzzz = new var("Gamzzz", ngfs++, 1, 1, -1);

    Rxx = new var("Rxx", ngfs++, 1, 1, 1);
    Rxy = new var("Rxy", ngfs++, -1, -1, 1);
    Rxz = new var("Rxz", ngfs++, -1, 1, -1);
    Ryy = new var("Ryy", ngfs++, 1, 1, 1);
    Ryz = new var("Ryz", ngfs++, 1, -1, -1);
    Rzz = new var("Rzz", ngfs++, 1, 1, 1);

    // refer to PRD, 77, 024027 (2008)
    Rpsi4 = new var("Rpsi4", ngfs++, 1, 1, 1);
    Ipsi4 = new var("Ipsi4", ngfs++, -1, -1, -1);
    t1Rpsi4 = new var("t1Rpsi4", ngfs++, 1, 1, 1);
    t1Ipsi4 = new var("t1Ipsi4", ngfs++, -1, -1, -1);
    t2Rpsi4 = new var("t2Rpsi4", ngfs++, 1, 1, 1);
    t2Ipsi4 = new var("t2Ipsi4", ngfs++, -1, -1, -1);

    // constraint violation monitor variables
    Cons_Ham = new var("Cons_Ham", ngfs++, 1, 1, 1);
    Cons_Px = new var("Cons_Px", ngfs++, -1, 1, 1);
    Cons_Py = new var("Cons_Py", ngfs++, 1, -1, 1);
    Cons_Pz = new var("Cons_Pz", ngfs++, 1, 1, -1);
    Cons_Gx = new var("Cons_Gx", ngfs++, -1, 1, 1);
    Cons_Gy = new var("Cons_Gy", ngfs++, 1, -1, 1);
    Cons_Gz = new var("Cons_Gz", ngfs++, 1, 1, -1);

    // specific properspeed for 1+log slice
    {
        const double vl = sqrt(2);
        trKo->setpropspeed(vl);
        trK0->setpropspeed(vl);
        trK->setpropspeed(vl);
        trK1->setpropspeed(vl);
        trK_rhs->setpropspeed(vl);

        phio->setpropspeed(vl);
        phi0->setpropspeed(vl);
        phi->setpropspeed(vl);
        phi1->setpropspeed(vl);
        phi_rhs->setpropspeed(vl);

        Lapo->setpropspeed(vl);
        Lap0->setpropspeed(vl);
        Lap->setpropspeed(vl);
        Lap1->setpropspeed(vl);
        Lap_rhs->setpropspeed(vl);
    }

    OldStateList = new MyList<var>(phio);
    OldStateList->insert(trKo);
    OldStateList->insert(gxxo);
    OldStateList->insert(gxyo);
    OldStateList->insert(gxzo);
    OldStateList->insert(gyyo);
    OldStateList->insert(gyzo);
    OldStateList->insert(gzzo);
    OldStateList->insert(Axxo);
    OldStateList->insert(Axyo);
    OldStateList->insert(Axzo);
    OldStateList->insert(Ayyo);
    OldStateList->insert(Ayzo);
    OldStateList->insert(Azzo);
    OldStateList->insert(Gmxo);
    OldStateList->insert(Gmyo);
    OldStateList->insert(Gmzo);
    OldStateList->insert(Lapo);
    OldStateList->insert(Sfxo);
    OldStateList->insert(Sfyo);
    OldStateList->insert(Sfzo);
    OldStateList->insert(dtSfxo);
    OldStateList->insert(dtSfyo);
    OldStateList->insert(dtSfzo);

    StateList = new MyList<var>(phi0);
    StateList->insert(trK0);
    StateList->insert(gxx0);
    StateList->insert(gxy0);
    StateList->insert(gxz0);
    StateList->insert(gyy0);
    StateList->insert(gyz0);
    StateList->insert(gzz0);
    StateList->insert(Axx0);
    StateList->insert(Axy0);
    StateList->insert(Axz0);
    StateList->insert(Ayy0);
    StateList->insert(Ayz0);
    StateList->insert(Azz0);
    StateList->insert(Gmx0);
    StateList->insert(Gmy0);
    StateList->insert(Gmz0);
    StateList->insert(Lap0);
    StateList->insert(Sfx0);
    StateList->insert(Sfy0);
    StateList->insert(Sfz0);
    StateList->insert(dtSfx0);
    StateList->insert(dtSfy0);
    StateList->insert(dtSfz0);

    RHSList = new MyList<var>(phi_rhs);
    RHSList->insert(trK_rhs);
    RHSList->insert(gxx_rhs);
    RHSList->insert(gxy_rhs);
    RHSList->insert(gxz_rhs);
    RHSList->insert(gyy_rhs);
    RHSList->insert(gyz_rhs);
    RHSList->insert(gzz_rhs);
    RHSList->insert(Axx_rhs);
    RHSList->insert(Axy_rhs);
    RHSList->insert(Axz_rhs);
    RHSList->insert(Ayy_rhs);
    RHSList->insert(Ayz_rhs);
    RHSList->insert(Azz_rhs);
    RHSList->insert(Gmx_rhs);
    RHSList->insert(Gmy_rhs);
    RHSList->insert(Gmz_rhs);
    RHSList->insert(Lap_rhs);
    RHSList->insert(Sfx_rhs);
    RHSList->insert(Sfy_rhs);
    RHSList->insert(Sfz_rhs);
    RHSList->insert(dtSfx_rhs);
    RHSList->insert(dtSfy_rhs);
    RHSList->insert(dtSfz_rhs);

    SynchList_pre = new MyList<var>(phi);
    SynchList_pre->insert(trK);
    SynchList_pre->insert(gxx);
    SynchList_pre->insert(gxy);
    SynchList_pre->insert(gxz);
    SynchList_pre->insert(gyy);
    SynchList_pre->insert(gyz);
    SynchList_pre->insert(gzz);
    SynchList_pre->insert(Axx);
    SynchList_pre->insert(Axy);
    SynchList_pre->insert(Axz);
    SynchList_pre->insert(Ayy);
    SynchList_pre->insert(Ayz);
    SynchList_pre->insert(Azz);
    SynchList_pre->insert(Gmx);
    SynchList_pre->insert(Gmy);
    SynchList_pre->insert(Gmz);
    SynchList_pre->insert(Lap);
    SynchList_pre->insert(Sfx);
    SynchList_pre->insert(Sfy);
    SynchList_pre->insert(Sfz);
    SynchList_pre->insert(dtSfx);
    SynchList_pre->insert(dtSfy);
    SynchList_pre->insert(dtSfz);

    SynchList_cor = new MyList<var>(phi1);
    SynchList_cor->insert(trK1);
    SynchList_cor->insert(gxx1);
    SynchList_cor->insert(gxy1);
    SynchList_cor->insert(gxz1);
    SynchList_cor->insert(gyy1);
    SynchList_cor->insert(gyz1);
    SynchList_cor->insert(gzz1);
    SynchList_cor->insert(Axx1);
    SynchList_cor->insert(Axy1);
    SynchList_cor->insert(Axz1);
    SynchList_cor->insert(Ayy1);
    SynchList_cor->insert(Ayz1);
    SynchList_cor->insert(Azz1);
    SynchList_cor->insert(Gmx1);
    SynchList_cor->insert(Gmy1);
    SynchList_cor->insert(Gmz1);
    SynchList_cor->insert(Lap1);
    SynchList_cor->insert(Sfx1);
    SynchList_cor->insert(Sfy1);
    SynchList_cor->insert(Sfz1);
    SynchList_cor->insert(dtSfx1);
    SynchList_cor->insert(dtSfy1);
    SynchList_cor->insert(dtSfz1);

    DumpList = new MyList<var>(phi0);
    DumpList->insert(trK0);
    DumpList->insert(gxx0);
    DumpList->insert(gxy0);
    DumpList->insert(gxz0);
    DumpList->insert(gyy0);
    DumpList->insert(gyz0);
    DumpList->insert(gzz0);
    DumpList->insert(Lap0);
    DumpList->insert(Rpsi4);
    DumpList->insert(Ipsi4);
    DumpList->insert(Cons_Ham);
    DumpList->insert(Cons_Px);
    DumpList->insert(Cons_Py);
    DumpList->insert(Cons_Pz);

    ConstraintList = new MyList<var>(Cons_Ham);
    ConstraintList->insert(Cons_Px);
    ConstraintList->insert(Cons_Py);
    ConstraintList->insert(Cons_Pz);
    ConstraintList->insert(Cons_Gx);
    ConstraintList->insert(Cons_Gy);
    ConstraintList->insert(Cons_Gz);

    MiscList = new MyList<var>(rho);
    MiscList->insert(Sx);
    MiscList->insert(Sy);
    MiscList->insert(Sz);
    MiscList->insert(Sxx);
    MiscList->insert(Sxy);
    MiscList->insert(Sxz);
    MiscList->insert(Syy);
    MiscList->insert(Syz);
    MiscList->insert(Szz);

    DGList = new MyList<var>(Rpsi4);
    DGList->insert(Ipsi4);
    DGList->insert(t1Rpsi4);
    DGList->insert(t1Ipsi4);
    DGList->insert(t2Rpsi4);
    DGList->insert(t2Ipsi4);

    CheckPoint = new checkpoint(checkrun, checkfilename, myrank);
}

//================================================================================================



//================================================================================================

// This member function initializes the class

//================================================================================================

void bssn_class::Initialize()
{
    if (myrank == 0)
        cout << "you have setted " << ngfs << " grid functions." << endl;

    CheckPoint->addvariablelist(StateList);
    CheckPoint->addvariablelist(OldStateList);

    GH = new cgh(0, ngfs, Symmetry, "input.par", checkrun, ErrorMonitor);
    if (checkrun)
        CheckPoint->readcheck_cgh(PhysTime, GH, myrank, nprocs, Symmetry);
    else
        GH->compose_cgh(nprocs);

    double h = GH->PatL[0]->data->blb->data->getdX(0);
    for (int i = 1; i < dim; i++)
        h = Mymin(h, GH->PatL[0]->data->blb->data->getdX(i));
    dT = Courant * h;

    if (checkrun)
    {
        CheckPoint->read_Black_Hole_position(BH_num_input, BH_num, Porg0, Pmom, Spin, Mass, Porgbr, Porg, Porg1, Porg_rhs);
    }
    else
    {
        PhysTime = StartTime;
        Setup_Black_Hole_position();
    }
}

//================================================================================================



//================================================================================================

// Destructor: free allocated variables

//================================================================================================

bssn_class::~bssn_class()
{
    StateList->clearList();
    RHSList->clearList();
    OldStateList->clearList();
    SynchList_pre->clearList();
    SynchList_cor->clearList();
    DumpList->clearList();
    ConstraintList->clearList();
    MiscList->clearList();
    DGList->clearList();

    delete phio;
    delete trKo;
    delete gxxo;
    delete gxyo;
    delete gxzo;
    delete gyyo;
    delete gyzo;
    delete gzzo;
    delete Axxo;
    delete Axyo;
    delete Axzo;
    delete Ayyo;
    delete Ayzo;
    delete Azzo;
    delete Gmxo;
    delete Gmyo;
    delete Gmzo;
    delete Lapo;
    delete Sfxo;
    delete Sfyo;
    delete Sfzo;
    delete dtSfxo;
    delete dtSfyo;
    delete dtSfzo;

    delete phi0;
    delete trK0;
    delete gxx0;
    delete gxy0;
    delete gxz0;
    delete gyy0;
    delete gyz0;
    delete gzz0;
    delete Axx0;
    delete Axy0;
    delete Axz0;
    delete Ayy0;
    delete Ayz0;
    delete Azz0;
    delete Gmx0;
    delete Gmy0;
    delete Gmz0;
    delete Lap0;
    delete Sfx0;
    delete Sfy0;
    delete Sfz0;
    delete dtSfx0;
    delete dtSfy0;
    delete dtSfz0;

    delete phi;
    delete trK;
    delete gxx;
    delete gxy;
    delete gxz;
    delete gyy;
    delete gyz;
    delete gzz;
    delete Axx;
    delete Axy;
    delete Axz;
    delete Ayy;
    delete Ayz;
    delete Azz;
    delete Gmx;
    delete Gmy;
    delete Gmz;
    delete Lap;
    delete Sfx;
    delete Sfy;
    delete Sfz;
    delete dtSfx;
    delete dtSfy;
    delete dtSfz;

    delete phi1;
    delete trK1;
    delete gxx1;
    delete gxy1;
    delete gxz1;
    delete gyy1;
    delete gyz1;
    delete gzz1;
    delete Axx1;
    delete Axy1;
    delete Axz1;
    delete Ayy1;
    delete Ayz1;
    delete Azz1;
    delete Gmx1;
    delete Gmy1;
    delete Gmz1;
    delete Lap1;
    delete Sfx1;
    delete Sfy1;
    delete Sfz1;
    delete dtSfx1;
    delete dtSfy1;
    delete dtSfz1;

    delete phi_rhs;
    delete trK_rhs;
    delete gxx_rhs;
    delete gxy_rhs;
    delete gxz_rhs;
    delete gyy_rhs;
    delete gyz_rhs;
    delete gzz_rhs;
    delete Axx_rhs;
    delete Axy_rhs;
    delete Axz_rhs;
    delete Ayy_rhs;
    delete Ayz_rhs;
    delete Azz_rhs;
    delete Gmx_rhs;
    delete Gmy_rhs;
    delete Gmz_rhs;
    delete Lap_rhs;
    delete Sfx_rhs;
    delete Sfy_rhs;
    delete Sfz_rhs;
    delete dtSfx_rhs;
    delete dtSfy_rhs;
    delete dtSfz_rhs;

    delete rho;
    delete Sx;
    delete Sy;
    delete Sz;
    delete Sxx;
    delete Sxy;
    delete Sxz;
    delete Syy;
    delete Syz;
    delete Szz;

    delete Gamxxx;
    delete Gamxxy;
    delete Gamxxz;
    delete Gamxyy;
    delete Gamxyz;
    delete Gamxzz;
    delete Gamyxx;
    delete Gamyxy;
    delete Gamyxz;
    delete Gamyyy;
    delete Gamyyz;
    delete Gamyzz;
    delete Gamzxx;
    delete Gamzxy;
    delete Gamzxz;
    delete Gamzyy;
    delete Gamzyz;
    delete Gamzzz;

    delete Rxx;
    delete Rxy;
    delete Rxz;
    delete Ryy;
    delete Ryz;
    delete Rzz;

    delete Rpsi4;
    delete Ipsi4;
    delete t1Rpsi4;
    delete t1Ipsi4;
    delete t2Rpsi4;
    delete t2Ipsi4;

    delete Cons_Ham;
    delete Cons_Px;
    delete Cons_Py;
    delete Cons_Pz;
    delete Cons_Gx;
    delete Cons_Gy;
    delete Cons_Gz;

    delete GH;

    for (int i = 0; i < BH_num; i++)
    {
        delete[] Porg0[i];
        delete[] Porgbr[i];
        delete[] Porg[i];
        delete[] Porg1[i];
        delete[] Porg_rhs[i];
    }

    delete[] Porg0;
    delete[] Porgbr;
    delete[] Porg;
    delete[] Porg1;
    delete[] Porg_rhs;

    delete[] Mass;
    delete[] Spin;
    delete[] Pmom;

    delete ErrorMonitor;
    delete Psi4Monitor;
    delete BHMonitor;
    delete MAPMonitor;
    delete ConVMonitor;
    delete Waveshell;

    delete CheckPoint;
}

//================================================================================================



//================================================================================================

// This member function computes initial data using Lousto's analytic formulas

//================================================================================================

void bssn_class::Setup_Initial_Data_Lousto()
{
    if (!checkrun)
    {
        if (myrank == 0)
            cout << "Setup initial data with Lousto's analytical formula." << endl;
        char filename[50];
        strcpy(filename, "input.par");
        int BH_NM;
        double *Porg_here, *Pmom_here, *Spin_here, *Mass_here;
        // read parameter from file
        {
            const int LEN = 256;
            char pline[LEN];
            string str, sgrp, skey, sval;
            int sind;
            ifstream inf(filename, ifstream::in);
            if (!inf.good() && myrank == 0)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "Can not open parameter file " << filename 
                                                                << " for inputing information of black holes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i = 1; inf.good(); i++)
            {
                inf.getline(pline, LEN);
                str = pline;

                int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                if (status == -1)
                {
                    if (ErrorMonitor->outfile)
                        ErrorMonitor->outfile << "error reading parameter file " << filename << " in line " << i << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                else if (status == 0)
                    continue;

                if (sgrp == "BSSN" && skey == "BH_num")
                {
                    BH_NM = atoi(sval.c_str());
                    break;
                }
            }
            inf.close();
        }

        Porg_here = new double[3 * BH_NM];
        Pmom_here = new double[3 * BH_NM];
        Spin_here = new double[3 * BH_NM];
        Mass_here = new double[BH_NM];
        // read parameter from file
        {
            const int LEN = 256;
            char pline[LEN];
            string str, sgrp, skey, sval;
            int sind;
            ifstream inf(filename, ifstream::in);
            if (!inf.good() && myrank == 0)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "Can not open parameter file " << filename
                                                                << " for inputing information of black holes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i = 1; inf.good(); i++)
            {
                inf.getline(pline, LEN);
                str = pline;

                int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                if (status == -1)
                {
                    if (ErrorMonitor->outfile)
                        ErrorMonitor->outfile << "error reading parameter file " << filename << " in line " << i << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                else if (status == 0)
                    continue;

                if (sgrp == "BSSN" && sind < BH_NM)
                {
                    if (skey == "Mass")
                        Mass_here[sind] = atof(sval.c_str());
                    else if (skey == "Porgx")
                        Porg_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Porgy")
                        Porg_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Porgz")
                        Porg_here[sind * 3 + 2] = atof(sval.c_str());
                    else if (skey == "Spinx")
                        Spin_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Spiny")
                        Spin_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Spinz")
                        Spin_here[sind * 3 + 2] = atof(sval.c_str());
                    else if (skey == "Pmomx")
                        Pmom_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Pmomy")
                        Pmom_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Pmomz")
                        Pmom_here[sind * 3 + 2] = atof(sval.c_str());
                }
            }
            inf.close();
        }
        // set initial data
        for (int lev = 0; lev < GH->levels; lev++)
        {
            MyList<Patch> *Pp = GH->PatL[lev];
            while (Pp)
            {
                MyList<Block> *BL = Pp->data->blb;
                while (BL)
                {
                    Block *cg = BL->data;
                    if (myrank == cg->rank)
                    {
                        // Use Lousto's analytic formulas to compute initial data
                        f_get_lousto_nbhs(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                                            cg->fgfs[phi0->sgfn], cg->fgfs[trK0->sgfn],
                                                            cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], 
                                                            cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
                                                            cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], 
                                                            cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn],
                                                            cg->fgfs[Gmx0->sgfn], cg->fgfs[Gmy0->sgfn], cg->fgfs[Gmz0->sgfn],
                                                            cg->fgfs[Lap0->sgfn], 
                                                            cg->fgfs[Sfx0->sgfn], cg->fgfs[Sfy0->sgfn], cg->fgfs[Sfz0->sgfn],
                                                            cg->fgfs[dtSfx0->sgfn], cg->fgfs[dtSfy0->sgfn], cg->fgfs[dtSfz0->sgfn], 
                                                            Mass_here, Porg_here, Pmom_here, Spin_here, BH_NM);
                    }
                    if (BL == Pp->data->ble)
                        break;
                    BL = BL->next;
                }
                Pp = Pp->next;
            }
        }
        // dump read_in initial data
        for (int lev = 0; lev < GH->levels; lev++)
            Parallel::Dump_Data(GH->PatL[lev], StateList, 0, PhysTime, dT);

        delete[] Porg_here;
        delete[] Mass_here;
        delete[] Pmom_here;
        delete[] Spin_here;        //   exit(0);
    }
}

//================================================================================================



//================================================================================================

// This member function computes initial data using analytic formulas by Prof. Cao

//================================================================================================

void bssn_class::Setup_Initial_Data_Cao()
{
    if (!checkrun)
    {
        if (myrank == 0)
            cout << "Setup initial data with Cao's analytical formula." << endl;
        char filename[50];
        strcpy(filename, "input.par");
        int BH_NM;
        double *Porg_here, *Pmom_here, *Spin_here, *Mass_here;
        // read parameter from file
        {
            const int LEN = 256;
            char pline[LEN];
            string str, sgrp, skey, sval;
            int sind;
            ifstream inf(filename, ifstream::in);
            if (!inf.good() && myrank == 0)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "Can not open parameter file " << filename 
                                                                << " for inputing information of black holes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i = 1; inf.good(); i++)
            {
                inf.getline(pline, LEN);
                str = pline;

                int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                if (status == -1)
                {
                    if (ErrorMonitor->outfile)
                        ErrorMonitor->outfile << "error reading parameter file " << filename << " in line " << i << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                else if (status == 0)
                    continue;

                if (sgrp == "BSSN" && skey == "BH_num")
                {
                    BH_NM = atoi(sval.c_str());
                    break;
                }
            }
            inf.close();
        }

        Porg_here = new double[3 * BH_NM];
        Pmom_here = new double[3 * BH_NM];
        Spin_here = new double[3 * BH_NM];
        Mass_here = new double[BH_NM];
        // read parameter from file
        {
            const int LEN = 256;
            char pline[LEN];
            string str, sgrp, skey, sval;
            int sind;
            ifstream inf(filename, ifstream::in);
            if (!inf.good() && myrank == 0)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "Can not open parameter file " << filename
                                                                << " for inputing information of black holes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i = 1; inf.good(); i++)
            {
                inf.getline(pline, LEN);
                str = pline;

                int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                if (status == -1)
                {
                    if (ErrorMonitor->outfile)
                        ErrorMonitor->outfile << "error reading parameter file " << filename << " in line " << i << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                else if (status == 0)
                    continue;

                if (sgrp == "BSSN" && sind < BH_NM)
                {
                    if (skey == "Mass")
                        Mass_here[sind] = atof(sval.c_str());
                    else if (skey == "Porgx")
                        Porg_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Porgy")
                        Porg_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Porgz")
                        Porg_here[sind * 3 + 2] = atof(sval.c_str());
                    else if (skey == "Spinx")
                        Spin_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Spiny")
                        Spin_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Spinz")
                        Spin_here[sind * 3 + 2] = atof(sval.c_str());
                    else if (skey == "Pmomx")
                        Pmom_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Pmomy")
                        Pmom_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Pmomz")
                        Pmom_here[sind * 3 + 2] = atof(sval.c_str());
                }
            }
            inf.close();
        }
        // set initial data
        for (int lev = 0; lev < GH->levels; lev++)
        {
            MyList<Patch> *Pp = GH->PatL[lev];
            while (Pp)
            {
                MyList<Block> *BL = Pp->data->blb;
                while (BL)
                {
                    Block *cg = BL->data;
                    if (myrank == cg->rank)
                    {
                        // Use Prof. Cao's analytic formulas to compute initial data
                        f_get_initial_nbhs(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                                             cg->fgfs[phi0->sgfn], cg->fgfs[trK0->sgfn],
                                                             cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], 
                                                             cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
                                                             cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], 
                                                             cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn],
                                                             cg->fgfs[Gmx0->sgfn], cg->fgfs[Gmy0->sgfn], cg->fgfs[Gmz0->sgfn],
                                                             cg->fgfs[Lap0->sgfn], 
                                                             cg->fgfs[Sfx0->sgfn], cg->fgfs[Sfy0->sgfn], cg->fgfs[Sfz0->sgfn],
                                                             cg->fgfs[dtSfx0->sgfn], cg->fgfs[dtSfy0->sgfn], cg->fgfs[dtSfz0->sgfn], 
                                                             Mass_here, Porg_here, Pmom_here, Spin_here, BH_NM);
                    }
                    if (BL == Pp->data->ble)
                        break;
                    BL = BL->next;
                }
                Pp = Pp->next;
            }
        }
        // dump read_in initial data
        for (int lev = 0; lev < GH->levels; lev++)
            Parallel::Dump_Data(GH->PatL[lev], StateList, 0, PhysTime, dT);

        delete[] Porg_here;
        delete[] Mass_here;
        delete[] Pmom_here;
        delete[] Spin_here;        //   exit(0);
    }
}

//================================================================================================



//================================================================================================

// This member function computes Kerr-Schild initial data analytically

//================================================================================================

void bssn_class::Setup_KerrSchild()
{
    if (!checkrun)
    {
        // set initial data
        for (int lev = 0; lev < GH->levels; lev++)
        {
            MyList<Patch> *Pp = GH->PatL[lev];
            while (Pp)
            {
                MyList<Block> *BL = Pp->data->blb;
                while (BL)
                {
                    Block *cg = BL->data;
                    if (myrank == cg->rank)
                    {
                        f_get_initial_kerrschild(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                                                         cg->fgfs[phi0->sgfn], cg->fgfs[trK0->sgfn],
                                                                         cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], 
                                                                         cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
                                                                         cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], 
                                                                         cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn],
                                                                         cg->fgfs[Gmx0->sgfn], cg->fgfs[Gmy0->sgfn], cg->fgfs[Gmz0->sgfn],
                                                                         cg->fgfs[Lap0->sgfn], 
                                                                         cg->fgfs[Sfx0->sgfn], cg->fgfs[Sfy0->sgfn], cg->fgfs[Sfz0->sgfn],
                                                                         cg->fgfs[dtSfx0->sgfn], cg->fgfs[dtSfy0->sgfn], cg->fgfs[dtSfz0->sgfn]);
                    }
                    if (BL == Pp->data->ble)
                        break;
                    BL = BL->next;
                }
                Pp = Pp->next;
            }
        }
        // dump read_in initial data        //   for(int lev=0;lev<GH->levels;lev++) Parallel::Dump_Data(GH->PatL[lev],StateList,0,PhysTime,dT);        //   exit(0);
    }
}

//================================================================================================



//================================================================================================

// This member function reads initial data produced by Pablo Galaviz's Olliptic program

//================================================================================================

//|----------------------------------------------------------------------------
//  read ASCII file with the style of Pablo
//|----------------------------------------------------------------------------
bool bssn_class::read_Pablo_file(int *ext, double *datain, char *filename)
{
    int nx = ext[0], ny = ext[1], nz = ext[2];
    int i, j, k;
    double x, y, z;
    //|--->open in put file
    ifstream infile;
    infile.open(filename);
    if (!infile)
    {
        cout << "bssn_class: read_Pablo_file can't open " << filename << " for input." << endl;
        return false;
    }
    for (k = 0; k < nz; k++)
        for (j = 0; j < ny; j++)
            for (i = 0; i < nx; i++)
            {
                infile >> x >> y >> z >> datain[i + j * nx + k * nx * ny];
            }

    infile.close();

    return true;
}

//================================================================================================



//================================================================================================

// This member function writes initial data for Pablo Galaviz's Olliptic program

//================================================================================================

//|----------------------------------------------------------------------------
//  write ASCII file with the style of Pablo
//|----------------------------------------------------------------------------
void bssn_class::write_Pablo_file(int *ext, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
                                                                    char *filename)
{
    int nx = ext[0], ny = ext[1], nz = ext[2];
    int i, j, k;
    double *X, *Y, *Z;
    X = new double[nx];
    Y = new double[ny];
    Z = new double[nz];
    double dX, dY, dZ;

    dX = (xmax - xmin) / nx;
    for (i = 0; i < nx; i++)
        X[i] = xmin + (i + 0.5) * dX;
    dY = (ymax - ymin) / ny;
    for (j = 0; j < ny; j++)
        Y[j] = ymin + (j + 0.5) * dY;
    dZ = (zmax - zmin) / nz;
    for (k = 0; k < nz; k++)
        Z[k] = zmin + (k + 0.5) * dZ;

    //|--->open out put file
    ofstream outfile;
    outfile.open(filename);
    if (!outfile)
    {
        cout << "bssn=_class: write_Pablo_file can't open " << filename << " for output." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    outfile.setf(ios::scientific, ios::floatfield);
    outfile.precision(16);
    for (k = 0; k < nz; k++)
        for (j = 0; j < ny; j++)
            for (i = 0; i < nx; i++)
            {
                outfile << X[i] << " " << Y[j] << " " << Z[k] << " "
                                << 0 << endl;
            }
    outfile.close();

    delete[] X;
    delete[] Y;
    delete[] Z;
}

//================================================================================================




//================================================================================================

// This member function reads TwoPuncture initial data produced by the Ansorg solver

//================================================================================================

// Read initial data solved by Ansorg, PRD 70, 064011 (2004)

void bssn_class::Read_Ansorg()
{
    if (!checkrun)
    {
        if (myrank == 0)
            cout << "Read initial data from Ansorg's solver,"
                     << " please be sure the input parameters for black holes are puncture parameters!!" << endl;
        char filename[50];
        strcpy(filename, "input.par");
        int BH_NM;
        double *Porg_here, *Pmom_here, *Spin_here, *Mass_here;
        // read parameter from file
        {
            const int LEN = 256;
            char pline[LEN];
            string str, sgrp, skey, sval;
            int sind;
            ifstream inf(filename, ifstream::in);
            if (!inf.good() && myrank == 0)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "Can not open parameter file " << filename 
                                                                << " for inputing information of black holes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i = 1; inf.good(); i++)
            {
                inf.getline(pline, LEN);
                str = pline;

                int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                if (status == -1)
                {
                    if (ErrorMonitor->outfile)
                        ErrorMonitor->outfile << "error reading parameter file " << filename << " in line " << i << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                else if (status == 0)
                    continue;

                if (sgrp == "BSSN" && skey == "BH_num")
                {
                    BH_NM = atoi(sval.c_str());
                    break;
                }
            }
            inf.close();
        }

        Porg_here = new double[3 * BH_NM];
        Pmom_here = new double[3 * BH_NM];
        Spin_here = new double[3 * BH_NM];
        Mass_here = new double[BH_NM];
        // read parameter from file
        {
            const int LEN = 256;
            char pline[LEN];
            string str, sgrp, skey, sval;
            int sind;
            ifstream inf(filename, ifstream::in);
            if (!inf.good() && myrank == 0)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "Can not open parameter file " << filename
                                                                << " for inputing information of black holes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i = 1; inf.good(); i++)
            {
                inf.getline(pline, LEN);
                str = pline;

                int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                if (status == -1)
                {
                    if (ErrorMonitor->outfile)
                        ErrorMonitor->outfile << "error reading parameter file " << filename << " in line " << i << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                else if (status == 0)
                    continue;

                if (sgrp == "BSSN" && sind < BH_NM)
                {
                    if (skey == "Mass")
                        Mass_here[sind] = atof(sval.c_str());
                    else if (skey == "Porgx")
                        Porg_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Porgy")
                        Porg_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Porgz")
                        Porg_here[sind * 3 + 2] = atof(sval.c_str());
                    else if (skey == "Spinx")
                        Spin_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Spiny")
                        Spin_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Spinz")
                        Spin_here[sind * 3 + 2] = atof(sval.c_str());
                    else if (skey == "Pmomx")
                        Pmom_here[sind * 3] = atof(sval.c_str());
                    else if (skey == "Pmomy")
                        Pmom_here[sind * 3 + 1] = atof(sval.c_str());
                    else if (skey == "Pmomz")
                        Pmom_here[sind * 3 + 2] = atof(sval.c_str());
                }
            }
            inf.close();
        }

        int order = 6;
        Ansorg read_ansorg("Ansorg.psid", order);
        // set initial data
        for (int lev = 0; lev < GH->levels; lev++)
        {
            MyList<Patch> *Pp = GH->PatL[lev];
            while (Pp)
            {
                MyList<Block> *BL = Pp->data->blb;
                while (BL)
                {
                    Block *cg = BL->data;
                    if (myrank == cg->rank)
                    {
                        for (int k = 0; k < cg->shape[2]; k++)
                            for (int j = 0; j < cg->shape[1]; j++)
                                for (int i = 0; i < cg->shape[0]; i++)
                                    cg->fgfs[phi0->sgfn][i + j * cg->shape[0] + k * cg->shape[0] * cg->shape[1]] =
                                            read_ansorg.ps_u_at_xyz(cg->X[0][i], cg->X[1][j], cg->X[2][k]);

                        f_get_ansorg_nbhs(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                                            cg->fgfs[phi0->sgfn], cg->fgfs[trK0->sgfn],
                                                            cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], 
                                                            cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
                                                            cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], 
                                                            cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn],
                                                            cg->fgfs[Gmx0->sgfn], cg->fgfs[Gmy0->sgfn], cg->fgfs[Gmz0->sgfn],
                                                            cg->fgfs[Lap0->sgfn], 
                                                            cg->fgfs[Sfx0->sgfn], cg->fgfs[Sfy0->sgfn], cg->fgfs[Sfz0->sgfn],
                                                            cg->fgfs[dtSfx0->sgfn], cg->fgfs[dtSfy0->sgfn], cg->fgfs[dtSfz0->sgfn],
                                                            Mass_here, Porg_here, Pmom_here, Spin_here, BH_NM);
                    }
                    if (BL == Pp->data->ble)
                        break;
                    BL = BL->next;
                }
                Pp = Pp->next;
            }
        }

        delete[] Porg_here;
        delete[] Mass_here;
        delete[] Pmom_here;
        delete[] Spin_here;

        Compute_Constraint();
        // dump read_in initial data
        for (int lev = 0; lev < GH->levels; lev++)
            Parallel::Dump_Data(GH->PatL[lev], DumpList, 0, PhysTime, dT);

        //   if(myrank==0) MPI_Abort(MPI_COMM_WORLD,1);
    }
}

//================================================================================================

void bssn_class::move_to_gpu() {
    Helper::move_to_gpu_whole(GH, myrank, StateList);
    Helper::move_to_gpu_whole(GH, myrank, RHSList);
    Helper::move_to_gpu_whole(GH, myrank, MiscList);
    Helper::move_to_gpu_whole(GH, myrank, SynchList_pre);
    Helper::move_to_gpu_whole(GH, myrank, SynchList_cor);
    Helper::move_to_gpu_whole(GH, myrank, ConstraintList);
    Helper::move_to_gpu_whole(GH, myrank, DGList);       
}

void bssn_class::move_to_cpu() {
    Helper::move_to_cpu_whole(GH, myrank, StateList);
    Helper::move_to_cpu_whole(GH, myrank, RHSList);
    Helper::move_to_cpu_whole(GH, myrank, MiscList);
    Helper::move_to_cpu_whole(GH, myrank, SynchList_pre);
    Helper::move_to_cpu_whole(GH, myrank, SynchList_cor);
    Helper::move_to_cpu_whole(GH, myrank, ConstraintList);
    Helper::move_to_cpu_whole(GH, myrank, DGList);       
}

//================================================================================================

// This member function sets up the time evolution for the entire process

//================================================================================================

void bssn_class::Evolve(int Steps)
{

    double prev_clock, curr_clock = MPI_Wtime();
    double LastDump = 0.0, LastCheck = 0.0, Last2dDump = 0.0;
    LastAnas = 0;

    double beg_time;
    beg_time = MPI_Wtime();
    use_gpu = 1;
    // if (myrank % 2 == 1)
    //     use_gpu = 0;
    // use_gpu = 0;

    // for step 0 constraint interpolation
    Interp_Constraint(true);

    if (checkrun)
        CheckPoint->read_bssn(LastDump, Last2dDump, LastAnas);

    double dT_mon = dT * pow(0.5, Mymax(0, trfls));
    perf bssn_perf;
    size_t current_min, current_avg, current_max, peak_min, peak_avg, peak_max;

    for (int lev = 0; lev < GH->levels; lev++)
        GH->Lt[lev] = PhysTime;

    GH->settrfls(trfls);

    for (int ncount = 1; ncount < Steps + 1; ncount++) {
        cout << "Before Step: " << ncount << " My Rank: " << myrank 
                 << " takes " << MPI_Wtime() - beg_time << " seconds!" << endl;
        beg_time = MPI_Wtime();
        RecursiveStep(0);
        cout << "After Step: " << ncount << " My Rank: " << myrank 
                 << " takes " << MPI_Wtime() - beg_time << " seconds!" << endl;
        beg_time = MPI_Wtime();

        Constraint_Out(); // this will affect the Dump_List

        LastDump += dT_mon;
        Last2dDump += dT_mon;
        LastCheck += dT_mon;

        if (LastDump >= DumpTime)
        {

            for (int lev = 0; lev < GH->levels; lev++)
                Parallel::Dump_Data_GPU(GH->PatL[lev], DumpList, 0, PhysTime, dT_mon);

            LastDump = 0;

            if (myrank == 0)
            {
                cout << "Dump done." << endl;
            }
        }

        if (Last2dDump >= d2DumpTime)
        {
            //       misc::tillherecheck("before 2dDump_Data");

            for (int lev = 0; lev < GH->levels; lev++)
                Parallel::d2Dump_Data_GPU(GH->PatL[lev], DumpList, 0, PhysTime, dT_mon);

            Last2dDump = 0;

            if (myrank == 0)
            {
                cout << "2dDump done." << endl;
            }
        }

        if (myrank == 0)
        {
            prev_clock = curr_clock;
            curr_clock = MPI_Wtime();// clock();
            cout << "Timestep # " << ncount << ": integrating to time: " << PhysTime << endl;
            cout << "used " << curr_clock - prev_clock << " seconds!" << endl;
            // cout << "used " << (double)(curr_clock - prev_clock) / ((double)CLOCKS_PER_SEC) << " seconds!" << endl;
        }

        if (PhysTime >= TotalTime)
            break;

        bssn_perf.MemoryUsage(&current_min, &current_avg, &current_max,
                                                    &peak_min, &peak_avg, &peak_max, nprocs);
        if (myrank == 0)
            printf("Memory usage: current %0.4lg/%0.4lg/%0.4lgMB, "
                         "peak %0.4lg/%0.4lg/%0.4lgMB\n",
                         (double)current_min / (1024.0 * 1024.0),
                         (double)current_avg / (1024.0 * 1024.0),
                         (double)current_max / (1024.0 * 1024.0),
                         (double)peak_min / (1024.0 * 1024.0),
                         (double)peak_avg / (1024.0 * 1024.0),
                         (double)peak_max / (1024.0 * 1024.0));

        if (LastCheck >= CheckTime) {
            LastCheck = 0;
            CheckPoint->write_Black_Hole_position(BH_num_input, BH_num, Porg0, Porgbr, Mass);
            CheckPoint->writecheck_cgh(PhysTime, GH);
            CheckPoint->write_bssn(LastDump, Last2dDump, LastAnas);
        }
    }
}

//================================================================================================




//================================================================================================

// This member function sets up recursive time evolution across different grid levels

//================================================================================================

void bssn_class::RecursiveStep(int lev)
{
    double dT_lev = dT * pow(0.5, Mymax(lev, trfls));

    int NoIterations = 1, YN;
    if (lev <= trfls)
        NoIterations = 1;
    else
        NoIterations = 2;

    for (int i = 0; i < NoIterations; i++)
    {
        YN = (i == NoIterations - 1) ? 1 : 0; // 1: same time level for coarse level and fine level

        if (use_gpu == 1)
            Step_GPU(lev, YN);
        else
            Step(lev, YN);

        GH->Lt[lev] += dT_lev;

        if (lev < GH->levels - 1)
        {
            int lf = lev + 1;
            RecursiveStep(lf);
        }
        else
            PhysTime += dT * pow(0.5, lev);

        RestrictProlong(lev, YN, fgt(PhysTime - dT_lev, StartTime, dT_lev / 2), StateList, OldStateList, SynchList_cor);
    }
    GH->Regrid_Onelevel(
        lev, Symmetry, BH_num, Porgbr, Porg0,
        SynchList_cor, OldStateList, StateList, SynchList_pre,
        fgt(PhysTime - dT_lev, StartTime, dT_lev / 2), ErrorMonitor
    );
}

//================================================================================================

// This member function configures the single-step time evolution for each grid level
// during the time evolution process.
// For the case PSTR == 0

//================================================================================================

void bssn_class::Step(int lev, int YN)
{
    setpbh(BH_num, Porg0, Mass, BH_num_input);

    double dT_lev = dT * pow(0.5, Mymax(lev, trfls));

// new code 2013-2-15, zjcao
    // for black hole position
    if (BH_num > 0 && lev == GH->levels - 1)
    {
        compute_Porg_rhs(Porg0, Porg_rhs, Sfx0, Sfy0, Sfz0, lev);
        for (int ithBH = 0; ithBH < BH_num; ithBH++)
        {
            for (int ith = 0; ith < 3; ith++)
                Porg1[ithBH][ith] = Porg0[ithBH][ith] + Porg_rhs[ithBH][ith] * dT_lev;
            if (Symmetry > 0)
                Porg1[ithBH][2] = fabs(Porg1[ithBH][2]);
            if (Symmetry == 2)
            {
                Porg1[ithBH][0] = fabs(Porg1[ithBH][0]);
                Porg1[ithBH][1] = fabs(Porg1[ithBH][1]);
            }
            if (!finite(Porg1[ithBH][0]) || !finite(Porg1[ithBH][1]) || !finite(Porg1[ithBH][2]))
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "predictor step finds NaN for BH's position from ("
                                                                << Porg0[ithBH][0] << "," << Porg0[ithBH][1] << "," << Porg0[ithBH][2] << ")" << endl;

                MyList<var> *DG_List = new MyList<var>(Sfx0);
                DG_List->insert(Sfx0);
                DG_List->insert(Sfy0);
                DG_List->insert(Sfz0);
                Parallel::Dump_Data(GH->PatL[lev], DG_List, 0, PhysTime, dT_lev);
                DG_List->clearList();
            }
        }
    }

    // data analysis part
    // Warning NOTE: the variables1 are used as temp storege room
    if (lev == a_lev)
    {
        AnalysisStuff(lev, dT_lev);
    }

    bool BB = fgt(PhysTime, StartTime, dT_lev / 2);
    double ndeps = numepss;
    if (lev < GH->movls)
        ndeps = numepsb;
    double TRK4 = PhysTime;
    int iter_count = 0; // count RK4 substeps
    int pre = 0, cor = 1;
    int ERROR = 0;

    // Predictor
    MyList<Patch> *Pp = GH->PatL[lev];
    while (Pp)
    {
        MyList<Block> *BP = Pp->data->blb;
        while (BP)
        {
            Block *cg = BP->data;
            if (myrank == cg->rank)
            {
                f_enforce_ga(cg->shape,
                                         cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], 
                                         cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
                                         cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], 
                                         cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn]);

                if (f_compute_rhs_bssn(cg->shape, TRK4, cg->X[0], cg->X[1], cg->X[2],
                                                             cg->fgfs[phi0->sgfn], cg->fgfs[trK0->sgfn],
                                                             cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], 
                                                             cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
                                                             cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], 
                                                             cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn],
                                                             cg->fgfs[Gmx0->sgfn], cg->fgfs[Gmy0->sgfn], cg->fgfs[Gmz0->sgfn],
                                                             cg->fgfs[Lap0->sgfn], 
                                                             cg->fgfs[Sfx0->sgfn], cg->fgfs[Sfy0->sgfn], cg->fgfs[Sfz0->sgfn],
                                                             cg->fgfs[dtSfx0->sgfn], cg->fgfs[dtSfy0->sgfn], cg->fgfs[dtSfz0->sgfn],
                                                             cg->fgfs[phi_rhs->sgfn], cg->fgfs[trK_rhs->sgfn],
                                                             cg->fgfs[gxx_rhs->sgfn], cg->fgfs[gxy_rhs->sgfn], cg->fgfs[gxz_rhs->sgfn],
                                                             cg->fgfs[gyy_rhs->sgfn], cg->fgfs[gyz_rhs->sgfn], cg->fgfs[gzz_rhs->sgfn],
                                                             cg->fgfs[Axx_rhs->sgfn], cg->fgfs[Axy_rhs->sgfn], cg->fgfs[Axz_rhs->sgfn],
                                                             cg->fgfs[Ayy_rhs->sgfn], cg->fgfs[Ayz_rhs->sgfn], cg->fgfs[Azz_rhs->sgfn],
                                                             cg->fgfs[Gmx_rhs->sgfn], cg->fgfs[Gmy_rhs->sgfn], cg->fgfs[Gmz_rhs->sgfn],
                                                             cg->fgfs[Lap_rhs->sgfn], 
                                                             cg->fgfs[Sfx_rhs->sgfn], cg->fgfs[Sfy_rhs->sgfn], cg->fgfs[Sfz_rhs->sgfn],
                                                             cg->fgfs[dtSfx_rhs->sgfn], cg->fgfs[dtSfy_rhs->sgfn], cg->fgfs[dtSfz_rhs->sgfn],
                                                             cg->fgfs[rho->sgfn], cg->fgfs[Sx->sgfn], cg->fgfs[Sy->sgfn], cg->fgfs[Sz->sgfn],
                                                             cg->fgfs[Sxx->sgfn], cg->fgfs[Sxy->sgfn], cg->fgfs[Sxz->sgfn], 
                                                             cg->fgfs[Syy->sgfn], cg->fgfs[Syz->sgfn], cg->fgfs[Szz->sgfn],
                                                             cg->fgfs[Gamxxx->sgfn], cg->fgfs[Gamxxy->sgfn], cg->fgfs[Gamxxz->sgfn],
                                                             cg->fgfs[Gamxyy->sgfn], cg->fgfs[Gamxyz->sgfn], cg->fgfs[Gamxzz->sgfn],
                                                             cg->fgfs[Gamyxx->sgfn], cg->fgfs[Gamyxy->sgfn], cg->fgfs[Gamyxz->sgfn],
                                                             cg->fgfs[Gamyyy->sgfn], cg->fgfs[Gamyyz->sgfn], cg->fgfs[Gamyzz->sgfn],
                                                             cg->fgfs[Gamzxx->sgfn], cg->fgfs[Gamzxy->sgfn], cg->fgfs[Gamzxz->sgfn],
                                                             cg->fgfs[Gamzyy->sgfn], cg->fgfs[Gamzyz->sgfn], cg->fgfs[Gamzzz->sgfn],
                                                             cg->fgfs[Rxx->sgfn], cg->fgfs[Rxy->sgfn], cg->fgfs[Rxz->sgfn], 
                                                             cg->fgfs[Ryy->sgfn], cg->fgfs[Ryz->sgfn], cg->fgfs[Rzz->sgfn],
                                                             cg->fgfs[Cons_Ham->sgfn],
                                                             cg->fgfs[Cons_Px->sgfn], cg->fgfs[Cons_Py->sgfn], cg->fgfs[Cons_Pz->sgfn],
                                                             cg->fgfs[Cons_Gx->sgfn], cg->fgfs[Cons_Gy->sgfn], cg->fgfs[Cons_Gz->sgfn],
                                                             Symmetry, lev, ndeps, pre))
                {
                    cout << "find NaN in domain: (" 
                             << cg->bbox[0] << ":" << cg->bbox[3] << "," 
                             << cg->bbox[1] << ":" << cg->bbox[4] << ","
                             << cg->bbox[2] << ":" << cg->bbox[5] << ")" << endl;
                    ERROR = 1;
                }

                // rk4 substep and boundary
                {
                    MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varlrhs = RHSList; 
                    // we do not check the correspondence here
                    
                    while (varl0)
                    {
                        if (lev == 0) // sommerfeld indeed
                            f_sommerfeld_routbam(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                                                     Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], 
                                                                     Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
                                                                     cg->fgfs[varlrhs->data->sgfn],
                                                                     cg->fgfs[varl0->data->sgfn], 
                                                                     varl0->data->propspeed, varl0->data->SoA,
                                                                     Symmetry);
                        f_rungekutta4_rout(cg->shape, dT_lev, 
                                                             cg->fgfs[varl0->data->sgfn], 
                                                             cg->fgfs[varl->data->sgfn], 
                                                             cg->fgfs[varlrhs->data->sgfn],
                                                             iter_count);
                        if (lev > 0) // fix BD point
                            f_sommerfeld_rout(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                                                Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], 
                                                                Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
                                                                dT_lev, 
                                                                cg->fgfs[phi0->sgfn],
                                                                cg->fgfs[Lap0->sgfn], 
                                                                cg->fgfs[varl0->data->sgfn], cg->fgfs[varl->data->sgfn], 
                                                                varl0->data->SoA,
                                                                Symmetry, cor);

                        varl0 = varl0->next;
                        varl = varl->next;
                        varlrhs = varlrhs->next;
                    }
                }
                f_lowerboundset(cg->shape, cg->fgfs[phi->sgfn], chitiny);
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }
    // check error information
    {
        int erh = ERROR;
        MPI_Allreduce(&erh, &ERROR, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    if (ERROR)
    {
        Parallel::Dump_Data(GH->PatL[lev], StateList, 0, PhysTime, dT_lev);
        if (myrank == 0)
        {
            if (ErrorMonitor->outfile)
                ErrorMonitor->outfile << "find NaN in state variables at t = " << PhysTime << ", lev = " << lev << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    Parallel::Sync(GH->PatL[lev], SynchList_pre, Symmetry);

    // corrector
    for (iter_count = 1; iter_count < 4; iter_count++)
    {
        // for RK4: t0, t0+dt/2, t0+dt/2, t0+dt;
        if (iter_count == 1 || iter_count == 3)
            TRK4 += dT_lev / 2;
        Pp = GH->PatL[lev];
        while (Pp)
        {
            MyList<Block> *BP = Pp->data->blb;
            while (BP)
            {
                Block *cg = BP->data;
                if (myrank == cg->rank)
                {
                    f_enforce_ga(cg->shape,
                                             cg->fgfs[gxx->sgfn], cg->fgfs[gxy->sgfn], cg->fgfs[gxz->sgfn], 
                                             cg->fgfs[gyy->sgfn], cg->fgfs[gyz->sgfn], cg->fgfs[gzz->sgfn],
                                             cg->fgfs[Axx->sgfn], cg->fgfs[Axy->sgfn], cg->fgfs[Axz->sgfn], 
                                             cg->fgfs[Ayy->sgfn], cg->fgfs[Ayz->sgfn], cg->fgfs[Azz->sgfn]);
                    if (f_compute_rhs_bssn(cg->shape, TRK4, cg->X[0], cg->X[1], cg->X[2],
                                                                 cg->fgfs[phi->sgfn], cg->fgfs[trK->sgfn],
                                                                 cg->fgfs[gxx->sgfn], cg->fgfs[gxy->sgfn], cg->fgfs[gxz->sgfn], 
                                                                 cg->fgfs[gyy->sgfn], cg->fgfs[gyz->sgfn], cg->fgfs[gzz->sgfn],
                                                                 cg->fgfs[Axx->sgfn], cg->fgfs[Axy->sgfn], cg->fgfs[Axz->sgfn], 
                                                                 cg->fgfs[Ayy->sgfn], cg->fgfs[Ayz->sgfn], cg->fgfs[Azz->sgfn],
                                                                 cg->fgfs[Gmx->sgfn], cg->fgfs[Gmy->sgfn], cg->fgfs[Gmz->sgfn],
                                                                 cg->fgfs[Lap->sgfn], 
                                                                 cg->fgfs[Sfx->sgfn], cg->fgfs[Sfy->sgfn], cg->fgfs[Sfz->sgfn],
                                                                 cg->fgfs[dtSfx->sgfn], cg->fgfs[dtSfy->sgfn], cg->fgfs[dtSfz->sgfn],
                                                                 cg->fgfs[phi1->sgfn], cg->fgfs[trK1->sgfn],
                                                                 cg->fgfs[gxx1->sgfn], cg->fgfs[gxy1->sgfn], cg->fgfs[gxz1->sgfn],
                                                                 cg->fgfs[gyy1->sgfn], cg->fgfs[gyz1->sgfn], cg->fgfs[gzz1->sgfn],
                                                                 cg->fgfs[Axx1->sgfn], cg->fgfs[Axy1->sgfn], cg->fgfs[Axz1->sgfn],
                                                                 cg->fgfs[Ayy1->sgfn], cg->fgfs[Ayz1->sgfn], cg->fgfs[Azz1->sgfn],
                                                                 cg->fgfs[Gmx1->sgfn], cg->fgfs[Gmy1->sgfn], cg->fgfs[Gmz1->sgfn],
                                                                 cg->fgfs[Lap1->sgfn], 
                                                                 cg->fgfs[Sfx1->sgfn], cg->fgfs[Sfy1->sgfn], cg->fgfs[Sfz1->sgfn],
                                                                 cg->fgfs[dtSfx1->sgfn], cg->fgfs[dtSfy1->sgfn], cg->fgfs[dtSfz1->sgfn],
                                                                 cg->fgfs[rho->sgfn], 
                                                                 cg->fgfs[Sx->sgfn], cg->fgfs[Sy->sgfn], cg->fgfs[Sz->sgfn],
                                                                 cg->fgfs[Sxx->sgfn], cg->fgfs[Sxy->sgfn], cg->fgfs[Sxz->sgfn], 
                                                                 cg->fgfs[Syy->sgfn], cg->fgfs[Syz->sgfn], cg->fgfs[Szz->sgfn],
                                                                 cg->fgfs[Gamxxx->sgfn], cg->fgfs[Gamxxy->sgfn], cg->fgfs[Gamxxz->sgfn],
                                                                 cg->fgfs[Gamxyy->sgfn], cg->fgfs[Gamxyz->sgfn], cg->fgfs[Gamxzz->sgfn],
                                                                 cg->fgfs[Gamyxx->sgfn], cg->fgfs[Gamyxy->sgfn], cg->fgfs[Gamyxz->sgfn],
                                                                 cg->fgfs[Gamyyy->sgfn], cg->fgfs[Gamyyz->sgfn], cg->fgfs[Gamyzz->sgfn],
                                                                 cg->fgfs[Gamzxx->sgfn], cg->fgfs[Gamzxy->sgfn], cg->fgfs[Gamzxz->sgfn],
                                                                 cg->fgfs[Gamzyy->sgfn], cg->fgfs[Gamzyz->sgfn], cg->fgfs[Gamzzz->sgfn],
                                                                 cg->fgfs[Rxx->sgfn], cg->fgfs[Rxy->sgfn], cg->fgfs[Rxz->sgfn], 
                                                                 cg->fgfs[Ryy->sgfn], cg->fgfs[Ryz->sgfn], cg->fgfs[Rzz->sgfn],
                                                                 cg->fgfs[Cons_Ham->sgfn],
                                                                 cg->fgfs[Cons_Px->sgfn], cg->fgfs[Cons_Py->sgfn], cg->fgfs[Cons_Pz->sgfn],
                                                                 cg->fgfs[Cons_Gx->sgfn], cg->fgfs[Cons_Gy->sgfn], cg->fgfs[Cons_Gz->sgfn],
                                                                 Symmetry, lev, ndeps, cor))
                    {
                        cout << "find NaN in domain: (" 
                                 << cg->bbox[0] << ":" << cg->bbox[3] << "," 
                                 << cg->bbox[1] << ":" << cg->bbox[4] << ","
                                 << cg->bbox[2] << ":" << cg->bbox[5] << ")" << endl;
                        ERROR = 1;
                    }
                    // rk4 substep and boundary
                    {
                        MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varl1 = SynchList_cor, *varlrhs = RHSList; 
                        // we do not check the correspondence here
                        
                        while (varl0)
                        {
                            if (lev == 0) // sommerfeld indeed
                                f_sommerfeld_routbam(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                                                         Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], 
                                                                         Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
                                                                         cg->fgfs[varl1->data->sgfn],
                                                                         cg->fgfs[varl->data->sgfn], 
                                                                         varl0->data->propspeed, varl0->data->SoA,
                                                                         Symmetry);
                            f_rungekutta4_rout(cg->shape, dT_lev, 
                                                                 cg->fgfs[varl0->data->sgfn], 
                                                                 cg->fgfs[varl1->data->sgfn], 
                                                                 cg->fgfs[varlrhs->data->sgfn],
                                                                 iter_count);

                            if (lev > 0) // fix BD point
                                f_sommerfeld_rout(cg->shape, cg->X[0], cg->X[1], cg->X[2],
                                                                    Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], 
                                                                    Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
                                                                    dT_lev, 
                                                                    cg->fgfs[phi0->sgfn],
                                                                    cg->fgfs[Lap0->sgfn], 
                                                                    cg->fgfs[varl0->data->sgfn], cg->fgfs[varl1->data->sgfn], 
                                                                    varl0->data->SoA,
                                                                    Symmetry, cor);

                            varl0 = varl0->next;
                            varl = varl->next;
                            varl1 = varl1->next;
                            varlrhs = varlrhs->next;
                        }
                    }
                    f_lowerboundset(cg->shape, cg->fgfs[phi1->sgfn], chitiny);
                }
                if (BP == Pp->data->ble)
                    break;
                BP = BP->next;
            }
            Pp = Pp->next;
        }

        // check error information
        {
            int erh = ERROR;
            MPI_Allreduce(&erh, &ERROR, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }

        if (ERROR)
        {
            Parallel::Dump_Data(GH->PatL[lev], SynchList_pre, 0, PhysTime, dT_lev);
            if (myrank == 0)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "find NaN in RK4 substep#" << iter_count 
                                                                << " variables at t = " << PhysTime 
                                                                << ", lev = " << lev << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        Parallel::Sync(GH->PatL[lev], SynchList_cor, Symmetry);

        // swap time level
        if (iter_count < 3)
        {
            Pp = GH->PatL[lev];
            while (Pp)
            {
                MyList<Block> *BP = Pp->data->blb;
                while (BP)
                {
                    Block *cg = BP->data;
                    cg->swapList(SynchList_pre, SynchList_cor, myrank);
                    if (BP == Pp->data->ble)
                        break;
                    BP = BP->next;
                }
                Pp = Pp->next;
            }
        }
    }
    // note the data structure before update
    // SynchList_cor 1   -----------
    //
    // StateList     0   -----------
    //
    // OldStateList  old -----------
    // update
    Pp = GH->PatL[lev];
    while (Pp)
    {
        MyList<Block> *BP = Pp->data->blb;
        while (BP)
        {
            Block *cg = BP->data;
            cg->swapList(StateList, SynchList_cor, myrank);
            cg->swapList(OldStateList, SynchList_cor, myrank);
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }
    // for black hole position
    if (BH_num > 0 && lev == GH->levels - 1)
    {
        for (int ithBH = 0; ithBH < BH_num; ithBH++)
        {
            Porg0[ithBH][0] = Porg1[ithBH][0];
            Porg0[ithBH][1] = Porg1[ithBH][1];
            Porg0[ithBH][2] = Porg1[ithBH][2];
        }
    }
}

//================================================================================================



//================================================================================================

// This member function sets up the single-step time evolution for each grid level (alternate version)

//================================================================================================



//================================================================================================



//================================================================================================

// This member function sets up the single-step time evolution for each grid level
// For the case PSTR == 1

//================================================================================================



//================================================================================================



//================================================================================================

// 0: do not use mixing two levels data for OutBD; 1: do use

#define MIXOUTB 0
// we assume
// StateList      1   -----------
//
// OldStateList   0   -----------
//
// SynchList_cor  old -----------
void bssn_class::RestrictProlong(
    int lev, int YN, bool BB,
    MyList<var> *SL, MyList<var> *OL, MyList<var> *corL
) {
    if (lev > 0) {
        MyList<Patch> *Pp, *Ppc;
        if (lev > trfls && YN == 0) { // time refinement levels and for intermediat time level
            Pp = GH->PatL[lev - 1];
            while (Pp) {
                // if (BB) Parallel::prepare_inter_time_level(Pp->data, SL, OL, corL, SynchList_pre, 0); // use SynchList_pre as temporal storage space
                // else Parallel::prepare_inter_time_level(Pp->data, SL, OL, SynchList_pre, 0); // use SynchList_pre as temporal storage space
                if (BB) Parallel::gpu_prepare_inter_time_level(Pp->data, SL, OL, corL, SynchList_pre, 0); // use SynchList_pre as temporal storage space
                else Parallel::gpu_prepare_inter_time_level(Pp->data, SL, OL, SynchList_pre, 0); // use SynchList_pre as temporal storage space
                Pp = Pp->next;
            }
            Parallel::Restrict_GPU(GH->PatL[lev - 1], GH->PatL[lev], SL, SynchList_pre, Symmetry);
            // Parallel::Restrict(GH->PatL[lev - 1], GH->PatL[lev], SL, SynchList_pre, Symmetry);
            Parallel::Sync_GPU(GH->PatL[lev - 1], SynchList_pre, Symmetry);
            // Parallel::Sync(GH->PatL[lev - 1], SynchList_pre, Symmetry);
            Ppc = GH->PatL[lev - 1];
            while (Ppc) {
                Pp = GH->PatL[lev];
                while (Pp) {
                    Parallel::OutBdLow2Hi_GPU(Ppc->data, Pp->data, SynchList_pre, SL, Symmetry);
                    // Parallel::OutBdLow2Hi(Ppc->data, Pp->data, SynchList_pre, SL, Symmetry);
                    Pp = Pp->next;
                }
                Ppc = Ppc->next;
            }
        }
        else { // no time refinement levels and for all same time levels
            Parallel::Restrict_GPU(GH->PatL[lev - 1], GH->PatL[lev], SL, SL, Symmetry);
            // Parallel::Restrict(GH->PatL[lev - 1], GH->PatL[lev], SL, SL, Symmetry);
            Parallel::Sync_GPU(GH->PatL[lev - 1], SL, Symmetry);
            // Parallel::Sync(GH->PatL[lev - 1], SL, Symmetry);

            Ppc = GH->PatL[lev - 1];
            while (Ppc) {
                Pp = GH->PatL[lev];
                while (Pp) {
                    Parallel::OutBdLow2Hi_GPU(Ppc->data, Pp->data, SL, SL, Symmetry);
                    // Parallel::OutBdLow2Hi(Ppc->data, Pp->data, SL, SL, Symmetry);
                    Pp = Pp->next;
                }
                Ppc = Ppc->next;
            }
        }
        Parallel::Sync_GPU(GH->PatL[lev], SL, Symmetry);
        // Parallel::Sync(GH->PatL[lev], SL, Symmetry);
    }
}

//================================================================================================



//================================================================================================

// auxiliary operation, input lev means original lev-1

void bssn_class::RestrictProlong_aux(int lev, int YN, bool BB,
                                                                         MyList<var> *SL, MyList<var> *OL, MyList<var> *corL)
// we assume
// StateList      1   -----------
//
// OldStateList   0   -----------
//
// SynchList_cor  old -----------
{
    //  misc::tillherecheck(GH->Commlev[lev],GH->start_rank[lev],"starting RestrictProlong_aux");

    if (lev >= GH->levels - 1)
        return;
    lev = lev + 1;

    if (lev > 0)
    {
        MyList<Patch> *Pp, *Ppc;
        if (lev > trfls && YN == 0) // time refinement levels and for intermediat time level
        {
            Pp = GH->PatL[lev - 1];
            while (Pp)
            {
                if (BB)
                    Parallel::prepare_inter_time_level(Pp->data, SL, OL, corL,
                                                                                         SynchList_pre, 0); // use SynchList_pre as temporal storage space
                else
                    Parallel::prepare_inter_time_level(Pp->data, SL, OL,
                                                                                         SynchList_pre, 0); // use SynchList_pre as temporal storage space
                Pp = Pp->next;
            }

            Parallel::Restrict(GH->PatL[lev - 1], GH->PatL[lev], SL, SynchList_pre, Symmetry);

            Parallel::Sync(GH->PatL[lev - 1], SynchList_pre, Symmetry);

            Ppc = GH->PatL[lev - 1];
            while (Ppc)
            {
                Pp = GH->PatL[lev];
                while (Pp)
                {
                    Parallel::OutBdLow2Hi(Ppc->data, Pp->data, SynchList_pre, SL, Symmetry);
                    Pp = Pp->next;
                }
                Ppc = Ppc->next;
            }
        }
        else // no time refinement levels and for all same time levels
        {
            Parallel::Restrict(GH->PatL[lev - 1], GH->PatL[lev], SL, SL, Symmetry);

            Parallel::Sync(GH->PatL[lev - 1], SL, Symmetry);

            Ppc = GH->PatL[lev - 1];
            while (Ppc)
            {
                Pp = GH->PatL[lev];
                while (Pp)
                {
                    Parallel::OutBdLow2Hi(Ppc->data, Pp->data, SL, SL, Symmetry);
                    Pp = Pp->next;
                }
                Ppc = Ppc->next;
            }
        }

        Parallel::Sync(GH->PatL[lev], SL, Symmetry);
    }
}

//================================================================================================



//================================================================================================

void bssn_class::RestrictProlong(int lev, int YN, bool BB)
{
    double dT_lev = dT * pow(0.5, Mymax(lev, trfls));
    // we assume  for fine
    // SynchList_cor 1   -----------
    //
    // StateList     0   -----------
    //
    // OldStateList  old -----------
    //            for coarse
    // StateList      1   -----------
    //
    // OldStateList   0   -----------
    //
    // SynchList_cor  old -----------
    if (lev > 0)
    {
        MyList<Patch> *Pp, *Ppc;
        if (lev > trfls && YN == 0) // time refinement levels and for intermediat time level
        {
            if (myrank == 0)
                cout << "/=: " << GH->Lt[lev - 1] << "," << GH->Lt[lev] + dT_lev << endl;
            Pp = GH->PatL[lev - 1];
            while (Pp)
            {
                if (BB)
                    Parallel::prepare_inter_time_level(Pp->data, StateList, OldStateList, SynchList_cor,
                                                                                         SynchList_pre, 0); // use SynchList_pre as temporal storage space
                else
                    Parallel::prepare_inter_time_level(Pp->data, StateList, OldStateList,
                                                                                         SynchList_pre, 0); // use SynchList_pre as temporal storage space
                Pp = Pp->next;
            }

            Parallel::Restrict(GH->PatL[lev - 1], GH->PatL[lev], SynchList_cor, SynchList_pre, Symmetry);

            Parallel::Sync(GH->PatL[lev - 1], SynchList_pre, Symmetry);

            Ppc = GH->PatL[lev - 1];
            while (Ppc)
            {
                Pp = GH->PatL[lev];
                while (Pp)
                {
                    Parallel::OutBdLow2Hi(Ppc->data, Pp->data, SynchList_pre, SynchList_cor, Symmetry);
                    Pp = Pp->next;
                }
                Ppc = Ppc->next;
            }
        }
        else // no time refinement levels and for all same time levels
        {
            if (myrank == 0)
                cout << "===: " << GH->Lt[lev - 1] << "," << GH->Lt[lev] + dT_lev << endl;
            Parallel::Restrict(GH->PatL[lev - 1], GH->PatL[lev], SynchList_cor, StateList, Symmetry);

            Parallel::Sync(GH->PatL[lev - 1], StateList, Symmetry);

            Ppc = GH->PatL[lev - 1];
            while (Ppc)
            {
                Pp = GH->PatL[lev];
                while (Pp)
                {
                    Parallel::OutBdLow2Hi(Ppc->data, Pp->data, StateList, SynchList_cor, Symmetry);
                    Pp = Pp->next;
                }
                Ppc = Ppc->next;
            }
        }

        Parallel::Sync(GH->PatL[lev], SynchList_cor, Symmetry);
    }
}

//================================================================================================



//================================================================================================

void bssn_class::ProlongRestrict(int lev, int YN, bool BB)
{
    if (lev > 0)
    {
        MyList<Patch> *Pp, *Ppc;
        if (lev > trfls && YN == 0) // time refinement levels and for intermediat time level
        {
            Pp = GH->PatL[lev - 1];
            while (Pp)
            {
                if (BB)
                    Parallel::prepare_inter_time_level(Pp->data, StateList, OldStateList, SynchList_cor,
                                                                                         SynchList_pre, 0); // use SynchList_pre as temporal storage space
                else
                    Parallel::prepare_inter_time_level(Pp->data, StateList, OldStateList,
                                                                                         SynchList_pre, 0); // use SynchList_pre as temporal storage space
                Pp = Pp->next;
            }

            Ppc = GH->PatL[lev - 1];
            while (Ppc)
            {
                Pp = GH->PatL[lev];
                while (Pp)
                {
                    Parallel::OutBdLow2Hi(Ppc->data, Pp->data, SynchList_pre, SynchList_cor, Symmetry);
                    Pp = Pp->next;
                }
                Ppc = Ppc->next;
            }
        }
        else // no time refinement levels and for all same time levels
        {
            Ppc = GH->PatL[lev - 1];
            while (Ppc)
            {
                Pp = GH->PatL[lev];
                while (Pp)
                {
                    Parallel::OutBdLow2Hi(Ppc->data, Pp->data, StateList, SynchList_cor, Symmetry);
                    Pp = Pp->next;
                }
                Ppc = Ppc->next;
            }

            Parallel::Restrict_after(GH->PatL[lev - 1], GH->PatL[lev], SynchList_cor, StateList, Symmetry);
            Parallel::Sync(GH->PatL[lev - 1], StateList, Symmetry);
        }

        Parallel::Sync(GH->PatL[lev], SynchList_cor, Symmetry);
    }
}
#undef MIXOUTB

//================================================================================================



//================================================================================================

// This member function computes the gravitational radiation scalar Psi4

//================================================================================================

void bssn_class::Compute_Psi4(int lev)
{
    MyList<var> *DG_List = new MyList<var>(Rpsi4);
    DG_List->insert(Ipsi4);

    MyList<Patch> *Pp = GH->PatL[lev];
    while (Pp) {
        MyList<Block> *BP = Pp->data->blb;
        while (BP) {
            Block *cg = BP->data;
            if (myrank == cg->rank) {
                // the input arguments Gamma^i_jk and R_ij do not need synch, because we do not need to derivate them
                gpu_getnp4_launch(
                    cg->stream,
                    cg->shape, cg->d_X[0], cg->d_X[1], cg->d_X[2],
                    cg->d_fgfs[phi0->sgfn], cg->d_fgfs[trK0->sgfn],
                    cg->d_fgfs[gxx0->sgfn], cg->d_fgfs[gxy0->sgfn], cg->d_fgfs[gxz0->sgfn], 
                    cg->d_fgfs[gyy0->sgfn], cg->d_fgfs[gyz0->sgfn], cg->d_fgfs[gzz0->sgfn],
                    cg->d_fgfs[Axx0->sgfn], cg->d_fgfs[Axy0->sgfn], cg->d_fgfs[Axz0->sgfn], 
                    cg->d_fgfs[Ayy0->sgfn], cg->d_fgfs[Ayz0->sgfn], cg->d_fgfs[Azz0->sgfn],
                    cg->d_fgfs[Gamxxx->sgfn], cg->d_fgfs[Gamxxy->sgfn], cg->d_fgfs[Gamxxz->sgfn],
                    cg->d_fgfs[Gamxyy->sgfn], cg->d_fgfs[Gamxyz->sgfn], cg->d_fgfs[Gamxzz->sgfn],
                    cg->d_fgfs[Gamyxx->sgfn], cg->d_fgfs[Gamyxy->sgfn], cg->d_fgfs[Gamyxz->sgfn],
                    cg->d_fgfs[Gamyyy->sgfn], cg->d_fgfs[Gamyyz->sgfn], cg->d_fgfs[Gamyzz->sgfn],
                    cg->d_fgfs[Gamzxx->sgfn], cg->d_fgfs[Gamzxy->sgfn], cg->d_fgfs[Gamzxz->sgfn],
                    cg->d_fgfs[Gamzyy->sgfn], cg->d_fgfs[Gamzyz->sgfn], cg->d_fgfs[Gamzzz->sgfn],
                    cg->d_fgfs[Rxx->sgfn], cg->d_fgfs[Rxy->sgfn], cg->d_fgfs[Rxz->sgfn], 
                    cg->d_fgfs[Ryy->sgfn], cg->d_fgfs[Ryz->sgfn], cg->d_fgfs[Rzz->sgfn],
                    cg->d_fgfs[Rpsi4->sgfn], cg->d_fgfs[Ipsi4->sgfn],
                    Symmetry
                );
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }
    Parallel::Sync_GPU(GH->PatL[lev], DG_List, Symmetry);
    // Parallel::Sync(GH->PatL[lev], DG_List, Symmetry);

    DG_List->clearList();

    //    misc::tillherecheck(GH->Commlev[lev],GH->start_rank[lev],"end of Compute_Psi4");
}

//================================================================================================



//================================================================================================

// This member function sets the puncture positions of black holes at the initial time

//================================================================================================

void bssn_class::Setup_Black_Hole_position()
{
    char filename[50];
    strcpy(filename, "input.par");
    // read parameter from file
    {
        const int LEN = 256;
        char pline[LEN];
        string str, sgrp, skey, sval;
        int sind;
        ifstream inf(filename, ifstream::in);
        if (!inf.good() && myrank == 0)
        {
            if (ErrorMonitor->outfile)
                ErrorMonitor->outfile << "Can not open parameter file " << filename 
                                                            << " for inputing information of black holes" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 1; inf.good(); i++)
        {
            inf.getline(pline, LEN);
            str = pline;

            int status = misc::parse_parts(str, sgrp, skey, sval, sind);
            if (status == -1)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "error reading parameter file " << filename 
                                                                << " in line " << i << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            else if (status == 0)
                continue;

            if (sgrp == "BSSN" && skey == "BH_num")
            {
                BH_num_input = BH_num = atoi(sval.c_str());
                break;
            }
        }
        inf.close();
    }
    // set up the data for black holes
    // these arrays will be deleted when bssn_class is deleted
    Pmom = new double[3 * BH_num];
    Spin = new double[3 * BH_num];
    Mass = new double[BH_num];
    Porg0 = new double *[BH_num];
    Porgbr = new double *[BH_num];
    Porg = new double *[BH_num];
    Porg1 = new double *[BH_num];
    Porg_rhs = new double *[BH_num];
    for (int i = 0; i < BH_num; i++)
    {
        Porg0[i] = new double[3];
        Porgbr[i] = new double[3];
        Porg[i] = new double[3];
        Porg1[i] = new double[3];
        Porg_rhs[i] = new double[3];
    }
    // read parameter from file
    {
        const int LEN = 256;
        char pline[LEN];
        string str, sgrp, skey, sval;
        int sind;
        ifstream inf(filename, ifstream::in);
        if (!inf.good() && myrank == 0)
        {
            if (ErrorMonitor->outfile)
                ErrorMonitor->outfile << "Can not open parameter file " << filename
                                                            << " for inputing information of black holes" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 1; inf.good(); i++)
        {
            inf.getline(pline, LEN);
            str = pline;

            int status = misc::parse_parts(str, sgrp, skey, sval, sind);
            if (status == -1)
            {
                if (ErrorMonitor->outfile)
                    ErrorMonitor->outfile << "error reading parameter file " << filename 
                                                                << " in line " << i << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            else if (status == 0)
                continue;

            if (sgrp == "BSSN" && sind < BH_num)
            {
                if (skey == "Mass")
                    Mass[sind] = atof(sval.c_str());
                else if (skey == "Porgx")
                    Porg0[sind][0] = atof(sval.c_str());
                else if (skey == "Porgy")
                    Porg0[sind][1] = atof(sval.c_str());
                else if (skey == "Porgz")
                    Porg0[sind][2] = atof(sval.c_str());
                else if (skey == "Spinx")
                    Spin[sind * 3] = atof(sval.c_str());
                else if (skey == "Spiny")
                    Spin[sind * 3 + 1] = atof(sval.c_str());
                else if (skey == "Spinz")
                    Spin[sind * 3 + 2] = atof(sval.c_str());
                else if (skey == "Pmomx")
                    Pmom[sind * 3] = atof(sval.c_str());
                else if (skey == "Pmomy")
                    Pmom[sind * 3 + 1] = atof(sval.c_str());
                else if (skey == "Pmomz")
                    Pmom[sind * 3 + 2] = atof(sval.c_str());
            }
        }
        inf.close();
    }
    // echo information of Black holes
    if (myrank == 0)
    {
        cout << "initial information of " << BH_num << " Black Hole(s)" << endl;
        cout << setw(16) << "Mass"
                 << setw(16) << "x"
                 << setw(16) << "y"
                 << setw(16) << "z"
                 << setw(16) << "Px"
                 << setw(16) << "Py"
                 << setw(16) << "Pz"
                 << setw(16) << "Sx"
                 << setw(16) << "Sy"
                 << setw(16) << "Sz" << endl;
        for (int i = 0; i < BH_num; i++)
        {
            cout << setw(16) << Mass[i]
                     << setw(16) << Porg0[i][0]
                     << setw(16) << Porg0[i][1]
                     << setw(16) << Porg0[i][2]
                     << setw(16) << Pmom[i * 3]
                     << setw(16) << Pmom[i * 3 + 1]
                     << setw(16) << Pmom[i * 3 + 2]
                     << setw(16) << Spin[i * 3]
                     << setw(16) << Spin[i * 3 + 1]
                     << setw(16) << Spin[i * 3 + 2] << endl;
        }
    }

    int maxl = 1;
    int levels;
    int *grids;
    double bbox[6];
    // read parameter from file
    {
        const int LEN = 256;
        char pline[LEN];
        string str, sgrp, skey, sval;
        int sind1, sind2, sind3;
        ifstream inf(filename, ifstream::in);
        if (!inf.good() && myrank == 0)
        {
            cout << "bssn_class::Setup_Black_Hole_position: Can not open parameter file " << filename 
                     << " for inputing information of black holes" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 1; inf.good(); i++)
        {
            inf.getline(pline, LEN);
            str = pline;

            int status = misc::parse_parts(str, sgrp, skey, sval, sind1);
            if (status == -1)
            {
                cout << "error reading parameter file " << filename << " in line " << i << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            else if (status == 0)
                continue;

            if (sgrp == "cgh" && skey == "levels")
            {
                levels = atoi(sval.c_str());
                break;
            }
        }
        inf.close();
    }
    grids = new int[levels];
    // read parameter from file
    {
        const int LEN = 256;
        char pline[LEN];
        string str, sgrp, skey, sval;
        int sind1, sind2, sind3;
        ifstream inf(filename, ifstream::in);
        if (!inf.good() && myrank == 0)
        {
            cout << "bssn_class::Setup_Black_Hole_position: Can not open parameter file " << filename 
                     << " for inputing information of black holes" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 1; inf.good(); i++)
        {
            inf.getline(pline, LEN);
            str = pline;

            int status = misc::parse_parts(str, sgrp, skey, sval, sind1, sind2, sind3);
            if (status == -1)
            {
                cout << "error reading parameter file " << filename << " in line " << i << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            else if (status == 0)
                continue;

            if (sgrp == "cgh" && skey == "grids" && sind1 < levels)
                grids[sind1] = atoi(sval.c_str());
            if (sgrp == "cgh" && skey == "bbox" && sind1 == 0 && sind2 == 0)
                bbox[sind3] = atof(sval.c_str());
        }
        inf.close();
    }
    for (int i = 0; i < levels; i++)
        if (maxl < grids[i])
            maxl = grids[i];

    delete[] grids;

    if (BH_num > maxl)
    {
        int BH_numc = BH_num;
        for (int i = 0; i < BH_num; i++)
            if (Porg0[i][0] < bbox[0] || Porg0[i][0] > bbox[3] ||
                    Porg0[i][1] < bbox[1] || Porg0[i][1] > bbox[4] ||
                    Porg0[i][2] < bbox[2] || Porg0[i][2] > bbox[5])
            {
                delete[] Porg0[i];
                Porg0[i] = 0;
                BH_numc--;
            }

        if (BH_num > BH_numc)
        {
            maxl = BH_numc;
            int bhi;
            double *tmp;

            tmp = Pmom;
            Pmom = new double[3 * maxl];
            bhi = 0;
            for (int i = 0; i < BH_num; i++)
                if (Porg0[i])
                {
                    for (int j = 0; j < 3; j++)
                        Pmom[3 * bhi + j] = tmp[3 * i + j];
                    bhi++;
                }
            delete[] tmp;

            tmp = Spin;
            Spin = new double[3 * maxl];
            bhi = 0;
            for (int i = 0; i < BH_num; i++)
                if (Porg0[i])
                {
                    for (int j = 0; j < 3; j++)
                        Spin[3 * bhi + j] = tmp[3 * i + j];
                    bhi++;
                }
            delete[] tmp;

            tmp = Mass;
            Mass = new double[3 * maxl];
            bhi = 0;
            for (int i = 0; i < BH_num; i++)
                if (Porg0[i])
                {
                    Mass[bhi] = tmp[i];
                    bhi++;
                }
            delete[] tmp;

            double **ttmp;
            ttmp = Porg0;
            Porg0 = new double *[maxl];
            bhi = 0;
            for (int i = 0; i < BH_num; i++)
                if (ttmp[i])
                {
                    Porg0[bhi] = ttmp[i];
                    bhi++;
                }
            delete[] ttmp;

            for (int i = 0; i < BH_num; i++)
            {
                delete[] Porgbr[i];
                delete[] Porg[i];
                delete[] Porg1[i];
                delete[] Porg_rhs[i];
            }
            delete[] Porgbr;
            delete[] Porg;
            delete[] Porg1;
            delete[] Porg_rhs;

            BH_num = maxl;

            Porgbr = new double *[BH_num];
            Porg = new double *[BH_num];
            Porg1 = new double *[BH_num];
            Porg_rhs = new double *[BH_num];

            for (int i = 0; i < BH_num; i++)
            {
                Porgbr[i] = new double[3];
                Porg[i] = new double[3];
                Porg1[i] = new double[3];
                Porg_rhs[i] = new double[3];
            }
        }
    }

    for (int i = 0; i < BH_num; i++)
    {
        for (int j = 0; j < dim; j++)
            Porgbr[i][j] = Porg0[i][j];
    }
}

//================================================================================================



//================================================================================================

// This member function computes black hole positions

//================================================================================================

// new code considering diferent levels for different black hole
void bssn_class::compute_Porg_rhs(double **BH_PS, double **BH_RHS, var *forx, var *fory, var *forz, int ilev) {
    const int InList = 3;

    MyList<var> *DG_List = new MyList<var>(forx);
    DG_List->insert(fory);
    DG_List->insert(forz);

    double *d_pox[3];
    for (int i = 0; i < 3; i++) {
        d_pox[i] = GPUManager::getInstance().allocate_device_memory(1);
    }
    double *d_shellf = GPUManager::getInstance().allocate_device_memory(InList);
    double h_shellf[3] = {0.0, 0.0, 0.0};

    for (int n = 0; n < BH_num; n++) {
        double h_pox[3] = { BH_PS[n][0], BH_PS[n][1], BH_PS[n][2] };
        
        GPUManager::getInstance().sync_to_gpu(&h_pox[0], d_pox[0], 1);
        GPUManager::getInstance().sync_to_gpu(&h_pox[1], d_pox[1], 1);
        GPUManager::getInstance().sync_to_gpu(&h_pox[2], d_pox[2], 1);

        int lev = ilev;

        while (lev >= 0) {
            bool found = Parallel::PatList_Interp_Points_GPU(GPUManager::getInstance().get_stream(), GH->PatL[lev], DG_List, 1, d_pox, d_shellf, Symmetry);
            if (found) {
                break;
            }
            lev--;
        }

        if (lev < 0) {
            ErrorMonitor->outfile << "fail to find black holes at t = " << PhysTime << endl;
            ErrorMonitor->outfile << "(x,y,z) = (" << h_pox[0] << "," << h_pox[1] << "," << h_pox[2] << ")" << endl;
        }
        else {
            GPUManager::getInstance().sync_to_cpu(h_shellf, d_shellf, InList);
            BH_RHS[n][0] = -h_shellf[0];
            BH_RHS[n][1] = -h_shellf[1];
            BH_RHS[n][2] = -h_shellf[2];
        }
    }

    // 清理资源
    DG_List->clearList();
    for (int i = 0; i < 3; i++) {
        GPUManager::getInstance().free_device_memory(d_pox[i], 1);
    }
    GPUManager::getInstance().free_device_memory(d_shellf, InList);
}

//================================================================================================



//================================================================================================

// This member function computes gravitational-wave related data

//================================================================================================

void bssn_class::AnalysisStuff(int lev, double dT_lev) {
    LastAnas += dT_lev;

    if (LastAnas >= AnasTime) {
        Compute_Psi4(lev);
        double *RP, *IP, *RoutMAP;
        int NN = 0;
        for (int pl = 2; pl < maxl + 1; pl++)
            for (int pm = -pl; pm < pl + 1; pm++)
                NN++;
        RP = new double[NN];
        IP = new double[NN];
        RoutMAP = new double[7];
        double Rex = maxrex;
        for (int i = 0; i < decn; i++) {
            Waveshell->gpu_surf_Wave(Rex, lev, GH, Rpsi4, Ipsi4, 2, maxl, NN, RP, IP, ErrorMonitor);
            Waveshell->gpu_surf_MassPAng(
                Rex, lev, GH, phi0, trK0,
                gxx0, gxy0, gxz0, gyy0, gyz0, gzz0,
                Axx0, Axy0, Axz0, Ayy0, Ayz0, Azz0,
                Gmx0, Gmy0, Gmz0, Sfx1, Sfy1, Sfz1, // here we can not touch rhs variables, but 1 variables
                RoutMAP, ErrorMonitor
            );
            if (i == 0) {
                ADMMass = RoutMAP[0];
            }
            Psi4Monitor->writefile(PhysTime, NN, RP, IP);
            MAPMonitor->writefile(PhysTime, 7, RoutMAP);
            Rex = Rex - drex;
        }
        delete[] RP;
        delete[] IP;
        delete[] RoutMAP;

        // black hole's position
        {
            double *pox;
            pox = new double[dim * BH_num];
            for (int bhi = 0; bhi < BH_num; bhi++)
                for (int i = 0; i < dim; i++)
                    pox[dim * bhi + i] = Porg0[bhi][i];
            BHMonitor->writefile(PhysTime, dim * BH_num, pox);
            delete[] pox;
        }

        LastAnas = 0;
    }
}

//================================================================================================



//================================================================================================

// This member function computes and outputs constraint violations

//================================================================================================

void bssn_class::Constraint_Out()
{
    LastConsOut += dT * pow(0.5, Mymax(0, trfls));

    if (LastConsOut >= AnasTime)
    // Constraint violation
    {
        // recompute least the constraint data lost for moved new grid
        for (int lev = 0; lev < GH->levels; lev++)
        {
            // make sure the data consistent for higher levels
            if (lev > 0) // if the constrait quantities can be reused from the step rhs calculation
            {
                double TRK4 = PhysTime;
                double ndeps = numepsb;
                int pre = 0;
                MyList<Patch> *Pp = GH->PatL[lev];
                while (Pp)
                {
                    MyList<Block> *BP = Pp->data->blb;
                    while (BP)
                    {
                        Block *cg = BP->data;
                        if (myrank == cg->rank)
                        {
                            if (use_gpu == 1) {
                                gpu_compute_rhs_bssn_launch(
                                    cg->stream,
                                    cg->shape, TRK4, cg->d_X[0], cg->d_X[1], cg->d_X[2],
                                    cg->d_fgfs[phi0->sgfn], cg->d_fgfs[trK0->sgfn],
                                    cg->d_fgfs[gxx0->sgfn], cg->d_fgfs[gxy0->sgfn], cg->d_fgfs[gxz0->sgfn], 
                                    cg->d_fgfs[gyy0->sgfn], cg->d_fgfs[gyz0->sgfn], cg->d_fgfs[gzz0->sgfn],
                                    cg->d_fgfs[Axx0->sgfn], cg->d_fgfs[Axy0->sgfn], cg->d_fgfs[Axz0->sgfn], 
                                    cg->d_fgfs[Ayy0->sgfn], cg->d_fgfs[Ayz0->sgfn], cg->d_fgfs[Azz0->sgfn],
                                    cg->d_fgfs[Gmx0->sgfn], cg->d_fgfs[Gmy0->sgfn], cg->d_fgfs[Gmz0->sgfn],
                                    cg->d_fgfs[Lap0->sgfn], 
                                    cg->d_fgfs[Sfx0->sgfn], cg->d_fgfs[Sfy0->sgfn], cg->d_fgfs[Sfz0->sgfn],
                                    cg->d_fgfs[dtSfx0->sgfn], cg->d_fgfs[dtSfy0->sgfn], cg->d_fgfs[dtSfz0->sgfn],
                                    cg->d_fgfs[phi_rhs->sgfn], cg->d_fgfs[trK_rhs->sgfn],
                                    cg->d_fgfs[gxx_rhs->sgfn], cg->d_fgfs[gxy_rhs->sgfn], cg->d_fgfs[gxz_rhs->sgfn],
                                    cg->d_fgfs[gyy_rhs->sgfn], cg->d_fgfs[gyz_rhs->sgfn], cg->d_fgfs[gzz_rhs->sgfn],
                                    cg->d_fgfs[Axx_rhs->sgfn], cg->d_fgfs[Axy_rhs->sgfn], cg->d_fgfs[Axz_rhs->sgfn],
                                    cg->d_fgfs[Ayy_rhs->sgfn], cg->d_fgfs[Ayz_rhs->sgfn], cg->d_fgfs[Azz_rhs->sgfn],
                                    cg->d_fgfs[Gmx_rhs->sgfn], cg->d_fgfs[Gmy_rhs->sgfn], cg->d_fgfs[Gmz_rhs->sgfn],
                                    cg->d_fgfs[Lap_rhs->sgfn],
                                    cg->d_fgfs[Sfx_rhs->sgfn], cg->d_fgfs[Sfy_rhs->sgfn], cg->d_fgfs[Sfz_rhs->sgfn],
                                    cg->d_fgfs[dtSfx_rhs->sgfn], cg->d_fgfs[dtSfy_rhs->sgfn], cg->d_fgfs[dtSfz_rhs->sgfn],
                                    cg->d_fgfs[rho->sgfn], cg->d_fgfs[Sx->sgfn], cg->d_fgfs[Sy->sgfn], cg->d_fgfs[Sz->sgfn],
                                    cg->d_fgfs[Sxx->sgfn], cg->d_fgfs[Sxy->sgfn], cg->d_fgfs[Sxz->sgfn], 
                                    cg->d_fgfs[Syy->sgfn], cg->d_fgfs[Syz->sgfn], cg->d_fgfs[Szz->sgfn],
                                    cg->d_fgfs[Gamxxx->sgfn], cg->d_fgfs[Gamxxy->sgfn], cg->d_fgfs[Gamxxz->sgfn],
                                    cg->d_fgfs[Gamxyy->sgfn], cg->d_fgfs[Gamxyz->sgfn], cg->d_fgfs[Gamxzz->sgfn],
                                    cg->d_fgfs[Gamyxx->sgfn], cg->d_fgfs[Gamyxy->sgfn], cg->d_fgfs[Gamyxz->sgfn],
                                    cg->d_fgfs[Gamyyy->sgfn], cg->d_fgfs[Gamyyz->sgfn], cg->d_fgfs[Gamyzz->sgfn],
                                    cg->d_fgfs[Gamzxx->sgfn], cg->d_fgfs[Gamzxy->sgfn], cg->d_fgfs[Gamzxz->sgfn],
                                    cg->d_fgfs[Gamzyy->sgfn], cg->d_fgfs[Gamzyz->sgfn], cg->d_fgfs[Gamzzz->sgfn],
                                    cg->d_fgfs[Rxx->sgfn], cg->d_fgfs[Rxy->sgfn], cg->d_fgfs[Rxz->sgfn], 
                                    cg->d_fgfs[Ryy->sgfn], cg->d_fgfs[Ryz->sgfn], cg->d_fgfs[Rzz->sgfn],
                                    cg->d_fgfs[Cons_Ham->sgfn],
                                    cg->d_fgfs[Cons_Px->sgfn], cg->d_fgfs[Cons_Py->sgfn], cg->d_fgfs[Cons_Pz->sgfn],
                                    cg->d_fgfs[Cons_Gx->sgfn], cg->d_fgfs[Cons_Gy->sgfn], cg->d_fgfs[Cons_Gz->sgfn],
                                    Symmetry, lev, ndeps, pre
                                );
                            }
                            else
                                f_compute_rhs_bssn(RHS_PARA_CALLED_Constraint_Out);
                        }
                        if (BP == Pp->data->ble)
                            break;
                        BP = BP->next;
                    }
                    Pp = Pp->next;
                }
                GPUManager::getInstance().synchronize_all();
            }
            Parallel::Sync_GPU(GH->PatL[lev], ConstraintList, Symmetry);
        }

        double ConV[7];
        for (int levi = 0; levi < GH->levels; levi++)
        {
            ConV[0] = Parallel::L2Norm_GPU(GH->PatL[levi]->data, Cons_Ham);
            ConV[1] = Parallel::L2Norm_GPU(GH->PatL[levi]->data, Cons_Px);
            ConV[2] = Parallel::L2Norm_GPU(GH->PatL[levi]->data, Cons_Py);
            ConV[3] = Parallel::L2Norm_GPU(GH->PatL[levi]->data, Cons_Pz);
            ConV[4] = Parallel::L2Norm_GPU(GH->PatL[levi]->data, Cons_Gx);
            ConV[5] = Parallel::L2Norm_GPU(GH->PatL[levi]->data, Cons_Gy);
            ConV[6] = Parallel::L2Norm_GPU(GH->PatL[levi]->data, Cons_Gz);
            ConVMonitor->writefile(PhysTime, 7, ConV);
        }

        Interp_Constraint(false);

        LastConsOut = 0;
    }
}

//================================================================================================

//================================================================================================



//================================================================================================

// This member function interpolates constraint data

//================================================================================================
void bssn_class::Interp_Constraint(bool infg)
{
    if (infg)
    {
        // we do not support a_lev != 0 yet.
        if (a_lev > 0)
            return;

        // recompute least the constraint data lost for moved new grid
        for (int lev = 0; lev < GH->levels; lev++)
        {
            // make sure the data consistent for higher levels
            if (lev > 0) // if the constrait quantities can be reused from the step rhs calculation
            {
                double TRK4 = PhysTime;
                double ndeps = numepsb;
                int pre = 0;
                MyList<Patch> *Pp = GH->PatL[lev];
                while (Pp)
                {
                    MyList<Block> *BP = Pp->data->blb;
                    while (BP)
                    {
                        Block *cg = BP->data;
                        if (myrank == cg->rank)
                        {
                            if (use_gpu == 1) {
                                gpu_compute_rhs_bssn_launch(
                                    cg->stream,
                                    cg->shape, TRK4, cg->d_X[0], cg->d_X[1], cg->d_X[2],
                                    cg->d_fgfs[phi0->sgfn], cg->d_fgfs[trK0->sgfn],
                                    cg->d_fgfs[gxx0->sgfn], cg->d_fgfs[gxy0->sgfn], cg->d_fgfs[gxz0->sgfn], 
                                    cg->d_fgfs[gyy0->sgfn], cg->d_fgfs[gyz0->sgfn], cg->d_fgfs[gzz0->sgfn],
                                    cg->d_fgfs[Axx0->sgfn], cg->d_fgfs[Axy0->sgfn], cg->d_fgfs[Axz0->sgfn], 
                                    cg->d_fgfs[Ayy0->sgfn], cg->d_fgfs[Ayz0->sgfn], cg->d_fgfs[Azz0->sgfn],
                                    cg->d_fgfs[Gmx0->sgfn], cg->d_fgfs[Gmy0->sgfn], cg->d_fgfs[Gmz0->sgfn],
                                    cg->d_fgfs[Lap0->sgfn], 
                                    cg->d_fgfs[Sfx0->sgfn], cg->d_fgfs[Sfy0->sgfn], cg->d_fgfs[Sfz0->sgfn],
                                    cg->d_fgfs[dtSfx0->sgfn], cg->d_fgfs[dtSfy0->sgfn], cg->d_fgfs[dtSfz0->sgfn],
                                    cg->d_fgfs[phi_rhs->sgfn], cg->d_fgfs[trK_rhs->sgfn],
                                    cg->d_fgfs[gxx_rhs->sgfn], cg->d_fgfs[gxy_rhs->sgfn], cg->d_fgfs[gxz_rhs->sgfn],
                                    cg->d_fgfs[gyy_rhs->sgfn], cg->d_fgfs[gyz_rhs->sgfn], cg->d_fgfs[gzz_rhs->sgfn],
                                    cg->d_fgfs[Axx_rhs->sgfn], cg->d_fgfs[Axy_rhs->sgfn], cg->d_fgfs[Axz_rhs->sgfn],
                                    cg->d_fgfs[Ayy_rhs->sgfn], cg->d_fgfs[Ayz_rhs->sgfn], cg->d_fgfs[Azz_rhs->sgfn],
                                    cg->d_fgfs[Gmx_rhs->sgfn], cg->d_fgfs[Gmy_rhs->sgfn], cg->d_fgfs[Gmz_rhs->sgfn],
                                    cg->d_fgfs[Lap_rhs->sgfn],
                                    cg->d_fgfs[Sfx_rhs->sgfn], cg->d_fgfs[Sfy_rhs->sgfn], cg->d_fgfs[Sfz_rhs->sgfn],
                                    cg->d_fgfs[dtSfx_rhs->sgfn], cg->d_fgfs[dtSfy_rhs->sgfn], cg->d_fgfs[dtSfz_rhs->sgfn],
                                    cg->d_fgfs[rho->sgfn], cg->d_fgfs[Sx->sgfn], cg->d_fgfs[Sy->sgfn], cg->d_fgfs[Sz->sgfn],
                                    cg->d_fgfs[Sxx->sgfn], cg->d_fgfs[Sxy->sgfn], cg->d_fgfs[Sxz->sgfn], 
                                    cg->d_fgfs[Syy->sgfn], cg->d_fgfs[Syz->sgfn], cg->d_fgfs[Szz->sgfn],
                                    cg->d_fgfs[Gamxxx->sgfn], cg->d_fgfs[Gamxxy->sgfn], cg->d_fgfs[Gamxxz->sgfn],
                                    cg->d_fgfs[Gamxyy->sgfn], cg->d_fgfs[Gamxyz->sgfn], cg->d_fgfs[Gamxzz->sgfn],
                                    cg->d_fgfs[Gamyxx->sgfn], cg->d_fgfs[Gamyxy->sgfn], cg->d_fgfs[Gamyxz->sgfn],
                                    cg->d_fgfs[Gamyyy->sgfn], cg->d_fgfs[Gamyyz->sgfn], cg->d_fgfs[Gamyzz->sgfn],
                                    cg->d_fgfs[Gamzxx->sgfn], cg->d_fgfs[Gamzxy->sgfn], cg->d_fgfs[Gamzxz->sgfn],
                                    cg->d_fgfs[Gamzyy->sgfn], cg->d_fgfs[Gamzyz->sgfn], cg->d_fgfs[Gamzzz->sgfn],
                                    cg->d_fgfs[Rxx->sgfn], cg->d_fgfs[Rxy->sgfn], cg->d_fgfs[Rxz->sgfn], 
                                    cg->d_fgfs[Ryy->sgfn], cg->d_fgfs[Ryz->sgfn], cg->d_fgfs[Rzz->sgfn],
                                    cg->d_fgfs[Cons_Ham->sgfn],
                                    cg->d_fgfs[Cons_Px->sgfn], cg->d_fgfs[Cons_Py->sgfn], cg->d_fgfs[Cons_Pz->sgfn],
                                    cg->d_fgfs[Cons_Gx->sgfn], cg->d_fgfs[Cons_Gy->sgfn], cg->d_fgfs[Cons_Gz->sgfn],
                                    Symmetry, lev, ndeps, pre
                                );
                            }
                            else
                                f_compute_rhs_bssn(RHS_PARA_CALLED_Interp_Constraint);
                        }
                        if (BP == Pp->data->ble)
                            break;
                        BP = BP->next;
                    }
                    Pp = Pp->next;
                }
                GPUManager::getInstance().synchronize_all();
            }
            Parallel::Sync_GPU(GH->PatL[lev], ConstraintList, Symmetry);
        }
    }
    //    interpolate
    // 1. 准备插值点坐标
    int ordn = 2 * ghost_width;
    double *x1, *y1, *z1;
    const int n = 1000;
    double lmax, lmin, dd;
    lmin = 0;
    lmax = GH->bbox[0][0][4];
    dd = (lmax - lmin) / n;
    x1 = new double[n]; y1 = new double[n]; z1 = new double[n];
    for (int i = 0; i < n; i++)
    {
        x1[i] = 0;
        y1[i] = lmin + (i + 0.5) * dd;
        z1[i] = 0;
    }

    int InList = 0;
    MyList<var> *varl = ConstraintList;
    while (varl) { InList++; varl = varl->next; }

    double *d_x1, *d_y1, *d_z1, *d_shellf;
    cudaMalloc(&d_x1, n * sizeof(double));
    cudaMalloc(&d_y1, n * sizeof(double));
    cudaMalloc(&d_z1, n * sizeof(double));
    cudaMemcpy(d_x1, x1, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, y1, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z1, z1, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_shellf, n * InList * sizeof(double));
    cudaMemset(d_shellf, 0, n * InList * sizeof(double));

    // ==========================================
    // 2. 核心修正：CPU 上执行全局几何归属判定
    // 确保与 CPU 原版完全一致：细网格优先，先找到先认领，全 MPI 进程认领结果绝对一致
    // ==========================================
    void** assigned_bp = new void*[n];
    for(int i = 0; i < n; i++) assigned_bp[i] = nullptr;

    for (int i = 0; i < n; i++) {
        double XX[3] = {x1[i], y1[i], z1[i]};
        bool found = false;
        
        // 严格按照 CPU 从细到粗的逻辑进行优先级判定
        for (int lev = GH->levels - 1; lev >= 0 && !found; lev--) {
            MyList<Patch> *Pp = GH->PatL[lev];
            while (Pp && !found) {
                int pdim = dim; // 通常是3
                double DH[3];
                for(int d = 0; d < pdim; d++) DH[d] = Pp->data->getdX(d);
                
                // 判断是否在 Patch 的 Bounding Box 内
                bool in_patch = true;
                for(int d = 0; d < pdim; d++) {
                    if (XX[d] < Pp->data->bbox[d] + Pp->data->lli[d] * DH[d] - DH[d] / 100 || 
                        XX[d] > Pp->data->bbox[pdim + d] - Pp->data->uui[d] * DH[d] + DH[d] / 100) {
                        in_patch = false; break;
                    }
                }
                
                if (in_patch) {
                    MyList<Block> *Bp = Pp->data->blb;
                    while (Bp && !found) {
                        Block *BP = Bp->data;
                        bool in_block = true;
                        for (int d = 0; d < pdim; d++) {
                            double llb = (feq(BP->bbox[d], Pp->data->bbox[d], DH[d] / 2)) ? 
                                    BP->bbox[d] + Pp->data->lli[d] * DH[d] : BP->bbox[d] + ghost_width * DH[d];
                            double uub = (feq(BP->bbox[pdim + d], Pp->data->bbox[pdim + d], DH[d] / 2)) ? 
                                    BP->bbox[pdim + d] - Pp->data->uui[d] * DH[d] : BP->bbox[pdim + d] - ghost_width * DH[d];
                            
                            if (XX[d] - llb < -DH[d] / 2 || XX[d] - uub > DH[d] / 2) {
                                in_block = false; break;
                            }
                        }
                        
                        if (in_block) {
                            assigned_bp[i] = (void*)BP; // 全局锁定此点归属该 Block
                            found = true;
                        }
                        Bp = Bp->next;
                    }
                }
                Pp = Pp->next;
            }
        }
    }

    // 3. 遍历 Block 并针对活跃点派发 GPU 任务
    std::vector<int*> d_indices_to_free;

    for (int lev = GH->levels - 1; lev >= 0; lev--) {
        MyList<Patch> *Pp = GH->PatL[lev];
        while (Pp) {
            MyList<Block> *Bp = Pp->data->blb;
            while (Bp) {
                Block *BP = Bp->data;
                if (myrank == BP->rank) {
                    // 收集被分配给当前 Block 的所有插值点
                    std::vector<int> active_points;
                    for (int i = 0; i < n; i++) {
                        if (assigned_bp[i] == (void*)BP) {
                            active_points.push_back(i);
                        }
                    }

                    int active_count = active_points.size();
                    // 仅当此 Block 确实覆盖了目标点时，才触发 CUDA 拷贝和核函数
                    if (active_count > 0) {
                        int* d_active_indices;
                        cudaMalloc(&d_active_indices, active_count * sizeof(int));
                        cudaMemcpyAsync(d_active_indices, active_points.data(), active_count * sizeof(int), cudaMemcpyHostToDevice, BP->stream);
                        d_indices_to_free.push_back(d_active_indices);

                        varl = ConstraintList;
                        int k = 0;
                        while (varl) {
                            gpu_global_interp_amr_launch(
                                BP->stream, active_count, 3, d_active_indices,
                                d_x1, d_y1, d_z1,
                                BP->shape[0], BP->shape[1], BP->shape[2],
                                BP->d_X[0], BP->d_X[1], BP->d_X[2],
                                BP->d_fgfs[varl->data->sgfn], 
                                ordn, varl->data->SoA[0], varl->data->SoA[1], varl->data->SoA[2],
                                Symmetry, k, InList, d_shellf
                            );
                            k++;
                            varl = varl->next;
                        }
                    }
                }
                Bp = Bp->next;
            }
            Pp = Pp->next;
        }
    }

    // 4. 同步所有流，并清理临时的 GPU 索引数组
    GPUManager::getInstance().synchronize_all();
    for (int* ptr : d_indices_to_free) cudaFree(ptr);
    delete[] assigned_bp;

    // 5. 拷回 CPU 并做一次性全局归约
    double *shellf = new double[n * InList];
    cudaMemcpy(shellf, d_shellf, n * InList * sizeof(double), cudaMemcpyDeviceToHost);

    double *global_shellf = new double[n * InList];
    MPI_Allreduce(shellf, global_shellf, n * InList, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 6. Rank 0 写文件 (复用原有逻辑，使用 global_shellf)
    if (myrank == 0) {
        ofstream outfile;
        char filename[50];
        sprintf(filename, "%s/interp_constraint_%05d.dat", ErrorMonitor->out_dir.c_str(), int(PhysTime / dT + 0.5)); 
        // 0.5 for round off
        
        outfile.open(filename);
        outfile << "#  corrdinate, H_Res, Px_Res, Py_Res, Pz_Res, Gx_Res, Gy_Res, Gz_Res, ...." << endl;
        for (int i = 0; i < n; i++)
        {
            outfile << setw(10) << setprecision(10) << y1[i];
            for (int j = 0; j < InList; j++)
                outfile << " " << setw(16) << setprecision(15) << shellf[InList * i + j];
            outfile << endl;
        }
        outfile.close();
    }

    cudaFree(d_x1); cudaFree(d_y1); cudaFree(d_z1);
    cudaFree(d_shellf); 
    delete[] shellf; delete[] global_shellf;
    delete[] x1; delete[] y1; delete[] z1;
}

//================================================================================================



//================================================================================================

// This member function computes constraint violations

//================================================================================================

void bssn_class::Compute_Constraint()
{
    double TRK4 = PhysTime;
    double ndeps = numepsb;
    int pre = 0;
    int lev;

    for (lev = 0; lev < GH->levels; lev++)
    {
        {
            MyList<Patch> *Pp = GH->PatL[lev];
            while (Pp)
            {
                MyList<Block> *BP = Pp->data->blb;
                while (BP)
                {
                    Block *cg = BP->data;
                    if (myrank == cg->rank)
                    {
                        cg->move_to_gpu(StateList);
                        cg->move_to_gpu(MiscList);

                        gpu_compute_rhs_bssn_launch(
                            cg->stream,
                            cg->shape, TRK4, cg->d_X[0], cg->d_X[1], cg->d_X[2],
                            cg->d_fgfs[phi0->sgfn], cg->d_fgfs[trK0->sgfn],
                            cg->d_fgfs[gxx0->sgfn], cg->d_fgfs[gxy0->sgfn], cg->d_fgfs[gxz0->sgfn], 
                            cg->d_fgfs[gyy0->sgfn], cg->d_fgfs[gyz0->sgfn], cg->d_fgfs[gzz0->sgfn],
                            cg->d_fgfs[Axx0->sgfn], cg->d_fgfs[Axy0->sgfn], cg->d_fgfs[Axz0->sgfn], 
                            cg->d_fgfs[Ayy0->sgfn], cg->d_fgfs[Ayz0->sgfn], cg->d_fgfs[Azz0->sgfn],
                            cg->d_fgfs[Gmx0->sgfn], cg->d_fgfs[Gmy0->sgfn], cg->d_fgfs[Gmz0->sgfn],
                            cg->d_fgfs[Lap0->sgfn], 
                            cg->d_fgfs[Sfx0->sgfn], cg->d_fgfs[Sfy0->sgfn], cg->d_fgfs[Sfz0->sgfn],
                            cg->d_fgfs[dtSfx0->sgfn], cg->d_fgfs[dtSfy0->sgfn], cg->d_fgfs[dtSfz0->sgfn],
                            cg->d_fgfs[phi_rhs->sgfn], cg->d_fgfs[trK_rhs->sgfn],
                            cg->d_fgfs[gxx_rhs->sgfn], cg->d_fgfs[gxy_rhs->sgfn], cg->d_fgfs[gxz_rhs->sgfn],
                            cg->d_fgfs[gyy_rhs->sgfn], cg->d_fgfs[gyz_rhs->sgfn], cg->d_fgfs[gzz_rhs->sgfn],
                            cg->d_fgfs[Axx_rhs->sgfn], cg->d_fgfs[Axy_rhs->sgfn], cg->d_fgfs[Axz_rhs->sgfn],
                            cg->d_fgfs[Ayy_rhs->sgfn], cg->d_fgfs[Ayz_rhs->sgfn], cg->d_fgfs[Azz_rhs->sgfn],
                            cg->d_fgfs[Gmx_rhs->sgfn], cg->d_fgfs[Gmy_rhs->sgfn], cg->d_fgfs[Gmz_rhs->sgfn],
                            cg->d_fgfs[Lap_rhs->sgfn],
                            cg->d_fgfs[Sfx_rhs->sgfn], cg->d_fgfs[Sfy_rhs->sgfn], cg->d_fgfs[Sfz_rhs->sgfn],
                            cg->d_fgfs[dtSfx_rhs->sgfn], cg->d_fgfs[dtSfy_rhs->sgfn], cg->d_fgfs[dtSfz_rhs->sgfn],
                            cg->d_fgfs[rho->sgfn], cg->d_fgfs[Sx->sgfn], cg->d_fgfs[Sy->sgfn], cg->d_fgfs[Sz->sgfn],
                            cg->d_fgfs[Sxx->sgfn], cg->d_fgfs[Sxy->sgfn], cg->d_fgfs[Sxz->sgfn], 
                            cg->d_fgfs[Syy->sgfn], cg->d_fgfs[Syz->sgfn], cg->d_fgfs[Szz->sgfn],
                            cg->d_fgfs[Gamxxx->sgfn], cg->d_fgfs[Gamxxy->sgfn], cg->d_fgfs[Gamxxz->sgfn],
                            cg->d_fgfs[Gamxyy->sgfn], cg->d_fgfs[Gamxyz->sgfn], cg->d_fgfs[Gamxzz->sgfn],
                            cg->d_fgfs[Gamyxx->sgfn], cg->d_fgfs[Gamyxy->sgfn], cg->d_fgfs[Gamyxz->sgfn],
                            cg->d_fgfs[Gamyyy->sgfn], cg->d_fgfs[Gamyyz->sgfn], cg->d_fgfs[Gamyzz->sgfn],
                            cg->d_fgfs[Gamzxx->sgfn], cg->d_fgfs[Gamzxy->sgfn], cg->d_fgfs[Gamzxz->sgfn],
                            cg->d_fgfs[Gamzyy->sgfn], cg->d_fgfs[Gamzyz->sgfn], cg->d_fgfs[Gamzzz->sgfn],
                            cg->d_fgfs[Rxx->sgfn], cg->d_fgfs[Rxy->sgfn], cg->d_fgfs[Rxz->sgfn], 
                            cg->d_fgfs[Ryy->sgfn], cg->d_fgfs[Ryz->sgfn], cg->d_fgfs[Rzz->sgfn],
                            cg->d_fgfs[Cons_Ham->sgfn],
                            cg->d_fgfs[Cons_Px->sgfn], cg->d_fgfs[Cons_Py->sgfn], cg->d_fgfs[Cons_Pz->sgfn],
                            cg->d_fgfs[Cons_Gx->sgfn], cg->d_fgfs[Cons_Gy->sgfn], cg->d_fgfs[Cons_Gz->sgfn],
                            Symmetry, lev, ndeps, pre
                        );
                        GPUManager::getInstance().synchronize_all();
                        cg->move_to_cpu(RHSList);
                        cg->move_to_cpu(ConstraintList);
                    }
                    if (BP == Pp->data->ble)
                        break;
                    BP = BP->next;
                }
                Pp = Pp->next;
            }
        }
        Parallel::Sync(GH->PatL[lev], ConstraintList, Symmetry);
    }
    // prolong restrict constraint quantities
    for (lev = GH->levels - 1; lev > 0; lev--)
        RestrictProlong(lev, 1, false, ConstraintList, ConstraintList, ConstraintList);
}


void bssn_class::testRestrict()
{
    MyList<var> *DG_List = new MyList<var>(phi0);
    int lev = 0;
    double ZEO = 0, ONE = 1;
    MyList<Patch> *Pp = GH->PatL[lev];
    while (Pp)
    {
        MyList<Block> *BP = Pp->data->blb;
        while (BP)
        {
            Block *cg = BP->data;
            if (myrank == cg->rank)
            {
                f_set_value(cg->shape, cg->fgfs[phi0->sgfn], ZEO);
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }

    lev = 1;
    Pp = GH->PatL[lev];
    while (Pp)
    {
        MyList<Block> *BP = Pp->data->blb;
        while (BP)
        {
            Block *cg = BP->data;
            if (myrank == cg->rank)
            {
                f_set_value(cg->shape, cg->fgfs[phi0->sgfn], ONE);
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }

    Parallel::Restrict(GH->PatL[lev - 1], GH->PatL[lev], DG_List, DG_List, Symmetry);
    Parallel::Sync(GH->PatL[lev - 1], DG_List, Symmetry);

    Parallel::Dump_Data(GH->PatL[lev - 1], DG_List, 0, PhysTime, dT);
    Parallel::Dump_Data(GH->PatL[lev], DG_List, 0, PhysTime, dT);

    DG_List->clearList();
    exit(0);
}

//================================================================================================



//================================================================================================

void bssn_class::testOutBd()
{
    MyList<var> *DG_List = new MyList<var>(phi0);
    int lev = 1;
    double ZEO = 0, ONE = 1;
    MyList<Patch> *Pp = GH->PatL[lev];
    while (Pp)
    {
        MyList<Block> *BP = Pp->data->blb;
        while (BP)
        {
            Block *cg = BP->data;
            if (myrank == cg->rank)
            {
                f_set_value(cg->shape, cg->fgfs[phi0->sgfn], ZEO);
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }

    lev = 0;
    Pp = GH->PatL[lev];
    while (Pp)
    {
        MyList<Block> *BP = Pp->data->blb;
        while (BP)
        {
            Block *cg = BP->data;
            if (myrank == cg->rank)
            {
                f_set_value(cg->shape, cg->fgfs[phi0->sgfn], ONE);
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }

    lev = 1;
    MyList<Patch> *Ppc = GH->PatL[lev - 1];
    while (Ppc)
    {
        Pp = GH->PatL[lev];
        while (Pp)
        {
            Parallel::OutBdLow2Hi(Ppc->data, Pp->data, DG_List, DG_List, Symmetry);
            Pp = Pp->next;
        }
        Ppc = Ppc->next;
    }

    Parallel::Sync(GH->PatL[lev], DG_List, Symmetry);

    Parallel::Dump_Data(GH->PatL[lev], DG_List, 0, PhysTime, dT);
    Parallel::Dump_Data(GH->PatL[lev - 1], DG_List, 0, PhysTime, dT);

    DG_List->clearList();
    exit(0);
}

//================================================================================================



//================================================================================================

// This member function enforces/checks the trace-free condition

//================================================================================================

void bssn_class::Enforce_algcon(int lev, int fg)
{
    MyList<Patch> *Pp = GH->PatL[lev];
    while (Pp)
    {
        MyList<Block> *BP = Pp->data->blb;
        while (BP)
        {
            Block *cg = BP->data;
            if (myrank == cg->rank)
            {
                if (fg == 0)
                    f_enforce_ga(cg->shape,
                                             cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], 
                                             cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
                                             cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], 
                                             cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn]);
                else
                    f_enforce_ga(cg->shape,
                                             cg->fgfs[gxx->sgfn], cg->fgfs[gxy->sgfn], cg->fgfs[gxz->sgfn], 
                                             cg->fgfs[gyy->sgfn], cg->fgfs[gyz->sgfn], cg->fgfs[gzz->sgfn],
                                             cg->fgfs[Axx->sgfn], cg->fgfs[Axy->sgfn], cg->fgfs[Axz->sgfn], 
                                             cg->fgfs[Ayy->sgfn], cg->fgfs[Ayz->sgfn], cg->fgfs[Azz->sgfn]);
            }
            if (BP == Pp->data->ble)
                break;
            BP = BP->next;
        }
        Pp = Pp->next;
    }
}

// added by yangquan
void bssn_class::Get_runtime_envirment()
{
    // get processor name
    char pname[MPI_MAX_PROCESSOR_NAME];
    int resultlen = 0, pcode = 0;
    MPI_Get_processor_name(pname, &resultlen);
    cout << "MPI rank: " << myrank << "Processor name:" << pname << endl;
    for (int i = 0; i < resultlen; ++i)
    {
        pcode += ((int)(pname[i]) - 65) * i;
    }
}

//================================================================================================
