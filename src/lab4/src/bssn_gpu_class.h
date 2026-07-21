
#ifndef BSSN_GPU_CLASS_H
#define BSSN_GPU_CLASS_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

#include <mpi.h>

#include "macrodef.h"
#include "cgh.h"
#include "misc.h"
#include "var.h"
#include "MyList.h"
#include "monitor.h"
#include "surface_integral.h"
#include "checkpoint.h"

// added by yangquan
#include "bssn_macro.h"

extern void setpbh(int iBHN, double **iPBH, double *iMass, int rBHN);

class bssn_class
{
public:
       // added by yangquan
       //----------------------
       int gpu_num_mynode;
       int cpu_core_num_mynode;
       int mpi_process_num_mynode;
       int my_sequence_mynode;
       int mynode_id;
       int use_gpu;

       virtual void Step_GPU(int lev, int YN);
       virtual void Get_runtime_envirment();
       // virtual void Step_OPENMP(int lev,int YN);
       //----------------------

       int ngfs;
       int nprocs, myrank;
       cgh *GH;
       double PhysTime;

       int checkrun;
       char checkfilename[50];
       int Steps;
       double StartTime, TotalTime;
       double AnasTime, DumpTime, d2DumpTime, CheckTime;
       double LastAnas, LastConsOut;
       double Courant;
       double numepss, numepsb, numepsh;
       int Symmetry;
       int maxl, decn;
       double maxrex, drex;
       int trfls, a_lev;

       double dT;
       double chitiny;

       double **Porg0, **Porgbr, **Porg, **Porg1, **Porg_rhs;
       int BH_num, BH_num_input;
       double *Mass, *Pmom, *Spin;
       double ADMMass;

       // OldStateList
       var *phio, *trKo;
       var *gxxo, *gxyo, *gxzo, *gyyo, *gyzo, *gzzo;
       var *Axxo, *Axyo, *Axzo, *Ayyo, *Ayzo, *Azzo;
       var *Gmxo, *Gmyo, *Gmzo;
       var *Lapo, *Sfxo, *Sfyo, *Sfzo;
       var *dtSfxo, *dtSfyo, *dtSfzo;

       // StateList
       var *phi0, *trK0;
       var *gxx0, *gxy0, *gxz0, *gyy0, *gyz0, *gzz0;
       var *Axx0, *Axy0, *Axz0, *Ayy0, *Ayz0, *Azz0;
       var *Gmx0, *Gmy0, *Gmz0;
       var *Lap0, *Sfx0, *Sfy0, *Sfz0;
       var *dtSfx0, *dtSfy0, *dtSfz0;

       // SynchList_pre
       var *phi, *trK;
       var *gxx, *gxy, *gxz, *gyy, *gyz, *gzz;
       var *Axx, *Axy, *Axz, *Ayy, *Ayz, *Azz;
       var *Gmx, *Gmy, *Gmz;
       var *Lap, *Sfx, *Sfy, *Sfz;
       var *dtSfx, *dtSfy, *dtSfz;

       // SynchList_cor
       var *phi1, *trK1;
       var *gxx1, *gxy1, *gxz1, *gyy1, *gyz1, *gzz1;
       var *Axx1, *Axy1, *Axz1, *Ayy1, *Ayz1, *Azz1;
       var *Gmx1, *Gmy1, *Gmz1;
       var *Lap1, *Sfx1, *Sfy1, *Sfz1;
       var *dtSfx1, *dtSfy1, *dtSfz1;

       // RHSList
       var *phi_rhs, *trK_rhs;
       var *gxx_rhs, *gxy_rhs, *gxz_rhs, *gyy_rhs, *gyz_rhs, *gzz_rhs;
       var *Axx_rhs, *Axy_rhs, *Axz_rhs, *Ayy_rhs, *Ayz_rhs, *Azz_rhs;
       var *Gmx_rhs, *Gmy_rhs, *Gmz_rhs;
       var *Lap_rhs, *Sfx_rhs, *Sfy_rhs, *Sfz_rhs;
       var *dtSfx_rhs, *dtSfy_rhs, *dtSfz_rhs;

       // MiscList
       var *rho, *Sx, *Sy, *Sz, *Sxx, *Sxy, *Sxz, *Syy, *Syz, *Szz;

       var *Gamxxx, *Gamxxy, *Gamxxz, *Gamxyy, *Gamxyz, *Gamxzz;
       var *Gamyxx, *Gamyxy, *Gamyxz, *Gamyyy, *Gamyyz, *Gamyzz;
       var *Gamzxx, *Gamzxy, *Gamzxz, *Gamzyy, *Gamzyz, *Gamzzz;

       var *Rxx, *Rxy, *Rxz, *Ryy, *Ryz, *Rzz;

       // DGList
       var *Rpsi4, *Ipsi4;
       var *t1Rpsi4, *t1Ipsi4, *t2Rpsi4, *t2Ipsi4;

       // ConstraintList
       var *Cons_Ham, *Cons_Px, *Cons_Py, *Cons_Pz, *Cons_Gx, *Cons_Gy, *Cons_Gz;

       // FIXME: uc = StateList, up = OldStateList, upp = SynchList_cor; so never touch these three data
       MyList<var> *StateList, *SynchList_pre, *SynchList_cor, *RHSList;
       MyList<var> *OldStateList, *DumpList;
       MyList<var> *ConstraintList;
       MyList<var> *MiscList, *DGList;

       monitor *ErrorMonitor, *Psi4Monitor, *BHMonitor, *MAPMonitor;
       monitor *ConVMonitor;
       surface_integral *Waveshell;
       checkpoint *CheckPoint;

public:
       bssn_class(double Couranti, double StartTimei, double TotalTimei, double DumpTimei, double d2DumpTimei, double CheckTimei, double AnasTimei,
                  int Symmetryi, int checkruni, char *checkfilenamei, double numepssi, double numepsbi, double numepshi,
                  int a_levi, int maxli, int decni, double maxrexi, double drexi);
       ~bssn_class();

       void Evolve(int Steps);
       void RecursiveStep(int lev);
       void RestrictProlong(int lev, int YN, bool BB, MyList<var> *SL, MyList<var> *OL, MyList<var> *corL);
       void RestrictProlong_aux(int lev, int YN, bool BB, MyList<var> *SL, MyList<var> *OL, MyList<var> *corL);
       void RestrictProlong(int lev, int YN, bool BB);
       void ProlongRestrict(int lev, int YN, bool BB);
       void Setup_Black_Hole_position();
       void compute_Porg_rhs(double **BH_PS, double **BH_RHS, var *forx, var *fory, var *forz, int lev);
       bool read_Pablo_file(int *ext, double *datain, char *filename);
       void write_Pablo_file(int *ext, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
                             char *filename);
       void AnalysisStuff(int lev, double dT_lev);
       void Setup_KerrSchild();
       void Enforce_algcon(int lev, int fg);

       void testRestrict();
       void testOutBd();

       virtual void Setup_Initial_Data_Lousto();
       virtual void Setup_Initial_Data_Cao();
       virtual void Initialize();
       virtual void Read_Ansorg();
       virtual void Read_Pablo() {};
       virtual void Compute_Psi4(int lev);
       virtual void Step(int lev, int YN);
       virtual void Interp_Constraint(bool infg);
       virtual void Constraint_Out();
       virtual void Compute_Constraint();

       void move_to_cpu();
       void move_to_gpu();
};
#endif /* BSSN_GPU_CLASS_H */
