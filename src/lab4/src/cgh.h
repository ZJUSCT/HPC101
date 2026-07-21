
#ifndef CGH_H
#define CGH_H

#include <mpi.h>
#include "MyList.h"
#include "MPatch.h"
#include "macrodef.h"
#include "monitor.h"
#include "Parallel.h"

class cgh
{

public:
   int levels, movls, BH_num_in;
   // information of boxes
   int *grids;
   double ***bbox;
   int ***shape;
   double ***handle;
   double ***Porgls;
   double *Lt;

   // information of Patch list
   MyList<Patch> **PatL;

protected:
   int ingfs, fngfs;
   static constexpr double ratio = 0.75;
   int trfls;

public:
   cgh(int ingfsi, int fngfsi, int Symmetry, char *filename, int checkrun, monitor *ErrorMonitor);

   ~cgh();

   void compose_cgh(int nprocs);
   void sethandle(monitor *ErrorMonitor);
   void checkPatchList(MyList<Patch> *PatL, bool buflog);
   void Regrid(int Symmetry, int BH_num, double **Porgbr, double **Porg0,
               MyList<var> *OldList, MyList<var> *StateList,
               MyList<var> *FutureList, MyList<var> *tmList, bool BB,
               monitor *ErrorMonitor);
   void Regrid_fake(int Symmetry, int BH_num, double **Porgbr, double **Porg0,
                    MyList<var> *OldList, MyList<var> *StateList,
                    MyList<var> *FutureList, MyList<var> *tmList, bool BB,
                    monitor *ErrorMonitor);
   void recompose_cgh(int nprocs, bool *lev_flag,
                      MyList<var> *OldList, MyList<var> *StateList,
                      MyList<var> *FutureList, MyList<var> *tmList,
                      int Symmetry, bool BB);
   void recompose_cgh_fake(int nprocs, bool *lev_flag,
                           MyList<var> *OldList, MyList<var> *StateList,
                           MyList<var> *FutureList, MyList<var> *tmList,
                           int Symmetry, bool BB);
   void read_bbox(int Symmetry, char *filename);
   MyList<Patch> *construct_patchlist(int lev, int Symmetry);
   bool Interp_One_Point(MyList<var> *VarList,
                         double *XX, /*input global Cartesian coordinate*/
                         double *Shellf, int Symmetry);
   void recompose_cgh_Onelevel(int nprocs, int lev,
                               MyList<var> *OldList, MyList<var> *StateList,
                               MyList<var> *FutureList, MyList<var> *tmList,
                               int Symmetry, bool BB);
   void Regrid_Onelevel(int lev, int Symmetry, int BH_num, double **Porgbr, double **Porg0,
                        MyList<var> *OldList, MyList<var> *StateList,
                        MyList<var> *FutureList, MyList<var> *tmList, bool BB,
                        monitor *ErrorMonitor);
   void Regrid_Onelevel_aux(int lev, int Symmetry, int BH_num, double **Porgbr, double **Porg0,
                            MyList<var> *OldList, MyList<var> *StateList,
                            MyList<var> *FutureList, MyList<var> *tmList, bool BB,
                            monitor *ErrorMonitor);
   void settrfls(const int lev);
#ifdef USE_GPU
   bool Interp_N_Points_GPU(
      MyList<var> *VarList,
      int NN, double *d_XX_0, double *d_XX_1, double *d_XX_2,
      double *d_shellf, int *d_weight, int Symmetry
   );
#endif
};

#endif /* CGH_H */
