
#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <iostream>
#include <iomanip>
#include <strstream>
#include <fstream>
#include <string>
using namespace std;
#include <time.h>
#include <stdlib.h>

#include <mpi.h>

#include "var.h"
#include "MyList.h"
#include "cgh.h"
#include "macrodef.h"

class checkpoint
{

public:
  bool checkedrun;
  bool I_Print;
  char *filename;
  MyList<var> *CheckList;
  string out_dir;

public:
  checkpoint(bool checked, const char fname[], int myrank);
  // checkpoint(bool checked, char fname[50], int myrank);

  ~checkpoint();
  void addvariable(var *VV);
  void addvariablelist(MyList<var> *VL);

  void write_Black_Hole_position(int BH_num_input, int BH_num, double **Porg0, double **Porgbr, double *Mass);
  void read_Black_Hole_position(int &BH_num_input, int &BH_num, double **&Porg0, double *&Pmom,
                                double *&Spin, double *&Mass, double **&Porgbr, double **&Porg,
                                double **&Porg1, double **&Porg_rhs);
  void writecheck_cgh(double time, cgh *GH);
  void readcheck_cgh(double &time, cgh *GH, int myrank, int nprocs, int Symmetry);
  void write_bssn(double LastDump, double Last2dDump, double LastAnas);
  void read_bssn(double &LastDump, double &Last2dDump, double &LastAnas);
};

#endif /* CHECKPOINT */
