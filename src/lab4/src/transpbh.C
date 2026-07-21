// $Id: transpbh.C,v 1.2 2013/04/19 03:49:25 zjcao Exp $
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cmath>
using namespace std;

#include "macrodef.h"

// transmit black hole's position from bssn class

int BHN;
double Mass[3];
double PBH[9];

void setpbh(int iBHN, double **iPBH, double *iMass, int rBHN)
{
  BHN = Mymax(iBHN, rBHN);
  for (int i = 0; i < iBHN; i++)
  {
    for (int j = 0; j < 3; j++)
      PBH[3 * i + j] = iPBH[i][j];
    Mass[i] = iMass[i];
  }
  if (BHN < rBHN)
  {
    if (rBHN > 2)
      cout << "error in transpbh.C: something wrong." << endl;
    else
    {
      for (int j = 0; j < 3; j++)
        PBH[3 + j] = -iPBH[0][j];

      Mass[1] = Mass[0];
    }
  }
}
extern "C"
{

#ifdef fortran1
  void getpbh
#endif
#ifdef fortran2
      void GETPBH
#endif
#ifdef fortran3
      void
      getpbh_
#endif
      (int &oBHN, double *oPBH, double *oMass)
  {
    oBHN = BHN;
    for (int i = 0; i < BHN; i++)
      oMass[i] = Mass[i];
    for (int i = 0; i < 3 * BHN; i++)
      oPBH[i] = PBH[i];

    //  printf("have set BH_num = %d\n",oBHN);
  }
}
