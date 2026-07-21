
#ifdef newc
#include <algorithm>
#include <functional>
#include <vector>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cmath>
#include <strstream>
using namespace std;
#else
#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#endif

#include "TwoPunctures.h"

inline string &lTrim(string &ss)
{
  string::iterator p = find_if(ss.begin(), ss.end(), not1(ptr_fun<int, int>(isspace)));
  ss.erase(ss.begin(), p);
  return ss;
}
inline string &rTrim(string &ss)
{
  string::reverse_iterator p = find_if(ss.rbegin(), ss.rend(), not1(ptr_fun<int, int>(isspace)));
  ss.erase(p.base(), ss.end());
  return ss;
}
inline string &Trim(string &st)
{
  lTrim(rTrim(st));
  return st;
}

int parse_parts(string str, string &sgrp, string &skey, string &sval, int &ind)
{
  int pos1, pos2;
  string s0;

  ind = 0;

  // remove comments
  str = str.substr(0, str.find("#"));
  if (rTrim(str).empty())
    return 0; // continue;

  // parse {group, key, val}
  pos1 = str.find("::");
  pos2 = str.find("=");
  if (pos1 == string::npos || pos2 == string::npos)
    return -1;

  s0 = str.substr(0, pos1);
  sgrp = lTrim(s0);
  s0 = str.substr(pos1 + 2, pos2 - pos1 - 2);
  skey = rTrim(s0);
  s0 = str.substr(pos2 + 1);
  sval = Trim(s0);

  pos1 = sval.find("\"");
  pos2 = sval.rfind("\"");
  if (pos1 != string::npos)
  {
    sval = sval.substr(1, pos2 - 1);
  }

  pos1 = skey.find("[");
  pos2 = skey.find("]");
  if (pos1 != string::npos)
  {
    s0 = skey.substr(0, pos1);
    ind = atoi(skey.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
    skey = s0;
  }

  return 1;
}
//=======================================
int main(int argc, char *argv[])
{
  double mp, mm, b, Mp, Mm, admtol, Newtontol;
  int nA, nB, nphi, Newtonmaxit;
  double P_plusx, P_plusy, P_plusz;
  double P_minusx, P_minusy, P_minusz;
  double S_plusx, S_plusy, S_plusz;
  double S_minusx, S_minusy, S_minusz;
  // read parameter from file
  {
    const int LEN = 256;
    char pline[LEN];
    string str, sgrp, skey, sval;
    int sind;
    const char pname[] = "TwoPunctureinput.par";
    ifstream inf(pname, ifstream::in);
    if (!inf.good())
    {
      cout << "Can not open parameter file " << pname << endl;
      exit(0);
    }

    for (int i = 1; inf.good(); i++)
    {
      inf.getline(pline, LEN);
      str = pline;

      int status = parse_parts(str, sgrp, skey, sval, sind);
      if (status == -1)
      {
        cout << "error reading parameter file " << pname << " in line " << i << endl;
        exit(0);
      }
      else if (status == 0)
        continue;
      // we assume input in Brugmann's convention
      if (sgrp == "ABE")
      {
        if (skey == "mm")
          mm = atof(sval.c_str());
        else if (skey == "mp")
          mp = atof(sval.c_str());
        else if (skey == "b")
          b = atof(sval.c_str());
        else if (skey == "P_plusx")
          P_plusy = -atof(sval.c_str());
        else if (skey == "P_plusy")
          P_plusx = atof(sval.c_str());
        else if (skey == "P_plusz")
          P_plusz = atof(sval.c_str());
        else if (skey == "P_minusx")
          P_minusy = -atof(sval.c_str());
        else if (skey == "P_minusy")
          P_minusx = atof(sval.c_str());
        else if (skey == "P_minusz")
          P_minusz = atof(sval.c_str());
        else if (skey == "S_plusx")
          S_plusy = -atof(sval.c_str());
        else if (skey == "S_plusy")
          S_plusx = atof(sval.c_str());
        else if (skey == "S_plusz")
          S_plusz = atof(sval.c_str());
        else if (skey == "S_minusx")
          S_minusy = -atof(sval.c_str());
        else if (skey == "S_minusy")
          S_minusx = atof(sval.c_str());
        else if (skey == "S_minusz")
          S_minusz = atof(sval.c_str());
        else if (skey == "Mp")
          Mp = atof(sval.c_str());
        else if (skey == "Mm")
          Mm = atof(sval.c_str());
        else if (skey == "admtol")
          admtol = atof(sval.c_str());
        else if (skey == "Newtontol")
          Newtontol = atof(sval.c_str());
        else if (skey == "nA")
          nA = atoi(sval.c_str());
        else if (skey == "nB")
          nB = atoi(sval.c_str());
        else if (skey == "nphi")
          nphi = atoi(sval.c_str());
        else if (skey == "Newtonmaxit")
          Newtonmaxit = atoi(sval.c_str());
      }
    }
    inf.close();
  }
  // echo parameters
  {
    cout << "///////////////////////////////////////////////////////////////" << endl;
    cout << "     mp     = " << mp << endl;
    cout << "     mm     = " << mm << endl;
    cout << "     b      = " << b << endl;
    cout << "  P_plusx   = " << P_plusx << endl;
    cout << "  P_plusy   = " << P_plusy << endl;
    cout << "  P_plusz   = " << P_plusz << endl;
    cout << "  P_minusx  = " << P_minusx << endl;
    cout << "  P_minusy  = " << P_minusy << endl;
    cout << "  P_minusz  = " << P_minusz << endl;
    cout << "  S_plusx   = " << S_plusx << endl;
    cout << "  S_plusy   = " << S_plusy << endl;
    cout << "  S_plusz   = " << S_plusz << endl;
    cout << "  S_minusx  = " << S_minusx << endl;
    cout << "  S_minusy  = " << S_minusy << endl;
    cout << "  S_minusz  = " << S_minusz << endl;
    cout << "     Mp     = " << Mp << endl;
    cout << "     Mm     = " << Mm << endl;
    cout << "   admtol   = " << admtol << endl;
    cout << " Newtontol  = " << Newtontol << endl;
    cout << "    nA      = " << nA << endl;
    cout << "    nB      = " << nB << endl;
    cout << "   nphi     = " << nphi << endl;
    cout << "Newtonmaxit = " << Newtonmaxit << endl;
    cout << "///////////////////////////////////////////////////////////////" << endl;
  }
  //===========================the computation body====================================================
  TwoPunctures *ADM;

  ADM = new TwoPunctures(mp, mm, b, P_plusx, P_plusy, P_plusz, S_plusx, S_plusy, S_plusz,
                         P_minusx, P_minusy, P_minusz, S_minusx, S_minusy, S_minusz,
                         nA, nB, nphi, Mp, Mm, admtol, Newtontol, Newtonmaxit);
  ADM->Solve();
  ADM->Save("Ansorg.psid");

  delete ADM;
  //=======================caculation done=============================================================
  cout << "===============================================================" << endl;
  cout << "Initial data is successfully producede!!" << endl;

  exit(0);
}
