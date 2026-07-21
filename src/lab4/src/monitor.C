
#include <cstdio>
using namespace std;

#include "unistd.h"

#include "monitor.h"
#include "parameters.h"
#include "misc.h"

monitor::monitor(const char fname[], int myrank, string head)
{
  I_Print = (myrank == 0);

  if (I_Print)
  {
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
    // considering checkpoint run
    char filename[50];
    sprintf(filename, "%s/%s", out_dir.c_str(), fname);
    int i = 1;
    while ((access(filename, F_OK)) != -1)
    {
      sprintf(filename, "%s/%d_%s", out_dir.c_str(), i, fname);
      i++;
    }

    outfile.open(filename, ios::trunc);

    time_t tnow;
    time(&tnow);
    struct tm *loc_time;
    loc_time = localtime(&tnow);

    outfile << "# File created on " << asctime(loc_time);
    outfile << "#" << endl;
    outfile.setf(ios::left);
    outfile << head << endl;
  }
}

monitor::monitor(const char fname[], int myrank, const int out_rank, string head)
{
  I_Print = (myrank == out_rank);

  if (I_Print)
  {
    // considering checkpoint run
    char filename[50];
    sprintf(filename, "%s/%s", out_dir.c_str(), fname);
    int i = 1;
    while ((access(filename, F_OK)) != -1)
    {
      sprintf(filename, "%s/%d_%s", out_dir.c_str(), i, fname);
      i++;
    }

    outfile.open(filename, ios::trunc);

    time_t tnow;
    time(&tnow);
    struct tm *loc_time;
    loc_time = localtime(&tnow);

    outfile << "# File created on " << asctime(loc_time);
    outfile << "#" << endl;
    outfile.setf(ios::left);
    outfile << head << endl;
  }
}
monitor::~monitor()
{
  if (I_Print)
    outfile.close();
}
void monitor::writefile(double time, int NN, double *DDAT)
{
  if (I_Print)
  {
    outfile << setprecision(8);
    outfile << setw(14) << time;
    for (int countlm = 0; countlm < NN; countlm++)
    {
      outfile << " " << setw(15) << DDAT[countlm];
    }
    outfile << endl;
    flush(outfile);
  }
}
void monitor::writefile(double time, int NN, double *DDAT1, double *DDAT2)
{
  if (I_Print)
  {
    outfile << setprecision(8);
    outfile << setw(14) << time;
    for (int countlm = 0; countlm < NN; countlm++)
    {
      outfile << " " << setw(15) << DDAT1[countlm]
              << " " << setw(15) << DDAT2[countlm];
    }
    outfile << endl;
    flush(outfile);
  }
}
void monitor::print_message(string head)
{
  if (I_Print)
  {
    outfile << head << endl;
    flush(outfile);
  }
}
