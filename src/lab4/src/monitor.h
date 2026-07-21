
#ifndef MONITOR_H
#define MONITOR_H

#include <iostream>
#include <iomanip>
#include <strstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
using namespace std;

#include <time.h>

#include <mpi.h>

class monitor
{

public:
  string out_dir;
  ofstream outfile;

  bool I_Print;

public:
  monitor(const char fname[], int myrank, string head);
  monitor(const char fname[], int myrank, const int out_rank, string head);

  ~monitor();

  void writefile(double time, int NN, double *DDAT);
  void writefile(double time, int NN, double *DDAT1, double *DDAT2);
  void print_message(string head);
};

#endif /* MONITOR */
