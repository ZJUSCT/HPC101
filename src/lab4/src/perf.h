
#ifndef PERF_H
#define PERF_H
#ifdef newc
#include <iostream>
#include <iomanip>
#include <fstream>
#include <strstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
using namespace std;
#else
#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#endif

/* for open/read/close */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <mpi.h>

/* Real time */
#define TimerSignal SIGALRM
#define TimerType ITIMER_REAL

class perf
{
public:
   static size_t mem_peak;
   static size_t mem_current;
   /* The sampling interval of the timer in ms, <= 0 disables the timer. */
   static int sampling_interval;
   static char statm[40];
   static bool have_statm;
   static struct itimerval new_it, old;
   static struct sigaction sa, old_sa;

public:
   perf();
   ~perf();
   static void sample_mem_usage(int dummy);
   size_t MemoryUsage(size_t *current_min, size_t *current_avg, size_t *current_max,
                      size_t *peak_min, size_t *peak_avg, size_t *peak_max,
                      int nprocs);
};
#endif /* PERF_H */
