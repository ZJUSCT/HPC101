
#ifndef MICRODEF_H
#define MICRODEF_H

#include "macrodef.fh"

// application parameters

#define SommerType 0

#define GaussInt

#define ABEtype 0

//#define With_AHF

#define Psi4type 0

//#define Point_Psi4

#define RPS 1

#define AGM 0

#define RPB 0

#define MAPBH 1

#define PSTR 0

#define REGLEV 0

#ifndef MPI_CUDA_AWARE
#define MPI_CUDA_AWARE 0
#endif

// USE_GPU is passed per-target by CMake (GPU target only), not defined here
//#define USE_GPU

//#define CHECKDETAIL

//#define FAKECHECK

////================================================================
//  some basic parameters for numerical calculation
////================================================================

#define dim 3

//#define Cell or Vertex in "macrodef.fh"

#define buffer_width 6

#define SC_width buffer_width

#define CS_width (2*buffer_width)

#if(buffer_width < ghost_width)
#   error we always assume buffer_width>ghost_width
#endif

#define PACK 1
#define UNPACK 2

#define Mymax(a,b) (((a) > (b)) ? (a) : (b))
#define Mymin(a,b) (((a) < (b)) ? (a) : (b))

#define feq(a,b,d) (fabs(a-b)<d)
#define flt(a,b,d) ((a-b)<d)
#define fgt(a,b,d) ((a-b)>d)

#define TINY 1e-10

#endif   /* MICRODEF_H */
