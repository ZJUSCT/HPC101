
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <algorithm>   
#include <functional> 
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
using namespace std;

#include <mpi.h>

namespace parameters
{
    extern map<string,int> int_par;
    extern map<string,double> dou_par;
    extern map<string,string> str_par;
}
#endif   /* PARAMETERS_H */
