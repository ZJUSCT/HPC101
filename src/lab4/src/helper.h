#ifndef HELPER_H
#define HELPER_H

#include "MyList.h"
#include "MPatch.h"
#include "Block.h"
#include "var.h"
#include "bssn_gpu_class.h"

namespace Helper {
    void move_to_gpu_whole(MyList<Patch> *Pp, int myrank, MyList<var> *VarList);
    void move_to_gpu_whole(cgh *GH, int myrank, MyList<var> *VarList);
    void move_to_cpu_whole(MyList<Patch> *Pp, int myrank, MyList<var> *VarList);
    void move_to_cpu_whole(cgh *GH, int myrank, MyList<var> *VarList);
}

#endif