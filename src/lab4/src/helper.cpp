#include "helper.h"

#include "MyList.h"
#include "MPatch.h"
#include "Block.h"
#include "var.h"
#include "bssn_gpu_class.h"
namespace Helper {

void move_to_gpu_whole(MyList<Patch> *Pp, int myrank, MyList<var> *VarList) {
    while (Pp) {
        MyList<Block> *BP = Pp->data->blb;
        while (BP) {
            Block *cg = BP->data;
            if (myrank == cg->rank) {
				cg->move_to_gpu(VarList);
            }
            BP = BP->next;
        }
        Pp = Pp->next;
    }
}

void move_to_gpu_whole(cgh *GH, int myrank, MyList<var> *VarList) {
    for (int lev = 0; lev < GH->levels; lev ++) {
        Helper::move_to_gpu_whole(GH->PatL[lev], myrank, VarList);   
    }
}

void move_to_cpu_whole(MyList<Patch> *Pp, int myrank, MyList<var> *VarList) {
    while (Pp) {
        MyList<Block> *BP = Pp->data->blb;
        while (BP) {
            Block *cg = BP->data;
            if (myrank == cg->rank) {
				cg->move_to_cpu(VarList);
            }
            BP = BP->next;
        }
        Pp = Pp->next;
    }
}

void move_to_cpu_whole(cgh *GH, int myrank, MyList<var> *VarList) {
    for (int lev = 0; lev < GH->levels; lev ++) {
        Helper::move_to_cpu_whole(GH->PatL[lev], myrank, VarList);   
    }
}

}