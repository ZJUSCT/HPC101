#ifndef BSSN_STEP_H
#define BSSN_STEP_H
//1---------------------FLAGS---------------------

#define MAX_GPU_PROCESS_NUM 1
#define COUNT_CPU_RHS_TIME


//2---------------------TIMER---------------------
//2.1 TIMER_INIT
//2.2 TIMER_TIC_WITHOUT_OUTPUT
//2.3 TIMER_TIC(tag,order,label)
//2.4 TIMER_TIC_TAIL_OF_FUNC(tag,label)

#define TIME_COUNT_EACH_RANK 0

#define TIMER_INIT \
double clock_prev,clock_curr,step_begin_clock;\
if(1 == 1){\
	clock_curr =MPI_Wtime();\
	step_begin_clock = MPI_Wtime();\
}else{\
if(myrank == 0){\
	clock_curr= MPI_Wtime();\
	step_begin_clock = MPI_Wtime();\
}\
}
      
#define TIMER_TIC(tag,order,label) \
if(TIME_COUNT_EACH_RANK == 1){\
	  clock_prev= clock_curr;\
      clock_curr = MPI_Wtime();\
      cout<<#tag <<order <<":MPI Rank: "<<myrank<<" "<<#label <<" "<<(clock_curr-clock_prev)<<endl;\
}else{\
  if(myrank==0){\
      clock_prev= clock_curr;\
      clock_curr = MPI_Wtime();\
      cout<<#tag <<order <<" "<<#label " "<<(clock_curr-clock_prev)<<endl;\
  }\
}

#define TIMER_TIC_EACH_PROC(tag,order,label) \
	  clock_prev= clock_curr;\
      clock_curr = MPI_Wtime();\
      cout<<#tag <<order <<":MPI Rank: "<<myrank<<" "<<#label <<" "<<(clock_curr-clock_prev)<<endl;\
}

#define TIMER_TIC_WITHOUT_OUTPUT \
if(TIME_COUNT_EACH_RANK == 1){\
      clock_curr = MPI_Wtime();\
}else{\
  if(myrank==0){\
      clock_curr = MPI_Wtime();\
  }\
}

#define TIMER_TIC_TAIL_OF_FUNC(tag,label) \
if(TIME_COUNT_EACH_RANK == 1){\
	 cout<<#tag <<"MPI Rank: "<<myrank<<" "<<#label <<" "<<(MPI_Wtime()-step_begin_clock)<<" seconds!"<<endl;\
}else{\
     if(myrank==0)\
     {\
      cout<<#tag <<#label <<" "<<(MPI_Wtime()-step_begin_clock)<<" seconds!"<<endl;\
     }\
}

//3---------------------GPU---------------------
#define CALLED_BY_STEP 0
#define CALLED_BY_CONSTRAINT 1

#define RHS_PARA_CALLED_FIRST_TIME cg->shape,TRK4,cg->X[0],cg->X[1],cg->X[2],cg->fgfs[phi0->sgfn],cg->fgfs[trK0->sgfn],cg->fgfs[gxx0->sgfn],cg->fgfs[gxy0->sgfn],cg->fgfs[gxz0->sgfn],cg->fgfs[gyy0->sgfn],cg->fgfs[gyz0->sgfn],cg->fgfs[gzz0->sgfn],cg->fgfs[Axx0->sgfn],cg->fgfs[Axy0->sgfn],cg->fgfs[Axz0->sgfn],cg->fgfs[Ayy0->sgfn],cg->fgfs[Ayz0->sgfn],cg->fgfs[Azz0->sgfn],cg->fgfs[Gmx0->sgfn],cg->fgfs[Gmy0->sgfn],cg->fgfs[Gmz0->sgfn],cg->fgfs[Lap0->sgfn],cg->fgfs[Sfx0->sgfn],cg->fgfs[Sfy0->sgfn],cg->fgfs[Sfz0->sgfn],cg->fgfs[dtSfx0->sgfn],cg->fgfs[dtSfy0->sgfn],cg->fgfs[dtSfz0->sgfn],cg->fgfs[phi_rhs->sgfn],cg->fgfs[trK_rhs->sgfn],cg->fgfs[gxx_rhs->sgfn],cg->fgfs[gxy_rhs->sgfn],cg->fgfs[gxz_rhs->sgfn],cg->fgfs[gyy_rhs->sgfn],cg->fgfs[gyz_rhs->sgfn],cg->fgfs[gzz_rhs->sgfn],cg->fgfs[Axx_rhs->sgfn],cg->fgfs[Axy_rhs->sgfn],cg->fgfs[Axz_rhs->sgfn],cg->fgfs[Ayy_rhs->sgfn],cg->fgfs[Ayz_rhs->sgfn],cg->fgfs[Azz_rhs->sgfn],cg->fgfs[Gmx_rhs->sgfn],cg->fgfs[Gmy_rhs->sgfn],cg->fgfs[Gmz_rhs->sgfn],cg->fgfs[Lap_rhs->sgfn],cg->fgfs[Sfx_rhs->sgfn],cg->fgfs[Sfy_rhs->sgfn],cg->fgfs[Sfz_rhs->sgfn],cg->fgfs[dtSfx_rhs->sgfn],cg->fgfs[dtSfy_rhs->sgfn],cg->fgfs[dtSfz_rhs->sgfn],cg->fgfs[rho->sgfn],cg->fgfs[Sx->sgfn],cg->fgfs[Sy->sgfn],cg->fgfs[Sz->sgfn],cg->fgfs[Sxx->sgfn],cg->fgfs[Sxy->sgfn],cg->fgfs[Sxz->sgfn],cg->fgfs[Syy->sgfn],cg->fgfs[Syz->sgfn],cg->fgfs[Szz->sgfn],cg->fgfs[Gamxxx->sgfn],cg->fgfs[Gamxxy->sgfn],cg->fgfs[Gamxxz->sgfn],cg->fgfs[Gamxyy->sgfn],cg->fgfs[Gamxyz->sgfn],cg->fgfs[Gamxzz->sgfn],cg->fgfs[Gamyxx->sgfn],cg->fgfs[Gamyxy->sgfn],cg->fgfs[Gamyxz->sgfn],cg->fgfs[Gamyyy->sgfn],cg->fgfs[Gamyyz->sgfn],cg->fgfs[Gamyzz->sgfn],cg->fgfs[Gamzxx->sgfn],cg->fgfs[Gamzxy->sgfn],cg->fgfs[Gamzxz->sgfn],cg->fgfs[Gamzyy->sgfn],cg->fgfs[Gamzyz->sgfn],cg->fgfs[Gamzzz->sgfn],cg->fgfs[Rxx->sgfn],cg->fgfs[Rxy->sgfn],cg->fgfs[Rxz->sgfn],cg->fgfs[Ryy->sgfn],cg->fgfs[Ryz->sgfn],cg->fgfs[Rzz->sgfn],cg->fgfs[Cons_Ham->sgfn],cg->fgfs[Cons_Px->sgfn],cg->fgfs[Cons_Py->sgfn],cg->fgfs[Cons_Pz->sgfn],cg->fgfs[Cons_Gx->sgfn],cg->fgfs[Cons_Gy->sgfn],cg->fgfs[Cons_Gz->sgfn],Symmetry,lev,ndeps,pre

#define RHS_PARA_CALLED_THEN cg->shape,TRK4,cg->X[0],cg->X[1],cg->X[2],cg->fgfs[phi->sgfn],cg->fgfs[trK->sgfn],cg->fgfs[gxx->sgfn],cg->fgfs[gxy->sgfn],cg->fgfs[gxz->sgfn],cg->fgfs[gyy->sgfn],cg->fgfs[gyz->sgfn],cg->fgfs[gzz->sgfn],cg->fgfs[Axx->sgfn],cg->fgfs[Axy->sgfn],cg->fgfs[Axz->sgfn],cg->fgfs[Ayy->sgfn],cg->fgfs[Ayz->sgfn],cg->fgfs[Azz->sgfn],cg->fgfs[Gmx->sgfn],cg->fgfs[Gmy->sgfn],cg->fgfs[Gmz->sgfn],cg->fgfs[Lap->sgfn],cg->fgfs[Sfx->sgfn],cg->fgfs[Sfy->sgfn],cg->fgfs[Sfz->sgfn],cg->fgfs[dtSfx->sgfn],cg->fgfs[dtSfy->sgfn],cg->fgfs[dtSfz->sgfn],cg->fgfs[phi1->sgfn],cg->fgfs[trK1->sgfn],cg->fgfs[gxx1->sgfn],cg->fgfs[gxy1->sgfn],cg->fgfs[gxz1->sgfn],cg->fgfs[gyy1->sgfn],cg->fgfs[gyz1->sgfn],cg->fgfs[gzz1->sgfn],cg->fgfs[Axx1->sgfn],cg->fgfs[Axy1->sgfn],cg->fgfs[Axz1->sgfn],cg->fgfs[Ayy1->sgfn],cg->fgfs[Ayz1->sgfn],cg->fgfs[Azz1->sgfn],cg->fgfs[Gmx1->sgfn],cg->fgfs[Gmy1->sgfn],cg->fgfs[Gmz1->sgfn],cg->fgfs[Lap1->sgfn],cg->fgfs[Sfx1->sgfn],cg->fgfs[Sfy1->sgfn],cg->fgfs[Sfz1->sgfn],cg->fgfs[dtSfx1->sgfn],cg->fgfs[dtSfy1->sgfn],cg->fgfs[dtSfz1->sgfn],cg->fgfs[rho->sgfn],cg->fgfs[Sx->sgfn],cg->fgfs[Sy->sgfn],cg->fgfs[Sz->sgfn],cg->fgfs[Sxx->sgfn],cg->fgfs[Sxy->sgfn],cg->fgfs[Sxz->sgfn],cg->fgfs[Syy->sgfn],cg->fgfs[Syz->sgfn],cg->fgfs[Szz->sgfn],cg->fgfs[Gamxxx->sgfn],cg->fgfs[Gamxxy->sgfn],cg->fgfs[Gamxxz->sgfn],cg->fgfs[Gamxyy->sgfn],cg->fgfs[Gamxyz->sgfn],cg->fgfs[Gamxzz->sgfn],cg->fgfs[Gamyxx->sgfn],cg->fgfs[Gamyxy->sgfn],cg->fgfs[Gamyxz->sgfn],cg->fgfs[Gamyyy->sgfn],cg->fgfs[Gamyyz->sgfn],cg->fgfs[Gamyzz->sgfn],cg->fgfs[Gamzxx->sgfn],cg->fgfs[Gamzxy->sgfn],cg->fgfs[Gamzxz->sgfn],cg->fgfs[Gamzyy->sgfn],cg->fgfs[Gamzyz->sgfn],cg->fgfs[Gamzzz->sgfn],cg->fgfs[Rxx->sgfn],cg->fgfs[Rxy->sgfn],cg->fgfs[Rxz->sgfn],cg->fgfs[Ryy->sgfn],cg->fgfs[Ryz->sgfn],cg->fgfs[Rzz->sgfn],cg->fgfs[Cons_Ham->sgfn],cg->fgfs[Cons_Px->sgfn],cg->fgfs[Cons_Py->sgfn],cg->fgfs[Cons_Pz->sgfn],cg->fgfs[Cons_Gx->sgfn],cg->fgfs[Cons_Gy->sgfn],cg->fgfs[Cons_Gz->sgfn],Symmetry,lev,ndeps,cor

#define RHS_PARA_CALLED_Constraint_Out cg->shape,TRK4,cg->X[0],cg->X[1],cg->X[2],cg->fgfs[phi0->sgfn],cg->fgfs[trK0->sgfn],cg->fgfs[gxx0->sgfn],cg->fgfs[gxy0->sgfn],cg->fgfs[gxz0->sgfn],cg->fgfs[gyy0->sgfn],cg->fgfs[gyz0->sgfn],cg->fgfs[gzz0->sgfn],cg->fgfs[Axx0->sgfn],cg->fgfs[Axy0->sgfn],cg->fgfs[Axz0->sgfn],cg->fgfs[Ayy0->sgfn],cg->fgfs[Ayz0->sgfn],cg->fgfs[Azz0->sgfn],cg->fgfs[Gmx0->sgfn],cg->fgfs[Gmy0->sgfn],cg->fgfs[Gmz0->sgfn],cg->fgfs[Lap0->sgfn],cg->fgfs[Sfx0->sgfn],cg->fgfs[Sfy0->sgfn],cg->fgfs[Sfz0->sgfn],cg->fgfs[dtSfx0->sgfn],cg->fgfs[dtSfy0->sgfn],cg->fgfs[dtSfz0->sgfn],cg->fgfs[phi_rhs->sgfn],cg->fgfs[trK_rhs->sgfn],cg->fgfs[gxx_rhs->sgfn],cg->fgfs[gxy_rhs->sgfn],cg->fgfs[gxz_rhs->sgfn],cg->fgfs[gyy_rhs->sgfn],cg->fgfs[gyz_rhs->sgfn],cg->fgfs[gzz_rhs->sgfn],cg->fgfs[Axx_rhs->sgfn],cg->fgfs[Axy_rhs->sgfn],cg->fgfs[Axz_rhs->sgfn],cg->fgfs[Ayy_rhs->sgfn],cg->fgfs[Ayz_rhs->sgfn],cg->fgfs[Azz_rhs->sgfn],cg->fgfs[Gmx_rhs->sgfn],cg->fgfs[Gmy_rhs->sgfn],cg->fgfs[Gmz_rhs->sgfn],cg->fgfs[Lap_rhs->sgfn],cg->fgfs[Sfx_rhs->sgfn],cg->fgfs[Sfy_rhs->sgfn],cg->fgfs[Sfz_rhs->sgfn],cg->fgfs[dtSfx_rhs->sgfn],cg->fgfs[dtSfy_rhs->sgfn],cg->fgfs[dtSfz_rhs->sgfn],cg->fgfs[rho->sgfn],cg->fgfs[Sx->sgfn],cg->fgfs[Sy->sgfn],cg->fgfs[Sz->sgfn],cg->fgfs[Sxx->sgfn],cg->fgfs[Sxy->sgfn],cg->fgfs[Sxz->sgfn],cg->fgfs[Syy->sgfn],cg->fgfs[Syz->sgfn],cg->fgfs[Szz->sgfn],cg->fgfs[Gamxxx->sgfn],cg->fgfs[Gamxxy->sgfn],cg->fgfs[Gamxxz->sgfn],cg->fgfs[Gamxyy->sgfn],cg->fgfs[Gamxyz->sgfn],cg->fgfs[Gamxzz->sgfn],cg->fgfs[Gamyxx->sgfn],cg->fgfs[Gamyxy->sgfn],cg->fgfs[Gamyxz->sgfn],cg->fgfs[Gamyyy->sgfn],cg->fgfs[Gamyyz->sgfn],cg->fgfs[Gamyzz->sgfn],cg->fgfs[Gamzxx->sgfn],cg->fgfs[Gamzxy->sgfn],cg->fgfs[Gamzxz->sgfn],cg->fgfs[Gamzyy->sgfn],cg->fgfs[Gamzyz->sgfn],cg->fgfs[Gamzzz->sgfn],cg->fgfs[Rxx->sgfn],cg->fgfs[Rxy->sgfn],cg->fgfs[Rxz->sgfn],cg->fgfs[Ryy->sgfn],cg->fgfs[Ryz->sgfn],cg->fgfs[Rzz->sgfn],cg->fgfs[Cons_Ham->sgfn],cg->fgfs[Cons_Px->sgfn],cg->fgfs[Cons_Py->sgfn],cg->fgfs[Cons_Pz->sgfn],cg->fgfs[Cons_Gx->sgfn],cg->fgfs[Cons_Gy->sgfn],cg->fgfs[Cons_Gz->sgfn],Symmetry,lev,ndeps,pre


#define RHS_PARA_CALLED_Interp_Constraint cg->shape,TRK4,cg->X[0],cg->X[1],cg->X[2],cg->fgfs[phi0->sgfn],cg->fgfs[trK0->sgfn],cg->fgfs[gxx0->sgfn],cg->fgfs[gxy0->sgfn],cg->fgfs[gxz0->sgfn],cg->fgfs[gyy0->sgfn],cg->fgfs[gyz0->sgfn],cg->fgfs[gzz0->sgfn],cg->fgfs[Axx0->sgfn],cg->fgfs[Axy0->sgfn],cg->fgfs[Axz0->sgfn],cg->fgfs[Ayy0->sgfn],cg->fgfs[Ayz0->sgfn],cg->fgfs[Azz0->sgfn],cg->fgfs[Gmx0->sgfn],cg->fgfs[Gmy0->sgfn],cg->fgfs[Gmz0->sgfn],cg->fgfs[Lap0->sgfn],cg->fgfs[Sfx0->sgfn],cg->fgfs[Sfy0->sgfn],cg->fgfs[Sfz0->sgfn],cg->fgfs[dtSfx0->sgfn],cg->fgfs[dtSfy0->sgfn],cg->fgfs[dtSfz0->sgfn],cg->fgfs[phi_rhs->sgfn],cg->fgfs[trK_rhs->sgfn],cg->fgfs[gxx_rhs->sgfn],cg->fgfs[gxy_rhs->sgfn],cg->fgfs[gxz_rhs->sgfn],cg->fgfs[gyy_rhs->sgfn],cg->fgfs[gyz_rhs->sgfn],cg->fgfs[gzz_rhs->sgfn],cg->fgfs[Axx_rhs->sgfn],cg->fgfs[Axy_rhs->sgfn],cg->fgfs[Axz_rhs->sgfn],cg->fgfs[Ayy_rhs->sgfn],cg->fgfs[Ayz_rhs->sgfn],cg->fgfs[Azz_rhs->sgfn],cg->fgfs[Gmx_rhs->sgfn],cg->fgfs[Gmy_rhs->sgfn],cg->fgfs[Gmz_rhs->sgfn],cg->fgfs[Lap_rhs->sgfn],cg->fgfs[Sfx_rhs->sgfn],cg->fgfs[Sfy_rhs->sgfn],cg->fgfs[Sfz_rhs->sgfn],cg->fgfs[dtSfx_rhs->sgfn],cg->fgfs[dtSfy_rhs->sgfn],cg->fgfs[dtSfz_rhs->sgfn],cg->fgfs[rho->sgfn],cg->fgfs[Sx->sgfn],cg->fgfs[Sy->sgfn],cg->fgfs[Sz->sgfn],cg->fgfs[Sxx->sgfn],cg->fgfs[Sxy->sgfn],cg->fgfs[Sxz->sgfn],cg->fgfs[Syy->sgfn],cg->fgfs[Syz->sgfn],cg->fgfs[Szz->sgfn],cg->fgfs[Gamxxx->sgfn],cg->fgfs[Gamxxy->sgfn],cg->fgfs[Gamxxz->sgfn],cg->fgfs[Gamxyy->sgfn],cg->fgfs[Gamxyz->sgfn],cg->fgfs[Gamxzz->sgfn],cg->fgfs[Gamyxx->sgfn],cg->fgfs[Gamyxy->sgfn],cg->fgfs[Gamyxz->sgfn],cg->fgfs[Gamyyy->sgfn],cg->fgfs[Gamyyz->sgfn],cg->fgfs[Gamyzz->sgfn],cg->fgfs[Gamzxx->sgfn],cg->fgfs[Gamzxy->sgfn],cg->fgfs[Gamzxz->sgfn],cg->fgfs[Gamzyy->sgfn],cg->fgfs[Gamzyz->sgfn],cg->fgfs[Gamzzz->sgfn],cg->fgfs[Rxx->sgfn],cg->fgfs[Rxy->sgfn],cg->fgfs[Rxz->sgfn],cg->fgfs[Ryy->sgfn],cg->fgfs[Ryz->sgfn],cg->fgfs[Rzz->sgfn],cg->fgfs[Cons_Ham->sgfn],cg->fgfs[Cons_Px->sgfn],cg->fgfs[Cons_Py->sgfn],cg->fgfs[Cons_Pz->sgfn],cg->fgfs[Cons_Gx->sgfn],cg->fgfs[Cons_Gy->sgfn],cg->fgfs[Cons_Gz->sgfn],Symmetry,lev,ndeps,pre





//4------------tool------------------------------
int compare_result(int ftag1,double * d2,int data_num);



#endif
