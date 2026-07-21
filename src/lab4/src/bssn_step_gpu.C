// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sys/time.h>

// include BSSN class files
#include "macrodef.h"
#include "fmisc.h"
#include "bssn_gpu_class.h"
#include "bssn_rhs.h"
#include "enforce_algebra.h"
#include "rungekutta4_rout.h"
#include "sommerfeld_rout.h"

// include gpu files
#include "gpu_manager.h"
#include "helper.h"

void bssn_class::Step_GPU(int lev, int YN) {
	setpbh(BH_num, Porg0, Mass, BH_num_input);

	double dT_lev = dT * pow(0.5, Mymax(lev, trfls));

// new code 2013-2-15, zjcao
	// for black hole position
	if (BH_num > 0 && lev == GH->levels - 1) {
		compute_Porg_rhs(Porg0, Porg_rhs, Sfx0, Sfy0, Sfz0, lev);
		for (int ithBH = 0; ithBH < BH_num; ithBH++) {
			for (int ith = 0; ith < 3; ith++)
				Porg1[ithBH][ith] = Porg0[ithBH][ith] + Porg_rhs[ithBH][ith] * dT_lev;
			if (Symmetry > 0)
				Porg1[ithBH][2] = fabs(Porg1[ithBH][2]);
			if (Symmetry == 2)
			{
				Porg1[ithBH][0] = fabs(Porg1[ithBH][0]);
				Porg1[ithBH][1] = fabs(Porg1[ithBH][1]);
			}
			if (!finite(Porg1[ithBH][0]) || !finite(Porg1[ithBH][1]) || !finite(Porg1[ithBH][2]))
			{
				if (ErrorMonitor->outfile)
					ErrorMonitor->outfile << "predictor step finds NaN for BH's position from ("
																<< Porg0[ithBH][0] << "," << Porg0[ithBH][1] << "," << Porg0[ithBH][2] << ")" << endl;
				cout << "predictor step finds NaN for BH's position from ("
					 << Porg0[ithBH][0] << "," << Porg0[ithBH][1] << "," << Porg0[ithBH][2] << ")" << endl;
				cout << "to ("
					 << Porg1[ithBH][0] << "," << Porg1[ithBH][1] << "," << Porg1[ithBH][2] << ")" << endl;
				cout << "with velocity ("
					 << Porg_rhs[ithBH][0] << "," << Porg_rhs[ithBH][1] << "," << Porg_rhs[ithBH][2] << ")" << endl;
				MPI_Abort(MPI_COMM_WORLD, 1);
				MyList<var> *DG_List = new MyList<var>(Sfx0);
				DG_List->insert(Sfx0);
				DG_List->insert(Sfy0);
				DG_List->insert(Sfz0);
				Parallel::Dump_Data(GH->PatL[lev], DG_List, 0, PhysTime, dT_lev);
				DG_List->clearList();
			}
		}
	}

	// data analysis part
	// Warning NOTE: the variables1 are used as temp storege room
	if (lev == a_lev)
	{
		AnalysisStuff(lev, dT_lev);
	}

	bool BB = fgt(PhysTime, StartTime, dT_lev / 2);
	double ndeps = numepss;
	if (lev < GH->movls)
		ndeps = numepsb;
	double TRK4 = PhysTime;
	int iter_count = 0; // count RK4 substeps
	int pre = 0, cor = 1;
	int ERROR = 0;

	// Predictor
	MyList<Patch> *Pp = GH->PatL[lev];
	while (Pp) {
		MyList<Block> *BP = Pp->data->blb;
		while (BP) {
			Block *cg = BP->data;
			if (myrank == cg->rank) {
				gpu_enforce_ga_launch(
					cg->stream, cg->shape,
					cg->d_fgfs[gxx0->sgfn], cg->d_fgfs[gxy0->sgfn], cg->d_fgfs[gxz0->sgfn], cg->d_fgfs[gyy0->sgfn], cg->d_fgfs[gyz0->sgfn], cg->d_fgfs[gzz0->sgfn],
					cg->d_fgfs[Axx0->sgfn], cg->d_fgfs[Axy0->sgfn], cg->d_fgfs[Axz0->sgfn], cg->d_fgfs[Ayy0->sgfn], cg->d_fgfs[Ayz0->sgfn], cg->d_fgfs[Azz0->sgfn]
				);

				gpu_compute_rhs_bssn_launch(
					cg->stream,
					cg->shape, TRK4, cg->d_X[0], cg->d_X[1], cg->d_X[2],
					cg->d_fgfs[phi0->sgfn], cg->d_fgfs[trK0->sgfn],
					cg->d_fgfs[gxx0->sgfn], cg->d_fgfs[gxy0->sgfn], cg->d_fgfs[gxz0->sgfn], 
					cg->d_fgfs[gyy0->sgfn], cg->d_fgfs[gyz0->sgfn], cg->d_fgfs[gzz0->sgfn],
					cg->d_fgfs[Axx0->sgfn], cg->d_fgfs[Axy0->sgfn], cg->d_fgfs[Axz0->sgfn], 
					cg->d_fgfs[Ayy0->sgfn], cg->d_fgfs[Ayz0->sgfn], cg->d_fgfs[Azz0->sgfn],
					cg->d_fgfs[Gmx0->sgfn], cg->d_fgfs[Gmy0->sgfn], cg->d_fgfs[Gmz0->sgfn],
					cg->d_fgfs[Lap0->sgfn], 
					cg->d_fgfs[Sfx0->sgfn], cg->d_fgfs[Sfy0->sgfn], cg->d_fgfs[Sfz0->sgfn],
					cg->d_fgfs[dtSfx0->sgfn], cg->d_fgfs[dtSfy0->sgfn], cg->d_fgfs[dtSfz0->sgfn],
					cg->d_fgfs[phi_rhs->sgfn], cg->d_fgfs[trK_rhs->sgfn],
					cg->d_fgfs[gxx_rhs->sgfn], cg->d_fgfs[gxy_rhs->sgfn], cg->d_fgfs[gxz_rhs->sgfn],
					cg->d_fgfs[gyy_rhs->sgfn], cg->d_fgfs[gyz_rhs->sgfn], cg->d_fgfs[gzz_rhs->sgfn],
					cg->d_fgfs[Axx_rhs->sgfn], cg->d_fgfs[Axy_rhs->sgfn], cg->d_fgfs[Axz_rhs->sgfn],
					cg->d_fgfs[Ayy_rhs->sgfn], cg->d_fgfs[Ayz_rhs->sgfn], cg->d_fgfs[Azz_rhs->sgfn],
					cg->d_fgfs[Gmx_rhs->sgfn], cg->d_fgfs[Gmy_rhs->sgfn], cg->d_fgfs[Gmz_rhs->sgfn],
					cg->d_fgfs[Lap_rhs->sgfn],
					cg->d_fgfs[Sfx_rhs->sgfn], cg->d_fgfs[Sfy_rhs->sgfn], cg->d_fgfs[Sfz_rhs->sgfn],
					cg->d_fgfs[dtSfx_rhs->sgfn], cg->d_fgfs[dtSfy_rhs->sgfn], cg->d_fgfs[dtSfz_rhs->sgfn],
					cg->d_fgfs[rho->sgfn], cg->d_fgfs[Sx->sgfn], cg->d_fgfs[Sy->sgfn], cg->d_fgfs[Sz->sgfn],
					cg->d_fgfs[Sxx->sgfn], cg->d_fgfs[Sxy->sgfn], cg->d_fgfs[Sxz->sgfn], 
					cg->d_fgfs[Syy->sgfn], cg->d_fgfs[Syz->sgfn], cg->d_fgfs[Szz->sgfn],
					cg->d_fgfs[Gamxxx->sgfn], cg->d_fgfs[Gamxxy->sgfn], cg->d_fgfs[Gamxxz->sgfn],
					cg->d_fgfs[Gamxyy->sgfn], cg->d_fgfs[Gamxyz->sgfn], cg->d_fgfs[Gamxzz->sgfn],
					cg->d_fgfs[Gamyxx->sgfn], cg->d_fgfs[Gamyxy->sgfn], cg->d_fgfs[Gamyxz->sgfn],
					cg->d_fgfs[Gamyyy->sgfn], cg->d_fgfs[Gamyyz->sgfn], cg->d_fgfs[Gamyzz->sgfn],
					cg->d_fgfs[Gamzxx->sgfn], cg->d_fgfs[Gamzxy->sgfn], cg->d_fgfs[Gamzxz->sgfn],
					cg->d_fgfs[Gamzyy->sgfn], cg->d_fgfs[Gamzyz->sgfn], cg->d_fgfs[Gamzzz->sgfn],
					cg->d_fgfs[Rxx->sgfn], cg->d_fgfs[Rxy->sgfn], cg->d_fgfs[Rxz->sgfn], 
					cg->d_fgfs[Ryy->sgfn], cg->d_fgfs[Ryz->sgfn], cg->d_fgfs[Rzz->sgfn],
					cg->d_fgfs[Cons_Ham->sgfn],
					cg->d_fgfs[Cons_Px->sgfn], cg->d_fgfs[Cons_Py->sgfn], cg->d_fgfs[Cons_Pz->sgfn],
					cg->d_fgfs[Cons_Gx->sgfn], cg->d_fgfs[Cons_Gy->sgfn], cg->d_fgfs[Cons_Gz->sgfn],
					Symmetry, lev, ndeps, pre
				);

				MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varlrhs = RHSList; // we do not check the correspondence here
				while (varl0) {
					if (lev == 0) { // sommerfeld indeed
						gpu_sommerfeld_routbam_launch(
							cg->stream,
							cg->shape,
							cg->d_X[0], cg->d_X[1], cg->d_X[2],
							Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
							cg->d_fgfs[varlrhs->data->sgfn],
							cg->d_fgfs[varl0->data->sgfn], varl0->data->propspeed, varl0->data->SoA,
							Symmetry
						);
					}
					gpu_rungekutta4_rout_launch(
						cg->stream,
						cg->shape, dT_lev, 
						cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl->data->sgfn], cg->d_fgfs[varlrhs->data->sgfn], iter_count
					);
					if (lev > 0) {// fix BD point
						gpu_sommerfeld_rout_launch(
							cg->stream,
							cg->shape,
							cg->d_X[0], cg->d_X[1], cg->d_X[2],
							Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
							dT_lev, cg->d_fgfs[phi0->sgfn],
							cg->d_fgfs[Lap0->sgfn], cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl->data->sgfn], varl0->data->SoA,
							Symmetry, cor
						);
					}

					varl0 = varl0->next;
					varl = varl->next;
					varlrhs = varlrhs->next;
				}
				gpu_lowerboundset_launch(cg->stream, cg->shape, cg->d_fgfs[phi->sgfn], chitiny);
			}
			if (BP == Pp->data->ble)
				break;
			BP = BP->next;
		}
		Pp = Pp->next;
	}
	GPUManager::getInstance().synchronize_all();
	// check error information
	{
		int erh = ERROR;
		MPI_Allreduce(&erh, &ERROR, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}
	if (ERROR) {
		Parallel::Dump_Data(GH->PatL[lev], StateList, 0, PhysTime, dT_lev);
		if (myrank == 0) {
			if (ErrorMonitor->outfile) ErrorMonitor->outfile << "find NaN in state variables at t = " << PhysTime << ", lev = " << lev << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	Parallel::Sync_GPU(GH->PatL[lev], SynchList_pre, Symmetry);
	// corrector
	for (iter_count = 1; iter_count < 4; iter_count++) {
		// for RK4: t0, t0+dt/2, t0+dt/2, t0+dt;
		if (iter_count == 1 || iter_count == 3) TRK4 += dT_lev / 2;
		Pp = GH->PatL[lev];
		while (Pp) {
			MyList<Block> *BP = Pp->data->blb;
			while (BP) {
				Block *cg = BP->data;
				if (myrank == cg->rank) {	
					gpu_enforce_ga_launch(
						cg->stream, cg->shape,
						cg->d_fgfs[gxx->sgfn], cg->d_fgfs[gxy->sgfn], cg->d_fgfs[gxz->sgfn], cg->d_fgfs[gyy->sgfn], cg->d_fgfs[gyz->sgfn], cg->d_fgfs[gzz->sgfn],
						cg->d_fgfs[Axx->sgfn], cg->d_fgfs[Axy->sgfn], cg->d_fgfs[Axz->sgfn], cg->d_fgfs[Ayy->sgfn], cg->d_fgfs[Ayz->sgfn], cg->d_fgfs[Azz->sgfn]
					);

					gpu_compute_rhs_bssn_launch(
						cg->stream,
						cg->shape, TRK4, cg->d_X[0], cg->d_X[1], cg->d_X[2],
						cg->d_fgfs[phi->sgfn], cg->d_fgfs[trK->sgfn],
						cg->d_fgfs[gxx->sgfn], cg->d_fgfs[gxy->sgfn], cg->d_fgfs[gxz->sgfn], 
						cg->d_fgfs[gyy->sgfn], cg->d_fgfs[gyz->sgfn], cg->d_fgfs[gzz->sgfn],
						cg->d_fgfs[Axx->sgfn], cg->d_fgfs[Axy->sgfn], cg->d_fgfs[Axz->sgfn], 
						cg->d_fgfs[Ayy->sgfn], cg->d_fgfs[Ayz->sgfn], cg->d_fgfs[Azz->sgfn],
						cg->d_fgfs[Gmx->sgfn], cg->d_fgfs[Gmy->sgfn], cg->d_fgfs[Gmz->sgfn],
						cg->d_fgfs[Lap->sgfn], 
						cg->d_fgfs[Sfx->sgfn], cg->d_fgfs[Sfy->sgfn], cg->d_fgfs[Sfz->sgfn],
						cg->d_fgfs[dtSfx->sgfn], cg->d_fgfs[dtSfy->sgfn], cg->d_fgfs[dtSfz->sgfn],
						cg->d_fgfs[phi1->sgfn], cg->d_fgfs[trK1->sgfn],
						cg->d_fgfs[gxx1->sgfn], cg->d_fgfs[gxy1->sgfn], cg->d_fgfs[gxz1->sgfn],
						cg->d_fgfs[gyy1->sgfn], cg->d_fgfs[gyz1->sgfn], cg->d_fgfs[gzz1->sgfn],
						cg->d_fgfs[Axx1->sgfn], cg->d_fgfs[Axy1->sgfn], cg->d_fgfs[Axz1->sgfn],
						cg->d_fgfs[Ayy1->sgfn], cg->d_fgfs[Ayz1->sgfn], cg->d_fgfs[Azz1->sgfn],
						cg->d_fgfs[Gmx1->sgfn], cg->d_fgfs[Gmy1->sgfn], cg->d_fgfs[Gmz1->sgfn],
						cg->d_fgfs[Lap1->sgfn], 
						cg->d_fgfs[Sfx1->sgfn], cg->d_fgfs[Sfy1->sgfn], cg->d_fgfs[Sfz1->sgfn],
						cg->d_fgfs[dtSfx1->sgfn], cg->d_fgfs[dtSfy1->sgfn], cg->d_fgfs[dtSfz1->sgfn],
						cg->d_fgfs[rho->sgfn], 
						cg->d_fgfs[Sx->sgfn], cg->d_fgfs[Sy->sgfn], cg->d_fgfs[Sz->sgfn],
						cg->d_fgfs[Sxx->sgfn], cg->d_fgfs[Sxy->sgfn], cg->d_fgfs[Sxz->sgfn], 
						cg->d_fgfs[Syy->sgfn], cg->d_fgfs[Syz->sgfn], cg->d_fgfs[Szz->sgfn],
						cg->d_fgfs[Gamxxx->sgfn], cg->d_fgfs[Gamxxy->sgfn], cg->d_fgfs[Gamxxz->sgfn],
						cg->d_fgfs[Gamxyy->sgfn], cg->d_fgfs[Gamxyz->sgfn], cg->d_fgfs[Gamxzz->sgfn],
						cg->d_fgfs[Gamyxx->sgfn], cg->d_fgfs[Gamyxy->sgfn], cg->d_fgfs[Gamyxz->sgfn],
						cg->d_fgfs[Gamyyy->sgfn], cg->d_fgfs[Gamyyz->sgfn], cg->d_fgfs[Gamyzz->sgfn],
						cg->d_fgfs[Gamzxx->sgfn], cg->d_fgfs[Gamzxy->sgfn], cg->d_fgfs[Gamzxz->sgfn],
						cg->d_fgfs[Gamzyy->sgfn], cg->d_fgfs[Gamzyz->sgfn], cg->d_fgfs[Gamzzz->sgfn],
						cg->d_fgfs[Rxx->sgfn], cg->d_fgfs[Rxy->sgfn], cg->d_fgfs[Rxz->sgfn], 
						cg->d_fgfs[Ryy->sgfn], cg->d_fgfs[Ryz->sgfn], cg->d_fgfs[Rzz->sgfn],
						cg->d_fgfs[Cons_Ham->sgfn],
						cg->d_fgfs[Cons_Px->sgfn], cg->d_fgfs[Cons_Py->sgfn], cg->d_fgfs[Cons_Pz->sgfn],
						cg->d_fgfs[Cons_Gx->sgfn], cg->d_fgfs[Cons_Gy->sgfn], cg->d_fgfs[Cons_Gz->sgfn],
						Symmetry, lev, ndeps, cor
					);

					MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varl1 = SynchList_cor, *varlrhs = RHSList; // we do not check the correspondence here
					while (varl0) {
						if (lev == 0) { // sommerfeld indeed
							gpu_sommerfeld_routbam_launch(
								cg->stream, cg->shape, cg->d_X[0], cg->d_X[1], cg->d_X[2],
								Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
								cg->d_fgfs[varl1->data->sgfn],
								cg->d_fgfs[varl->data->sgfn], varl0->data->propspeed, varl0->data->SoA,
								Symmetry
							);
						}
						gpu_rungekutta4_rout_launch(
							cg->stream, cg->shape, dT_lev, 
							cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl1->data->sgfn], cg->d_fgfs[varlrhs->data->sgfn], iter_count
						);
						if (lev > 0) { // fix BD point
							gpu_sommerfeld_rout_launch(
								cg->stream, cg->shape, cg->d_X[0], cg->d_X[1], cg->d_X[2],
								Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
								dT_lev, cg->d_fgfs[phi0->sgfn],
								cg->d_fgfs[Lap0->sgfn], cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl1->data->sgfn], varl0->data->SoA,
								Symmetry, cor
							);
						}

						varl0 = varl0->next;
						varl = varl->next;
						varl1 = varl1->next;
						varlrhs = varlrhs->next;
					}

					gpu_lowerboundset_launch(cg->stream, cg->shape, cg->d_fgfs[phi1->sgfn], chitiny);
				}
				if (BP == Pp->data->ble)
					break;
				BP = BP->next;
			}
			Pp = Pp->next;
		}
		GPUManager::getInstance().synchronize_all();
		// check error information
		{
			int erh = ERROR;
			MPI_Allreduce(&erh, &ERROR, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		}

		if (ERROR) {
			Parallel::Dump_Data(GH->PatL[lev], SynchList_pre, 0, PhysTime, dT_lev);
			if (myrank == 0) {
				if (ErrorMonitor->outfile)
					ErrorMonitor->outfile << "find NaN in RK4 substep#" << iter_count << " variables at t = " << PhysTime << ", lev = " << lev << endl;
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		Parallel::Sync_GPU(GH->PatL[lev], SynchList_cor, Symmetry);

		// swap time level
		if (iter_count < 3) {
			Pp = GH->PatL[lev];
			while (Pp) {
				MyList<Block> *BP = Pp->data->blb;
				while (BP) {
					Block *cg = BP->data;
					cg->swapList(SynchList_pre, SynchList_cor, myrank);
					if (BP == Pp->data->ble)
						break;
					BP = BP->next;
				}
				Pp = Pp->next;
			}
		}
	}
	// note the data structure before update
	// SynchList_cor 1   -----------
	//
	// StateList     0   -----------
	//
	// OldStateList  old -----------
	// update
	Pp = GH->PatL[lev];
	while (Pp)
	{
		MyList<Block> *BP = Pp->data->blb;
		while (BP)
		{
			Block *cg = BP->data;
			cg->swapList(StateList, SynchList_cor, myrank);
			cg->swapList(OldStateList, SynchList_cor, myrank);
			if (BP == Pp->data->ble)
				break;
			BP = BP->next;
		}
		Pp = Pp->next;
	}
	// for black hole position
	if (BH_num > 0 && lev == GH->levels - 1)
	{
		for (int ithBH = 0; ithBH < BH_num; ithBH++)
		{
			Porg0[ithBH][0] = Porg1[ithBH][0];
			Porg0[ithBH][1] = Porg1[ithBH][1];
			Porg0[ithBH][2] = Porg1[ithBH][2];
		}
	}
}
