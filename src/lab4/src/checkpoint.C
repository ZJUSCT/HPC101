
#include <cstdio>
using namespace std;

#include "checkpoint.h"
#include "misc.h"
#include "fmisc.h"
#include "parameters.h"

checkpoint::checkpoint(bool checked, const char fname[], int myrank) : filename(0), CheckList(0), checkedrun(checked)
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
		cout << "checkpoint 01" << endl;
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

	I_Print = (myrank == 0);

	int i = strlen(fname);
	filename = new char[i+30];
	// cout << filename << endl;
	// cout << i << endl;
	// cout << "checkpoint 5" << endl;
	sprintf(filename, "%s/%s", out_dir.c_str(), fname);
	// cout << "checkpoint 6" << endl;
        if (myrank==0) {
                cout << " checkpoint class created " << endl;
        }
}
checkpoint::~checkpoint()
{
	CheckList->clearList();
	if (I_Print)
		delete[] filename;
}

void checkpoint::addvariable(var *VV)
{
	if (!VV)
		return;

	if (CheckList)
		CheckList->insert(VV);
	else
		CheckList = new MyList<var>(VV);
}
void checkpoint::addvariablelist(MyList<var> *VL)
{
	while (VL)
	{
		if (CheckList)
			CheckList->insert(VL->data);
		else
			CheckList = new MyList<var>(VL->data);
		VL = VL->next;
	}
}

void checkpoint::writecheck_cgh(double time, cgh *GH)
{
	ofstream outfile;

	if (I_Print)
	{
		// char fname[50];
		char fname[50+50]; 
		sprintf(fname, "%s_cgh.CHK", filename);

		outfile.open(fname, ios::out | ios::trunc);
		if (!outfile)
		{
			cout << "Can't open " << fname << " for check point out." << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		outfile.write((char *)&time, sizeof(double));
		outfile.write((char *)&(GH->levels), sizeof(int));
		outfile.write((char *)&(GH->movls), sizeof(int));
		outfile.write((char *)&(GH->BH_num_in), sizeof(int));
		outfile.write((char *)GH->grids, GH->levels * sizeof(int));
		outfile.write((char *)GH->Lt, GH->levels * sizeof(double));
		for (int lev = 0; lev < GH->levels; lev++)
		{
			for (int grd = 0; grd < GH->grids[lev]; grd++)
			{
				outfile.write((char *)GH->bbox[lev][grd], 6 * sizeof(double));
				outfile.write((char *)GH->shape[lev][grd], 3 * sizeof(int));
				outfile.write((char *)GH->handle[lev][grd], 3 * sizeof(double));
			}
			for (int ibh = 0; ibh < GH->BH_num_in; ibh++)
			{
				outfile.write((char *)GH->Porgls[lev][ibh], 3 * sizeof(double));
			}
		}
	}
	// write variable data
	for (int lev = 0; lev < GH->levels; lev++)
	{
		MyList<Patch> *PL = GH->PatL[lev];
		while (PL)
		{
			Patch *PP = PL->data;
			int nn = PP->shape[0] * PP->shape[1] * PP->shape[2];
			MyList<var> *VL = CheckList;
			while (VL)
			{
				double *databuffer = Parallel::Collect_Data(PP, VL->data);
				if (I_Print)
					outfile.write((char *)databuffer, sizeof(double) * nn);
				if (databuffer)
					delete[] databuffer;
				VL = VL->next;
			}
			PL = PL->next;
		}
	}

	if (I_Print)
		outfile.close();
}
void checkpoint::readcheck_cgh(double &time, cgh *GH, int myrank, int nprocs, int Symmetry)
{
	int DIM = dim;
	ifstream infile;
	// char fname[50];
	char fname[50+50]; 
	sprintf(fname, "%s_cgh.CHK", filename);

	infile.open(fname);
	if (!infile)
	{
		cout << "Can't open " << fname << " for check point in." << endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	infile.seekg(0, ios::beg);
	infile.read((char *)&time, sizeof(double));
	if (I_Print)
		cout << "check cgh in at t = " << time << endl;
	infile.read((char *)&(GH->levels), sizeof(int));
	infile.read((char *)&(GH->movls), sizeof(int));
	infile.read((char *)&(GH->BH_num_in), sizeof(int));
	GH->grids = new int[GH->levels];
	GH->bbox = new double **[GH->levels];
	GH->shape = new int **[GH->levels];
	GH->handle = new double **[GH->levels];
	GH->PatL = new MyList<Patch> *[GH->levels];
	GH->Lt = new double[GH->levels];
	GH->Porgls = new double **[GH->levels];
	infile.read((char *)GH->grids, GH->levels * sizeof(int));
	infile.read((char *)GH->Lt, GH->levels * sizeof(double));
	for (int lev = 0; lev < GH->levels; lev++)
	{
		GH->bbox[lev] = new double *[GH->grids[lev]];
		GH->shape[lev] = new int *[GH->grids[lev]];
		GH->handle[lev] = new double *[GH->grids[lev]];
		GH->Porgls[lev] = new double *[GH->BH_num_in];
		for (int grd = 0; grd < GH->grids[lev]; grd++)
		{
			GH->bbox[lev][grd] = new double[6];
			GH->shape[lev][grd] = new int[3];
			GH->handle[lev][grd] = new double[3];
			infile.read((char *)GH->bbox[lev][grd], 6 * sizeof(double));
			infile.read((char *)GH->shape[lev][grd], 3 * sizeof(int));
			infile.read((char *)GH->handle[lev][grd], 3 * sizeof(double));
		}
		for (int ibh = 0; ibh < GH->BH_num_in; ibh++)
		{
			GH->Porgls[lev][ibh] = new double[dim];
			infile.read((char *)GH->Porgls[lev][ibh], 3 * sizeof(double));
		}
	}

	for (int lev = 0; lev < GH->levels; lev++)
		GH->PatL[lev] = GH->construct_patchlist(lev, Symmetry);

	GH->compose_cgh(nprocs);
	// write variable data
	for (int lev = 0; lev < GH->levels; lev++)
	{
		MyList<Patch> *PL = GH->PatL[lev];
		while (PL)
		{
			Patch *PP = PL->data;
			int nn = PP->shape[0] * PP->shape[1] * PP->shape[2];
			double *databuffer = new double[nn];
			MyList<var> *VL = CheckList;
			while (VL)
			{
				infile.read((char *)databuffer, sizeof(double) * nn);

				{
					MyList<Block> *BL = PP->blb;
					while (BL)
					{
						Block *cg = BL->data;
						if (myrank == cg->rank)
						{
							f_copy(DIM, cg->bbox, cg->bbox + DIM, cg->shape, cg->fgfs[VL->data->sgfn],
								   PP->bbox, PP->bbox + DIM, PP->shape, databuffer,
								   cg->bbox, cg->bbox + DIM);
						}
						if (BL == PP->ble)
							break;
						BL = BL->next;
					}
				}

				VL = VL->next;
			}
			delete[] databuffer;
			PL = PL->next;
		}
	}

	infile.close();
}
void checkpoint::write_Black_Hole_position(int BH_num_input, int BH_num, double **Porg0, double **Porgbr, double *Mass)
{
	ofstream outfile;

	if (I_Print)
	{
		char fname[50];
		sprintf(fname, "%s_BHp.CHK", filename);

		outfile.open(fname, ios::out | ios::trunc);
		if (!outfile)
		{
			cout << "Can't open " << fname << " for check point out." << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		outfile.write((char *)&BH_num_input, sizeof(int));
		outfile.write((char *)&BH_num, sizeof(int));
		outfile.write((char *)Mass, 3 * sizeof(double));
		for (int i = 0; i < BH_num; i++)
		{
			outfile.write((char *)Porg0[i], 3 * sizeof(double));
			outfile.write((char *)Porgbr[i], 3 * sizeof(double));
		}

		outfile.close();
	}
}
void checkpoint::read_Black_Hole_position(int &BH_num_input, int &BH_num, double **&Porg0, double *&Pmom,
										  double *&Spin, double *&Mass, double **&Porgbr, double **&Porg,
										  double **&Porg1, double **&Porg_rhs)
{
	ifstream infile;
	// char fname[50];
	char fname[50+50]; 
	sprintf(fname, "%s_BHp.CHK", filename);

	infile.open(fname);
	if (!infile)
	{
		cout << "Can't open " << fname << " for check point in." << endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	else if (I_Print)
		cout << "checking in Black_Hole_position" << endl;

	infile.seekg(0, ios::beg);
	infile.read((char *)&BH_num_input, sizeof(int));
	infile.read((char *)&BH_num, sizeof(int));
	// these arrays will be deleted when bssn_class is deleted
	Pmom = new double[3 * BH_num];
	Spin = new double[3 * BH_num];
	Mass = new double[BH_num];
	Porg0 = new double *[BH_num];
	Porgbr = new double *[BH_num];
	Porg = new double *[BH_num];
	Porg1 = new double *[BH_num];
	Porg_rhs = new double *[BH_num];
	infile.read((char *)Mass, 3 * sizeof(double));
	for (int i = 0; i < BH_num; i++)
	{
		Porg0[i] = new double[3];
		Porgbr[i] = new double[3];
		Porg[i] = new double[3];
		Porg1[i] = new double[3];
		Porg_rhs[i] = new double[3];
		infile.read((char *)Porg0[i], 3 * sizeof(double));
		infile.read((char *)Porgbr[i], 3 * sizeof(double));
	}

	infile.close();
}
void checkpoint::write_bssn(double LastDump, double Last2dDump, double LastAnas)
{
	ofstream outfile;

	if (I_Print)
	{
		char fname[50];
		sprintf(fname, "%s_bssn.CHK", filename);

		outfile.open(fname, ios::out | ios::trunc);
		if (!outfile)
		{
			cout << "Can't open " << fname << " for check point out." << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		outfile.write((char *)&LastDump, sizeof(double));
		outfile.write((char *)&Last2dDump, sizeof(double));
		outfile.write((char *)&LastAnas, sizeof(double));

		outfile.close();
	}
}
void checkpoint::read_bssn(double &LastDump, double &Last2dDump, double &LastAnas)
{
	ifstream infile;
	// char fname[50];
	char fname[50+50]; 
	sprintf(fname, "%s_bssn.CHK", filename);

	infile.open(fname);
	if (!infile)
	{
		cout << "Can't open " << fname << " for check point in." << endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	else if (I_Print)
		cout << "checking in bssn parameters" << endl;

	infile.seekg(0, ios::beg);
	infile.read((char *)&LastDump, sizeof(double));
	infile.read((char *)&Last2dDump, sizeof(double));
	infile.read((char *)&LastAnas, sizeof(double));

	infile.close();
}