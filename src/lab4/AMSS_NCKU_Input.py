
#################################################
##
## This file provides the input parameters required for numerical relativity.
## XIAOQU
## 2024/03/19 --- 2025/09/14
##
#################################################

import numpy    

#################################################

## Setting MPI processes and the output file directory

File_directory   = "GW250118"                    ## output file directory
Output_directory = "binary_output"               ## binary data file directory
                                                 ## The file directory name should not be too long
MPI_processes    = 48                             ## number of mpi processes used in the simulation
OMP_threads      = 1

GPU_Calculation  = "no"                          ## Use GPU or not
                                                 ## (prefer "no" in the current version, because the GPU part may have bugs when integrated in this Python interface)

#################################################


#################################################

## Setting the physical system and numerical method

Symmetry                 = "equatorial-symmetry"   ## fixed in this trimmed lab build
Equation_Class           = "BSSN"                  ## fixed in this trimmed lab build
Initial_Data_Method      = "Ansorg-TwoPuncture"    ## fixed in this trimmed lab build
Time_Evolution_Method    = "runge-kutta-45"        ## time evolution method: choose "runge-kutta-45"
Finite_Diffenence_Method = "4th-order"             ## fixed in this trimmed lab build

#################################################


#################################################

## Setting the time evolutionary information

Start_Evolution_Time     = 0.0                    ## start evolution time t0
Final_Evolution_Time     = 100.0                 ## final evolution time t1
Check_Time               = 1000.0
Dump_Time                = 1000.0                  ## time inteval dT for dumping binary data
D2_Dump_Time             = 1000.0                  ## dump the ascii data for 2d surface after dT'
Analysis_Time            = 0.1                    ## dump the puncture position and GW psi4 after dT"
Evolution_Step_Number    = 10000000               ## stop the calculation after the maximal step number
Courant_Factor           = 0.5                    ## Courant Factor
Dissipation              = 0.15                   ## Kreiss-Oliger Dissipation Strength

#################################################


#################################################

## Setting the grid structure

basic_grid_set    = "Patch"                          ## fixed in this trimmed lab build
grid_center_set   = "Cell"                           ## fixed in this trimmed lab build

grid_level        = 9                                ## total number of AMR grid levels
static_grid_level = 5                                ## number of AMR static grid levels
moving_grid_level = grid_level - static_grid_level   ## number of AMR moving grid levels

analysis_level    = 0
refinement_level  = 3                                ## time refinement start from this grid level

largest_box_xyz_max = [320.0, 320.0, 320.0]          ## scale of the largest box
                                                     ## not ne cess ary to be cubic for "Patch" grid s tructure
                                                     ## need to be a cubic box for "Shell-Patch" grid structure
largest_box_xyz_min = - numpy.array(largest_box_xyz_max)  

static_grid_number = 40                              ## grid points of each static AMR grid (in x direction)
                                                     ## (grid points in y and z directions are automatically adjusted)
moving_grid_number = 48                              ## grid points of each moving AMR grid
shell_grid_number  = [32, 32, 100]                   ## grid points of Shell-Patch grid
                                                     ## in (phi, theta, r) direction
devide_factor      = 2.0                             ## resolution between different grid levels dh0/dh1, only support 2.0 now
                                                     

static_grid_type   = 'Linear'                        ## AMR static grid structure , only supports "Linear"
moving_grid_type   = 'Linear'                        ## AMR moving grid structure , only supports "Linear"

quarter_sphere_number = 96                           ## grid number of 1/4 s pher ical surface
                                                     ## (which is needed for evaluating the spherical surface integral)

#################################################


#################################################

## Setting the puncture information

puncture_number       = 2                                     

position_BH           = numpy.zeros( (puncture_number, 3) )   
parameter_BH          = numpy.zeros( (puncture_number, 3) )   
dimensionless_spin_BH = numpy.zeros( (puncture_number, 3) )   
momentum_BH           = numpy.zeros( (puncture_number, 3) )   

puncture_data_set     = "Manually"                       ## Method to give Puncture’s positions and momentum
                                                         ## choose "Manually" or "Automatically-BBH"
                                                         ## Prefer to choose "Manually", because "Automatically-BBH" is developing now

## initial orbital distance and ellipticity for BBHs system
## ( needed for "Automatically-BBH" case , not affect the "Manually" case )
Distance = 6.0
e0       = 0.0

## black hole parameter (M Q* a*)
parameter_BH[0] = [ 51.5/(51.5+34.5),  0.0,  0.0 ]
parameter_BH[1] = [ 34.5/(51.5+34.5),  0.0,  0.0 ]
## dimensionless spin in each direction
dimensionless_spin_BH[0] = [ 0.0,  0.0,  0.0 ]
dimensionless_spin_BH[1] = [ 0.0,  0.0,  0.0 ]

## use Brugmann's convention
##  -----0-----> y
##   -      +     

#---------------------------------------------

## If puncture_data_set is chosen to be "Manually", it is necessary to set the position and momentum of each puncture manually

## initial position for each puncture
position_BH[0]  = [  0.0,  6.0*34.5/(51.5+34.5),  0.0 ]
position_BH[1]  = [  0.0, -6.0*51.5/(51.5+34.5),  0.0 ]

## initial mumentum for each puncture
## (needed for "Manually" case, does not affect the "Automatically-BBH" case)
momentum_BH[0]  = [ -0.1275,  -0.00514081481119059,   0.0 ]
momentum_BH[1]  = [ +0.1275,  +0.00514081481119059,   0.0 ]


#################################################


#################################################

## Setting the gravitational wave information

GW_L_max        = 4                      ## maximal L number in gravitational wave
GW_M_max        = 4                      ## maximal M number in gravitational wave
Detector_Number = 8                     ## number of dector
Detector_Rmin   = 50.0                   ## nearest dector distance
Detector_Rmax   = 100.0                  ## farest dector distance

#################################################


#################################################

## Setting the apprent horizon

AHF_Find       = "no"                    ## whether to find the apparent horizon: choose "yes" or "no"

AHF_Find_Every = 24
AHF_Dump_Time  = 20.0

#################################################


## Other parameters (testing)
## (please do not change if not necessary)

boundary_choice = "BAM-choice"     ## Sommerfeld boundary condition : choose "BAM-choice" or "Shibata-choice" 
                                   ## prefer "BAM-choice"

gauge_choice  = 0                  ## gauge choice
                                   ## 0: B^i gauge
                                   ## 1: David's puncture gauge
                                   ## 2: MB B^i gauge               
                                   ## 3: RIT B^i gauge
                                   ## 4: MB beta gauge 
                                   ## 5: RIT beta gauge 
                                   ## 6: MGB1 B^i gauge
                                   ## 7: MGB2 B^i gauge
                                   ## prefer 0 or 1
                                   
tetrad_type  = 2                   ## tetradtype 
                                   ##  v:r; u: phi; w: theta
                                   ##      v^a = (x,y,z)
                                   ## 0: orthonormal order: v,u,w
                                   ##    v^a = (x,y,z)   
                                   ##    m = (phi - i theta)/sqrt(2) 
                                   ##    following Frans, Eq.(8) of  PRD 75, 124018(2007)
                                   ## 1: orthonormal order: w,u,v
                                   ##    m = (theta + i phi)/sqrt(2) 
                                   ##    following Sperhake, Eq.(3.2) of  PRD 85, 124062(2012)    
                                   ## 2: orthonormal order: v,u,w
                                   ##    v_a = (x,y,z)
                                   ##    m = (phi - i theta)/sqrt(2) 
                                   ##    following Frans, Eq.(8) of  PRD 75, 124018(2007)
                                   ## this version recommend set to 2
                                   ## prefer 2
                                   
#################################################
                                   
