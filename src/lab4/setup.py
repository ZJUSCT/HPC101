
##################################################################
##
## definition of printing the basic information of AMSS-NCKU program
## author:xiaoqu
## 2024/03/22
## 2025/09/13 modified
##
##################################################################

import AMSS_NCKU_Input as input_data
import numpy 
import os
import math

##################################################################

devide_factor = input_data.devide_factor

static_grid_level = input_data.static_grid_level
moving_grid_level = input_data.moving_grid_level
total_grid_level  = input_data.grid_level

static_grid_number = input_data.static_grid_number
moving_grid_number = input_data.moving_grid_number

if ( input_data.Symmetry=="octant-symmetry" ):
    maximal_domain_size_static_x = numpy.array( [ 0.0, input_data.largest_box_xyz_max[0] ] )
    maximal_domain_size_static_y = numpy.array( [ 0.0, input_data.largest_box_xyz_max[1] ] )
    maximal_domain_size_static_z = numpy.array( [ 0.0, input_data.largest_box_xyz_max[2] ] )
elif( input_data.Symmetry=="octant-symmetry" ):
    maximal_domain_size_static_x = numpy.array( [ input_data.largest_box_xyz_min[0], input_data.largest_box_xyz_max[0] ] )
    maximal_domain_size_static_y = numpy.array( [ input_data.largest_box_xyz_min[1], input_data.largest_box_xyz_max[1] ] )
    maximal_domain_size_static_z = numpy.array( [ 0.0, input_data.largest_box_xyz_max[2] ] )
else:
    maximal_domain_size_static_x = numpy.array( [ input_data.largest_box_xyz_min[0], input_data.largest_box_xyz_max[0] ] )
    maximal_domain_size_static_y = numpy.array( [ input_data.largest_box_xyz_min[1], input_data.largest_box_xyz_max[1] ] )
    maximal_domain_size_static_z = numpy.array( [ input_data.largest_box_xyz_min[2], input_data.largest_box_xyz_max[2] ] )
    
minimal_domain_size_static_x =   maximal_domain_size_static_x / ( (devide_factor)**(static_grid_level-1) )
minimal_domain_size_static_y =   maximal_domain_size_static_y / ( (devide_factor)**(static_grid_level-1) )
minimal_domain_size_static_z =   maximal_domain_size_static_z / ( (devide_factor)**(static_grid_level-1) )
maximal_domain_size_moving = ( minimal_domain_size_static_x / devide_factor ) * ( moving_grid_number / static_grid_number )
minimal_domain_size_moving =   maximal_domain_size_moving / ( (devide_factor)**(input_data.moving_grid_level-1) )

maximal_resolution_static = (input_data.largest_box_xyz_max[0] - input_data.largest_box_xyz_min[0]) / static_grid_number
minimal_resolution_static = maximal_resolution_static / ( (devide_factor)**(static_grid_level-1) )
maximal_resolution_moving = minimal_resolution_static / devide_factor
minimal_resolution_moving = maximal_resolution_moving / ( (devide_factor)**(moving_grid_level-1) )

TimeStep = input_data.Courant_Factor * maximal_resolution_static / ( (devide_factor)**(input_data.refinement_level) )

shell_grid_number              = input_data.shell_grid_number
minimal_domain_size_shellpatch = input_data.largest_box_xyz_max[0]
maximal_domain_size_shellpatch = input_data.largest_box_xyz_max[0] + maximal_resolution_static * shell_grid_number[2]
shellpatch_resolution_R        = maximal_resolution_static
shellpatch_resolution_theta    = 0.5 * math.pi / shell_grid_number[1]
shellpatch_resolution_phi      = 0.5 * math.pi / shell_grid_number[0]

##################################################################

## this function is used to print the basic input data of the whole program

def print_input_data( File_directory ):
        
    print( "------------------------------------------------------------------------------------------" )
    print(                                                                                           )
    print( " Printing the basic parameter and setting in the AMSS-NCKU simulation "                  )
    print(                                                                                           )
    print( " The number of MPI processes in the AMSS-NCKU simulation = ", input_data.MPI_processes   ) 
    print(                                                                                           )
    print( " The form of computational equation  = ",            input_data.Equation_Class           )
    print( " The initial data in this simulation = ",            input_data.Initial_Data_Method      )
    print(                                                                                           )
    print( " Starting evolution time   = ",                      input_data.Start_Evolution_Time     ) 
    print( " Final evolution time      = ",                      input_data.Final_Evolution_Time     ) 
    print( " Maximal iteration number  = ",                      input_data.Evolution_Step_Number    )
    print( " Courant factor            = ",                      input_data.Courant_Factor           )
    print( " Strength of dissipation   = ",                      input_data.Dissipation              )
    print( " Symmetry of system        = ",                      input_data.Symmetry                 )
    print( " The Runge-Kutta scheme in the time evolution   = ", input_data.Time_Evolution_Method    ) 
    print( " The finite-difference scheme in the simulation = ", input_data.Finite_Diffenence_Method )
    print(                                                                                           )
    print( " The static AMR grid type = ",                       input_data.static_grid_type         )
    print( " The moving AMR grid type = ",                       input_data.moving_grid_type         )
    print(                                                                                           )
    print( " The number of static AMR grid levels = ",           static_grid_level                   )      
    print( " The number of moving AMR grid levels = ",           moving_grid_level                   )      
    print( " The number of total  AMR grid levels = ",           total_grid_level                    )
    print(                                                                                           )
    print( " The grid number of each static AMR grid level = ",  static_grid_number                  )
    print( " The grid number of each moving AMR grid level = ",  moving_grid_number                  )
    print(                                                                                           )
    print( " The scale for largest  static AMR grid in X direction = ", maximal_domain_size_static_x )
    print( " The scale for largest  static AMR grid in Y direction = ", maximal_domain_size_static_y )
    print( " The scale for largest  static AMR grid in Z direction = ", maximal_domain_size_static_z )
    print( " The scale for smallest static AMR grid in X direction = ", minimal_domain_size_static_x )
    print( " The scale for smallest static AMR grid in Y direction = ", minimal_domain_size_static_y )
    print( " The scale for smallest static AMR grid in Z direction = ", minimal_domain_size_static_z )
    print(                                                                                           )
    
    if ( input_data.moving_grid_level > 0):
        print( " The scale for largest  moving AMR grid = ",     maximal_domain_size_moving          )
        print( " The scale for smallest moving AMR grid = ",     minimal_domain_size_moving          )

    print(                                                                                           )
    print( " The coarest resolution for static AMR grid = ",     maximal_resolution_static           )
    print( " The finest  resolution for static AMR grid = ",     minimal_resolution_static           )
    
    if ( input_data.moving_grid_level > 0):
        print( " The coarest resolution for moving AMR grid = ", maximal_resolution_moving           )
        print( " The finest  resolution for moving AMR grid = ", minimal_resolution_moving           )
    
    print(                                                                                            )
    print( " The time refinement starts from AMR grid level = ", input_data.refinement_level+1        )
    print( " The time interval in each step for coarest AMR grid during time evaluation = ", TimeStep )
    print(                                                                                            )

    print( " This simulation uses the Patch AMR grid structure "  )
    print(                                                        )

    print( "------------------------------------------------------------------------------------------" ) 
    
    ## file output
    
    filepath = os.path.join( File_directory, "AMSS_NCKU_resolution" )
    file0    = open(filepath, 'w')
    
    print(                                                                                              file=file0 )
    print( " Printing the basic parameter and setting in the AMSS-NCKU simulation ",                    file=file0 )
    print(                                                                                              file=file0 )
    print( " The number of MPI processes in the AMSS-NCKU simulation = ", input_data.MPI_processes,     file=file0 ) 
    print(                                                                                              file=file0 )
    print( " The form of computational equation  = ",            input_data.Equation_Class,             file=file0 )
    print( " The initial data in this simulation = ",            input_data.Initial_Data_Method,        file=file0 )
    print(                                                                                              file=file0 )
    print( " Starting evolution time   = ",                      input_data.Start_Evolution_Time,       file=file0 ) 
    print( " Final evolution time      = ",                      input_data.Final_Evolution_Time,       file=file0 ) 
    print( " Maximal iteration number  = ",                      input_data.Evolution_Step_Number,      file=file0 )
    print( " Courant factor            = ",                      input_data.Courant_Factor,             file=file0 )
    print( " Strength of dissipation   = ",                      input_data.Dissipation,                file=file0 )
    print( " Symmetry of system        = ",                      input_data.Symmetry,                   file=file0 )
    print( " The Runge-Kutta scheme in the time evolution   = ", input_data.Time_Evolution_Method,      file=file0 ) 
    print( " The finite-difference scheme in the simulation = ", input_data.Finite_Diffenence_Method,   file=file0 )
    print(                                                                                              file=file0 )
    print( " The static AMR grid type = ",                       input_data.static_grid_type,           file=file0 )
    print( " The moving AMR grid type = ",                       input_data.moving_grid_type,           file=file0 )
    print(                                                                                              file=file0 )
    print( " The number of static AMR grid levels = ",           static_grid_level,                     file=file0 )      
    print( " The number of moving AMR grid levels = ",           moving_grid_level,                     file=file0 )      
    print( " The number of total  AMR grid levels = ",           total_grid_level,                      file=file0 )
    print(                                                                                              file=file0 )
    print( " The grid number of each static AMR grid level = ",  static_grid_number,                    file=file0 )
    print( " The grid number of each moving AMR grid level = ",  moving_grid_number,                    file=file0 )
    print(                                                                                              file=file0 )
    print( " The scale for largest  static AMR grid in X direction = ", maximal_domain_size_static_x,   file=file0 )
    print( " The scale for largest  static AMR grid in Y direction = ", maximal_domain_size_static_y,   file=file0 )
    print( " The scale for largest  static AMR grid in Z direction = ", maximal_domain_size_static_z,   file=file0 )
    print( " The scale for smallest static AMR grid in X direction = ", minimal_domain_size_static_x,   file=file0 )
    print( " The scale for smallest static AMR grid in Y direction = ", minimal_domain_size_static_y,   file=file0 )
    print( " The scale for smallest static AMR grid in Z direction = ", minimal_domain_size_static_z,   file=file0 )
    print(                                                                                                         )
    
    if ( input_data.moving_grid_level > 0):
        print( " The scale for largest  moving AMR grid = ",     maximal_domain_size_moving,            file=file0 )
        print( " The scale for smallest moving AMR grid = ",     minimal_domain_size_moving,            file=file0 )

    print(                                                                                              file=file0 )
    print( " The coarest resolution for static AMR grid = ",     maximal_resolution_static,             file=file0 )
    print( " The finest  resolution for static AMR grid = ",     minimal_resolution_static,             file=file0 )
    
    if ( input_data.moving_grid_level > 0):
        print( " The coarest resolution for moving AMR grid = ", maximal_resolution_moving,             file=file0 )
        print( " The finest  resolution for moving AMR grid = ", minimal_resolution_moving,             file=file0 )
    
    print(                                                                                              file=file0 )
    print( " The time refinement starts from AMR grid level = ", input_data.refinement_level+1,         file=file0 )
    print( " The time interval in each step for coarest AMR grid during time evaluation = ", TimeStep,  file=file0 )
    print(                                                                                              file=file0 )

    print( " This simulation uses the Patch AMR grid structure ", file=file0 )
    print(                                                        file=file0 )

##################################################################
    

##################################################################

# output the puncture information

def print_puncture_information():
    position         = numpy.zeros( (input_data.puncture_number, 3) )         ## initialize the position of each black hole
    momentum         = numpy.zeros( (input_data.puncture_number, 3) )         ## initialize the momentum of each black hole
    angular_momentum = numpy.zeros( (input_data.puncture_number, 3) )         ## initialize the angular momentum of each black hole
    parameter        = numpy.zeros( (input_data.puncture_number, 3) )         ## initialize the parameter of each black hole

    print("------------------------------------------------------------------------------------------") 
    print(                                       )   
    print( " Printing the puncture information " )
    print(                                       )   

    for i in range(input_data.puncture_number):
        
        ## set the parameter of each black hole
        parameter[i] = input_data.parameter_BH[i]
        position[i]  = input_data.position_BH[i] 
        momentum[i]  = input_data.momentum_BH[i]
        ## angular_momentum[i] = input_data.angular_momentum_BH[i]

        ## setting the angular momentum of each black hole according to the input file
        if ( input_data.Symmetry == "equatorial-symmetry" ):
            angular_momentum[i] = [ 0.0, 0.0, (input_data.parameter_BH[i,0]**2) * input_data.parameter_BH[i,2] ]
        elif ( input_data.Symmetry == "no-symmetry" ):
            angular_momentum[i] = (input_data.parameter_BH[i,0]**2) * input_data.dimensionless_spin_BH[i] 

        print( f" The information for {i+1} puncture " ) 
        print( f" Mass({i+1}) = {parameter[i,0]       :>10.6f},  Charge({i+1}) = {parameter[i,1]       :>10.6f},  a({i+1})  = {parameter[i,2]       :>10.6f}" )
        print( f" X({i+1})    = {position[i,0]        :>10.6f},  Y({i+1})      = {position[i,1]        :>10.6f},  Z({i+1})  = {position[i,2]        :>10.6f}" )
        print( f" Px({i+1})   = {momentum[i,0]        :>10.6f},  Py({i+1})     = {momentum[i,1]        :>10.6f},  Pz({i+1}) = {momentum[i,2]        :>10.6f}" )
        print( f" Jx({i+1})   = {angular_momentum[i,0]:>10.6f},  Jy({i+1})     = {angular_momentum[i,1]:>10.6f},  Jz({i+1}) = {angular_momentum[i,2]:>10.6f}" )
        print()   

    print("------------------------------------------------------------------------------------------") 
    
##################################################################

## Generate the input parfile for AMSS-NCKU program

def generate_AMSSNCKU_input(): 

    file1 = open( os.path.join(input_data.File_directory, "AMSS-NCKU.input"), "w" ) 
    ## file1 = open( "AMSS-NCKU.input", "w" )  

    ## output ABE related settings

    print( file=file1 )
    print( "ABE::checkrun  =  0",                                          file=file1 )
    print( "ABE::checkfile =  bssn.chk",                                   file=file1 )
    print( "ABE::Steps     = ",         input_data.Evolution_Step_Number,  file=file1 )
    print( "ABE::StartTime = ",         input_data.Start_Evolution_Time,   file=file1 )
    print( "ABE::TotalTime = ",         input_data.Final_Evolution_Time,   file=file1 )
    print( "ABE::DumpTime  = ",         input_data.Dump_Time,              file=file1 )
    print( "ABE::d2DumpTime   = ",      input_data.D2_Dump_Time,           file=file1 )
    print( "ABE::CheckTime    = ",      input_data.Check_Time,             file=file1 )
    print( "ABE::AnalysisTime = ",      input_data.Analysis_Time,          file=file1 )
    print( "ABE::Courant      = ",      input_data.Courant_Factor,         file=file1 )

    print( "ABE::Symmetry     = 1 ",                                       file=file1 )

    print( "ABE::small dissipation = ",        input_data.Dissipation,     file=file1 )
    print( "ABE::big dissipation   = ",        input_data.Dissipation,     file=file1 )
    print( "ABE::shell dissipation = ",        input_data.Dissipation,     file=file1 )
    print( "ABE::Analysis Level    = ",        input_data.analysis_level,  file=file1 )
    print( "ABE::Max mode l        = ",        input_data.GW_L_max,        file=file1 )
    print( "ABE::detector number   = ",        input_data.Detector_Number, file=file1 )
    print( "ABE::farest detector position = ", input_data.Detector_Rmax,   file=file1 ) 
    print( f"ABE::detector distance = { (input_data.Detector_Rmax-input_data.Detector_Rmin) / (input_data.Detector_Number-1) }", \
           file=file1 )  
    print( "ABE::output dir = ", input_data.Output_directory,              file=file1 ) 
    
    print( "ABE::ID Type    = 0",  file=file1 )
        
    print( file=file1 )
    
    ## output AHF related settings
    
    print( "AHF::AHfindevery = ", input_data.AHF_Find_Every, file=file1 ) 
    print( "AHF::AHdumptime  = ", input_data.AHF_Dump_Time,  file=file1 ) 
    print(                                                   file=file1 )

    ## output other settings
    print(                                                                                              file=file1 )
    print( "SurfaceIntegral::number of points for quarter sphere = ", input_data.quarter_sphere_number, file=file1 )
    print(                                                                                              file=file1 )
    
    print(                                                       file=file1 )
         
    file1.close()

    return file1
    
##################################################################

        
