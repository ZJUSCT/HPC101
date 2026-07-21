
##################################################################
##
## Generate input file for the AMSS-NCKU TwoPuncture routine
## Author: Xiaoqu
## 2024/11/27
## Modified: 2025/01/21
##
##################################################################


import numpy
import os 
import AMSS_NCKU_Input as input_data          ## import program input file
import math

##################################################################

## Import binary black hole coordinates

## Read initial positions and momenta directly from AMSS_NCKU_Input.py and
## rescale the total binary mass to M = 1 for TwoPuncture input.
## (The lab fixes puncture_data_set to "Manually"; the "Automatically-BBH"
## post-Newtonian path and its BBH_orbit_parameter dependency are removed.)

mass_ratio_Q = input_data.parameter_BH[0,0] / input_data.parameter_BH[1,0]

if ( mass_ratio_Q < 1.0 ):
    print( " mass_ratio setting is wrong, please reset!!!" )
    print( " set the first black hole to be the larger mass!!!" )

BBH_M1 = mass_ratio_Q / ( 1.0 + mass_ratio_Q )
BBH_M2 = 1.0          / ( 1.0 + mass_ratio_Q )

parameter_BH = input_data.parameter_BH
position_BH  = input_data.position_BH
momentum_BH  = input_data.momentum_BH

## Compute binary separation and load eccentricity
distance = math.sqrt( (position_BH[0,0]-position_BH[1,0])**2 + (position_BH[0,1]-position_BH[1,1])**2 + (position_BH[0,2]-position_BH[1,2])**2 )
e0       = input_data.e0

## Set spin angular momentum input for TwoPuncture
## Note: these are dimensional angular momenta (not dimensionless); multiply
## by the square of the mass scale. Here masses are scaled so total M=1.

angular_momentum_BH = numpy.zeros( (input_data.puncture_number, 3) )

for i in range(input_data.puncture_number):
    if i == 0:
        angular_momentum_BH[i] = [0.0, 0.0, (BBH_M1**2) * parameter_BH[i,2]]
    elif i == 1:
        angular_momentum_BH[i] = [0.0, 0.0, (BBH_M2**2) * parameter_BH[i,2]]
    else:
        angular_momentum_BH[i] = [0.0, 0.0, (parameter_BH[i,0]**2) * parameter_BH[i,2]]


##################################################################

## Write the above binary data into the AMSS-NCKU TwoPuncture input file
    
def generate_AMSSNCKU_TwoPuncture_input(): 

    file1 = open( os.path.join(input_data.File_directory, "AMSS-NCKU-TwoPuncture.input"), "w") 

    print( "#  -----0-----> y",                           file=file1 )
    print( "#   -      +      use Brugmann's convention", file=file1 )
    print( "ABE::mp        = -1.0",                       file=file1 )   ## use negative values so the code solves for bare masses automatically
    print( "ABE::mm        = -1.0",                       file=file1 )
    print( "# b            =  D/2",                       file=file1 )
    print( "ABE::b         = ", ( distance / 2.0 ),       file=file1 )
    print( "ABE::P_plusx   = ", momentum_BH[0,0],         file=file1 )
    print( "ABE::P_plusy   = ", momentum_BH[0,1],         file=file1 )
    print( "ABE::P_plusz   = ", momentum_BH[0,2],         file=file1 )
    print( "ABE::P_minusx  = ", momentum_BH[1,0],         file=file1 )
    print( "ABE::P_minusy  = ", momentum_BH[1,1],         file=file1 )
    print( "ABE::P_minusz  = ", momentum_BH[1,2],         file=file1 )
    print( "ABE::S_plusx   = ", angular_momentum_BH[0,0], file=file1 )
    print( "ABE::S_plusy   = ", angular_momentum_BH[0,1], file=file1 )
    print( "ABE::S_plusz   = ", angular_momentum_BH[0,2], file=file1 )
    print( "ABE::S_minusx  = ", angular_momentum_BH[1,0], file=file1 )
    print( "ABE::S_minusy  = ", angular_momentum_BH[1,1], file=file1 )
    print( "ABE::S_minusz  = ", angular_momentum_BH[1,2], file=file1 )
    print( "ABE::Mp        = ", BBH_M1,                   file=file1 )
    print( "ABE::Mm        = ", BBH_M2,                   file=file1 )
    print( "ABE::admtol    =  1.e-8",                     file=file1 )
    print( "ABE::Newtontol =  5.e-12",                    file=file1 )
    print( "ABE::nA        =  50",                        file=file1 )
    print( "ABE::nB        =  50",                        file=file1 )
    print( "ABE::nphi      =  26",                        file=file1 )
    print( "ABE::Newtonmaxit =  50",                      file=file1 )
    
    file1.close()

    return file1
    
##################################################################
    
    
