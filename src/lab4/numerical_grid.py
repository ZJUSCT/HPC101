
#################################################
##
## This file includes the numerical grid needed in numerical relativity
## author: xiaoqu
## 2024/03/20
## 2025/09/14 modified
##
#################################################

import numpy                              
import matplotlib.pyplot as plt           
import os                                 

import AMSS_NCKU_Input   as input_data    
## import print_information

#################################################

# set the information of black hole puncture

puncture = numpy.zeros( (input_data.puncture_number,3) )      

print(                                   )
print( " Setting Puncture's position and momentum " )
print(                                              )

#################################################

## setting puncture position

## read resetted puncture position if puncture_data_set is Automatically-BBH
 
if (input_data.puncture_data_set == "Automatically-BBH" ):

    import generate_TwoPuncture_input
    
    for i in range(input_data.puncture_number):
        if (i<=1):
            puncture[i] = generate_TwoPuncture_input.position_BH[i]
        else:
            puncture[i] = input_data.position_BH[i]
    
## read in puncture position directly if puncture_data_set is Manually 
   
elif (input_data.puncture_data_set == "Manually" ):

    puncture = input_data.position_BH 
    
else: 
   
   print(                                          )
   print( " Found Error in setting Puncture's position and momentum !!! " )
   print(                                                                 )

#################################################

## output grid information

print(                                     )   
print( " Wirte Down The Grid Information " )
print(                                     )   
print( " Number of Total Grid Level = ",          input_data.grid_level        )      
print( " Number of Static Grid Level = ",         input_data.static_grid_level )     
print( " Number of Moving Grid Level = ",         input_data.moving_grid_level )    
## print( " Number of Points in Each Grid Level = ", input_data.grid_number    )      
print(                                                                         )

#################################################

print(                     )
print( " Setting the demanded numerical grid " )
print(                                         )

#################################################

## initialize the grid information

## initialize the grid min and max points and grid number

Grid_X_Min = numpy.zeros( (input_data.grid_level) )    
Grid_X_Max = numpy.zeros( (input_data.grid_level) )    
Grid_Y_Min = numpy.zeros( (input_data.grid_level) )    
Grid_Y_Max = numpy.zeros( (input_data.grid_level) )    
Grid_Z_Min = numpy.zeros( (input_data.grid_level) )    
Grid_Z_Max = numpy.zeros( (input_data.grid_level) )    

Grid_Resolution = numpy.zeros( input_data.grid_level ) 

largest_box_X_Max = input_data.largest_box_xyz_max[0] 
largest_box_Y_Max = input_data.largest_box_xyz_max[1]
largest_box_Z_Max = input_data.largest_box_xyz_max[2]
largest_box_X_Min = input_data.largest_box_xyz_min[0] 
largest_box_Y_Min = input_data.largest_box_xyz_min[1]
largest_box_Z_Min = input_data.largest_box_xyz_min[2]

# define integer number as the grid number in each direction of static grid at each level
static_grid_number_x = input_data.static_grid_number 
static_grid_number_y = int( (largest_box_Y_Max - largest_box_Y_Min) * ( static_grid_number_x / (largest_box_X_Max-largest_box_X_Min) ) )
static_grid_number_z = int( (largest_box_Z_Max - largest_box_Z_Min) * ( static_grid_number_x / (largest_box_X_Max-largest_box_X_Min) ) )
    
# define integer array as the grid number in each direction of moving grid at each level
moving_grid_number   = input_data.moving_grid_number   

#################################################

## initialize static grids

# adjust the grid number in each direction to be even number
# print(static_grid_number_x % 2)
if ( (static_grid_number_x % 2) != 0) :
    static_grid_number_x = static_grid_number_x + 1
if ( (static_grid_number_y % 2) != 0) :
    static_grid_number_y = static_grid_number_y + 1
if ( (static_grid_number_z % 2) != 0) :
    static_grid_number_z = static_grid_number_z + 1
# require the grid number in each direction to be the multiple of 4 for better alignment between moving and static grids

if ( (static_grid_number_x % 4) != 0) :
    static_grid_number_x = static_grid_number_x + 2
if ( (static_grid_number_y % 4) != 0) :
    static_grid_number_y = static_grid_number_y + 2
if ( (static_grid_number_z % 4) != 0) :
    static_grid_number_z = static_grid_number_z + 2
'''
# require the grid number in each direction to be the multiple of 8 for better alignment between moving and static grids
if ( (static_grid_number_x % 8) != 0) :
    static_grid_number_x = static_grid_number_x + 4
if ( (static_grid_number_y % 8) != 0) :
    static_grid_number_y = static_grid_number_y + 4
if ( (static_grid_number_z % 8) != 0) :
    static_grid_number_z = static_grid_number_z + 4
'''

## Define real arrays, dimension grid_number * static_grid_level, as the X Y Z coordinates of each level of static grid
Static_Grid_X = numpy.zeros( (input_data.static_grid_level, static_grid_number_x+1) )   
Static_Grid_Y = numpy.zeros( (input_data.static_grid_level, static_grid_number_y+1) )   
Static_Grid_Z = numpy.zeros( (input_data.static_grid_level, static_grid_number_z+1) )  

#################################################

## initialize moving grids

##  define real arrays, dimension grid_number * puncture_number * moving_grid_level, as the X Y Z coordinates of each level of moving grid
Moving_Grid_X = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number, input_data.moving_grid_number+1) )
Moving_Grid_Y = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number, input_data.moving_grid_number+1) ) 
Moving_Grid_Z = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number, input_data.moving_grid_number+1) )

#################################################

## initialize the min and max grid points of moving grids

Moving_Grid_X_Min = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number) ) 
Moving_Grid_X_Max = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number) )                
Moving_Grid_Y_Min = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number) )                
Moving_Grid_Y_Max = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number) )                
Moving_Grid_Z_Min = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number) )
Moving_Grid_Z_Max = numpy.zeros( (input_data.moving_grid_level, input_data.puncture_number) )          

#################################################

## set the grid resolution of each level

for i in range(input_data.static_grid_level) :
    if i==0:
        Grid_Resolution[i] = ( largest_box_X_Max - largest_box_X_Min ) / static_grid_number_x
    else:
        Grid_Resolution[i] = Grid_Resolution[i-1] / input_data.devide_factor
    
for j in range(input_data.moving_grid_level) : 
    i = j + input_data.static_grid_level
    Grid_Resolution[i] = Grid_Resolution[i-1] / input_data.devide_factor

#################################################

## according to the input file, set the minimum and maximum grid points of each level static patch grid

## set the maximum and minimum grid points of the first static patch grid
Grid_X_Min[0] = largest_box_X_Min 
Grid_X_Max[0] = largest_box_X_Max
Grid_Y_Min[0] = largest_box_Y_Min  
Grid_Y_Max[0] = largest_box_Y_Max
Grid_Z_Min[0] = largest_box_Z_Min
Grid_Z_Max[0] = largest_box_Z_Max
## recalculate Grid_Y_Max[0] to ensure the same resolution in xyz directions
Grid_Y_Max[0] = Grid_Y_Min[0] + Grid_Resolution[0] * static_grid_number_y
Grid_Z_Max[0] = Grid_Z_Min[0] + Grid_Resolution[0] * static_grid_number_z
## adjust grid boundary according to the symmetry condition
if ( input_data.Symmetry == "equatorial-symmetry" ):
    Grid_Z_Min[0] = - Grid_Resolution[0] * static_grid_number_z / 2
    Grid_Z_Max[0] = + Grid_Resolution[0] * static_grid_number_z / 2
elif ( input_data.Symmetry == "octant-symmetry" ):
    Grid_X_Min[0] = - Grid_Resolution[0] * static_grid_number_x / 2
    Grid_X_Max[0] = + Grid_Resolution[0] * static_grid_number_x / 2
    Grid_Y_Min[0] = - Grid_Resolution[0] * static_grid_number_y / 2
    Grid_Y_Max[0] = + Grid_Resolution[0] * static_grid_number_y / 2
    Grid_Z_Min[0] = - Grid_Resolution[0] * static_grid_number_z / 2
    Grid_Z_Max[0] = + Grid_Resolution[0] * static_grid_number_z / 2

## print( " Grid_Y_Max[0] = ", Grid_Y_Max[0] )


print( " adjusting the static gird points, making the original point (0,0,0) to the static gird points " )
print(                                                                                                   )

## set maximum and minimum grid points of other static patch grids
for i in range(input_data.static_grid_level-1) :
    ## if the coordinate origin is not on the outermost static grid, adjust the outermost static grid to make the origin on the grid
    if i==0:
        for nn in range(static_grid_number_x):
            if (Grid_X_Min[i] + nn*Grid_Resolution[i]) < 0.0 < (Grid_X_Min[i] + (nn+1)*Grid_Resolution[i]):
                print( " before adjust: Grid X_min = ", Grid_X_Min[i] )
                print( " before adjust: Grid X_max = ", Grid_X_Max[i] )
                grid_adjust   = Grid_X_Min[i] + (nn+1)*Grid_Resolution[i]
                Grid_X_Min[i] = Grid_X_Min[i] - grid_adjust
                Grid_X_Max[i] = Grid_X_Max[i] - grid_adjust
                print( " after adjust: Grid X_min = ", Grid_X_Min[i] )
                print( " after adjust: Grid X_max = ", Grid_X_Max[i] )
        for nn in range(static_grid_number_y):
            if (Grid_Y_Min[i] + nn*Grid_Resolution[i]) < 0.0 < (Grid_Y_Min[i] + (nn+1)*Grid_Resolution[i]):
                print( " before adjust: Grid Y_min = ", Grid_Y_Min[i] )
                print( " before adjust: Grid Y_max = ", Grid_Y_Max[i] )
                grid_adjust   = Grid_Y_Min[i] + (nn+1)*Grid_Resolution[i]
                Grid_Y_Min[i] = Grid_Y_Min[i] - grid_adjust
                Grid_Y_Max[i] = Grid_Y_Max[i] - grid_adjust
                print( " after adjust: Grid Y_min = ", Grid_Y_Min[i] )
                print( " after adjust: Grid Y_max = ", Grid_Y_Max[i] )
        for nn in range(static_grid_number_z):
            if (Grid_Z_Min[i] + nn*Grid_Resolution[i]) < 0.0 < (Grid_Z_Min[i] + (nn+1)*Grid_Resolution[i]):
                print( " before adjust: Grid Z_min = ", Grid_Z_Min[i] )
                print( " before adjust: Grid Z_max = ", Grid_Z_Max[i] )
                grid_adjust   = Grid_X_Min[i] + (nn+1)*Grid_Resolution[i]
                Grid_Z_Min[i] = Grid_Z_Min[i] - grid_adjust
                Grid_Z_Max[i] = Grid_Z_Max[i] - grid_adjust
                print( " after adjust: Grid Z_min = ", Grid_Z_Min[i] )
                print( " after adjust: Grid Z_max = ", Grid_Z_Max[i] )
    ## the maximum and minimum grid points equal to the previous grid level divided by devide_factor
    Grid_X_Min[i+1] = Grid_X_Min[i] / input_data.devide_factor    
    Grid_X_Max[i+1] = Grid_X_Max[i] / input_data.devide_factor    
    Grid_Y_Min[i+1] = Grid_Y_Min[i] / input_data.devide_factor
    Grid_Y_Max[i+1] = Grid_Y_Max[i] / input_data.devide_factor
    Grid_Z_Min[i+1] = Grid_Z_Min[i] / input_data.devide_factor
    Grid_Z_Max[i+1] = Grid_Z_Max[i] / input_data.devide_factor
    

## add adjust factor to ensure the moving grid boundary aligns with static grid boundary
adjust_factor = input_data.moving_grid_number / input_data.static_grid_number

## set maximum and minimum grid points of the first moving patch grid
i = input_data.static_grid_level 
if (i < input_data.grid_level):
    Grid_X_Min[i] = ( Grid_X_Min[i-1] / input_data.devide_factor ) * adjust_factor
    Grid_X_Max[i] = - Grid_X_Min[i]
    # Grid_X_Max[i] = ( Grid_X_Max[i-1] / input_data.devide_factor ) * adjust_factor   
    ## original setting
    # Grid_Y_Min[i] = ( Grid_Y_Min[i-1] / input_data.devide_factor ) * adjust_factor
    # Grid_Y_Max[i] = ( Grid_Y_Max[i-1] / input_data.devide_factor ) * adjust_factor
    # Grid_Z_Min[i] = ( Grid_Z_Min[i-1] / input_data.devide_factor ) * adjust_factor
    # Grid_Z_Max[i] = ( Grid_Z_Max[i-1] / input_data.devide_factor ) * adjust_factor
    ## current setting to ensure moving grid is cubic
    Grid_Y_Min[i] = Grid_X_Min[i]
    Grid_Y_Max[i] = Grid_X_Max[i]
    Grid_Z_Min[i] = Grid_X_Min[i]
    Grid_Z_Max[i] = Grid_X_Max[i]

    # print( " Grid_X_Max[i] = ", Grid_X_Max[i] )
    # print( " Grid_Y_Max[i] = ", Grid_Y_Max[i] )

## set maximum and minimum grid points of moving patch grids
for j in range(input_data.moving_grid_level-1) :
    k = input_data.static_grid_level + j
    Grid_X_Min[k+1] = Grid_X_Min[k] / input_data.devide_factor    
    Grid_X_Max[k+1] = Grid_X_Max[k] / input_data.devide_factor    
    Grid_Y_Min[k+1] = Grid_Y_Min[k] / input_data.devide_factor
    Grid_Y_Max[k+1] = Grid_Y_Max[k] / input_data.devide_factor
    Grid_Z_Min[k+1] = Grid_Z_Min[k] / input_data.devide_factor
    Grid_Z_Max[k+1] = Grid_Z_Max[k] / input_data.devide_factor
    
## set maximum and minimum grid points of the outermost  shell patch grid

Shell_R_Resolution = Grid_Resolution[0]
Shell_R_Min        = largest_box_X_Max
Shell_R_Max        = Shell_R_Min + Grid_Resolution[0] * input_data.shell_grid_number[2]

#################################################


#################################################

## set grid points position of each level

#################################################

## setting static grid points position

## linear grid points

if input_data.static_grid_type == 'Linear' : 

    for i in range(input_data.static_grid_level):
        Static_Grid_X[i] = numpy.linspace( Grid_X_Min[i], Grid_X_Max[i], static_grid_number_x+1 ) 
        Static_Grid_Y[i] = numpy.linspace( Grid_Y_Min[i], Grid_Y_Max[i], static_grid_number_y+1 ) 
        Static_Grid_Z[i] = numpy.linspace( Grid_Z_Min[i], Grid_Z_Max[i], static_grid_number_z+1 )
     # use numpy to set linear grid points, parameters are Rmin, Rmax, Rnum
     # Note that if it is linear grid points, the maximum grid point coordinate is GridMax; if it is logarithmic grid points, the maximum grid point coordinate is e^{GridMax}

else:
    print(                                                       )
    print( " Static Grid Error: Grid Type is Undifined !!!!!!! " )
    print(                                                       )

#################################################
  
## set moving grid points position 

print(                                                                                                            )
print( " adjusting the moving gird points, ensuring the alliance of moving grids points and static grids points " )
print(                                                                                                            )

## adjust puncture position to ensure moving grid boundary aligns with static grid boundary
adjust_puncture = numpy.zeros( (input_data.puncture_number, 3) )

## linear grid points
  
if ( input_data.moving_grid_type == "Linear" ): 
    
    ## circle over moving grid level
    for j in range(input_data.moving_grid_level) : 

        i = j + input_data.static_grid_level
        
        ## circle over puncture number
        for k in range(input_data.puncture_number) :
            
            ## adjust the puncture position
            if j==0 :
            
                level0 = input_data.static_grid_level - 1  ## add new variable to avoid long code
                
                for m in range(static_grid_number_x) : 
                    if ( Static_Grid_X[level0, m] <= puncture[k,0] <= Static_Grid_X[level0, m+1] ):
                        if ( abs( puncture[k,0] - Static_Grid_X[level0, m] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,0] = Static_Grid_X[level0, m]
                        elif ( abs( puncture[k,0] - Static_Grid_X[level0, m+1 ] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,0] = Static_Grid_X[level0, m+1 ]
                        else:
                            adjust_puncture[k,0] = ( Static_Grid_X[level0, m] + Static_Grid_X[level0, m+1] ) / 2.0
                
                for m in range(static_grid_number_y) :
                    if ( Static_Grid_Y[level0, m] <= puncture[k,1] <= Static_Grid_Y[level0, m+1] ):
                        if ( abs( puncture[k,1] - Static_Grid_Y[level0, m] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,1] = Static_Grid_Y[level0, m]
                        elif ( abs( puncture[k,1] - Static_Grid_Y[level0, m+1] ) < ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,1] = Static_Grid_Y[level0, m+1]
                        else:
                            adjust_puncture[k,1] = ( Static_Grid_Y[level0, m] + Static_Grid_Y[level0, m+1] ) / 2.0
                
                for m in range(static_grid_number_z) :
                    if ( Static_Grid_Z[level0, m] <= puncture[k,2] <= Static_Grid_Z[level0, m+1] ):
                        if ( abs( puncture[k,2] - Static_Grid_Z[level0, m] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,2] = Static_Grid_Z[level0, m]
                        elif ( abs( puncture[k,2] - Static_Grid_Z[level0, m+1] ) < ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,2] = Static_Grid_Z[level0, m+1]
                        else:
                            adjust_puncture[k,2] = ( Static_Grid_Z[level0, m] + Static_Grid_Z[level0, m+1] ) / 2.0


            elif j>0 :
                for m in range(moving_grid_number) :
                
                    if ( Moving_Grid_X[j-1,k,m] <= puncture[k,0] <= Moving_Grid_X[j-1,k,m+1] ):
                        if ( abs( puncture[k,0] - Moving_Grid_X[j-1,k,m] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,0] = Moving_Grid_X[j-1,k,m]
                        elif ( abs( puncture[k,0] - Moving_Grid_X[j-1,k,m+1] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,0] = Moving_Grid_X[j-1,k,m+1]
                        else:
                            adjust_puncture[k,0] = ( Moving_Grid_X[j-1,k,m] + Moving_Grid_X[j-1,k,m+1] ) / 2.0
                
                    if ( Moving_Grid_Y[j-1,k,m] <= puncture[k,1] <= Moving_Grid_Y[j-1,k,m+1] ):
                        if ( abs( puncture[k,1] - Moving_Grid_Y[j-1,k,m] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,1] = Moving_Grid_Y[j-1,k,m]
                        elif ( abs( puncture[k,1] - Moving_Grid_Y[j-1,k,m+1] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,1] = Moving_Grid_Y[j-1,k,m+1]
                        else:
                            adjust_puncture[k,1] = ( Moving_Grid_Y[j-1,k,m] + Moving_Grid_Y[j-1,k,m+1] ) / 2.0

                    if ( Moving_Grid_Z[j-1,k,m] <= puncture[k,2] <= Moving_Grid_Z[j-1,k,m+1] ):
                        if ( abs( puncture[k,2] - Moving_Grid_Z[j-1,k,m] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,2] = Moving_Grid_Z[j-1,k,m]
                        elif ( abs( puncture[k,2] - Moving_Grid_Z[j-1,k,m+1] )  <  ( Grid_Resolution[i]/2.0 ) ):
                            adjust_puncture[k,2] = Moving_Grid_Z[j-1,k,m+1]
                        else:
                            adjust_puncture[k,2] = ( Moving_Grid_Z[j-1,k,m] + Moving_Grid_Z[j-1,k,m+1] ) / 2.0

            else:
                print( " Adjusting puncture position to compatable with coaser grid !  Error !!! " )
            ## adjusted puncture position is done
            
            ## to avoid error in C++ input reading 
            ## small number as 1e-10 set to 0.00 with 2 decimal places
            if ( abs(adjust_puncture[k,0]) < 1e-10 ):
                adjust_puncture[k,0] = 0.00
                # adjust_puncture[k,0] = f"{ adjust_puncture[k,0]:.2f }
            if ( abs(adjust_puncture[k,1]) < 1e-10 ):
                adjust_puncture[k,1] = 0.00
                # adjust_puncture[k,1] = f"{ adjust_puncture[k,1]:.2f }
            if ( abs(adjust_puncture[k,2]) < 1e-10 ):
                adjust_puncture[k,2] = 0.00
                # adjust_puncture[k,2] = f"{ adjust_puncture[k,2]:.2f }
            
            # the j-th moving grid's XYZ min (or max) = the i-th grid's XYZ min (or max) + the k-th puncture's XYZ position
            Moving_Grid_X_Min[j,k] = adjust_puncture[k,0] + Grid_X_Min[i]  
            Moving_Grid_X_Max[j,k] = adjust_puncture[k,0] + Grid_X_Max[i]  
            Moving_Grid_Y_Min[j,k] = adjust_puncture[k,1] + Grid_Y_Min[i]  
            Moving_Grid_Y_Max[j,k] = adjust_puncture[k,1] + Grid_Y_Max[i]  
            Moving_Grid_Z_Min[j,k] = adjust_puncture[k,2] + Grid_Z_Min[i]  
            Moving_Grid_Z_Max[j,k] = adjust_puncture[k,2] + Grid_Z_Max[i]
            
            ## to avoid error in C++ input reading 
            ## small number as 1e-10 set to 0.00 with 2 decimal places
            if ( abs(Moving_Grid_X_Min[j,k]) < 1e-10 ):
                Moving_Grid_X_Min[j,k] = 0.00 
            if ( abs(Moving_Grid_X_Max[j,k]) < 1e-10 ):
                Moving_Grid_X_Max[j,k] = 0.00 
                
            if ( abs(Moving_Grid_Y_Min[j,k]) < 1e-10 ):
                Moving_Grid_Y_Min[j,k] = 0.00 
            if ( abs(Moving_Grid_Y_Max[j,k]) < 1e-10 ):
                Moving_Grid_Y_Max[j,k] = 0.00 
                
            if ( abs(Moving_Grid_Z_Min[j,k]) < 1e-10 ):
                Moving_Grid_Z_Min[j,k] = 0.00 
            if ( abs(Moving_Grid_Z_Max[j,k]) < 1e-10 ):
                Moving_Grid_Z_Max[j,k] = 0.00 
            
            print( f" adjust_puncture[{i},{k},0] = { adjust_puncture[k,0] } " )
            print( f" adjust_puncture[{i},{k},1] = { adjust_puncture[k,1] } " )
            print( f" adjust_puncture[{i},{k},2] = { adjust_puncture[k,2] } " )
            
            ## using numpy to set linear grid points, parameters are Rmin, Rmax, Rnum
            Moving_Grid_X[j,k] = numpy.linspace( Moving_Grid_X_Min[j,k], Moving_Grid_X_Max[j,k], moving_grid_number + 1 )  
            Moving_Grid_Y[j,k] = numpy.linspace( Moving_Grid_Y_Min[j,k], Moving_Grid_Y_Max[j,k], moving_grid_number + 1 )
            Moving_Grid_Z[j,k] = numpy.linspace( Moving_Grid_Z_Min[j,k], Moving_Grid_Z_Max[j,k], moving_grid_number + 1 )
        
else:
    print(                                                       )
    print( " Moving Grid Error: Grid Type is Undifined !!!!!!! " )
    print(                                                       )

print(                            )
print( " The moving grid puncture position adjustment is done " )
print(                            )

#################################################


#################################################

## this function plots the initial numerical grids
    
def plot_initial_grid():

## plot the final grids

    if (input_data.static_grid_level > 0):
        X0, Y0 = numpy.meshgrid( Static_Grid_X[0], Static_Grid_Y[0] )
        plt.plot( X0, Y0,                         
                  color='brown',  	          
                  marker='.',  	                  
                  linestyle='' )                  
              
    if (input_data.static_grid_level > 1):
        X1, Y1 = numpy.meshgrid( Static_Grid_X[1], Static_Grid_Y[1] )
        plt.plot( X1, Y1,                        
                  color='red',                    
                  marker='.',                     
                  linestyle='' )                 
                  
    if (input_data.static_grid_level > 2):
        X2, Y2 = numpy.meshgrid( Static_Grid_X[2], Static_Grid_Y[2] )
        plt.plot( X2, Y2,                        
                  color='orange',                 
                  marker='.',                     
                  linestyle='' )                  
                  
    if (input_data.static_grid_level > 3):
        X3, Y3 = numpy.meshgrid( Static_Grid_X[3], Static_Grid_Y[3] )
        plt.plot( X3, Y3,                         
                  color='yellow',                 
                  marker='.',                     
                  linestyle='' )                  
                  
    if (input_data.static_grid_level > 4):
        X4, Y4 = numpy.meshgrid( Static_Grid_X[4], Static_Grid_Y[4] )
        plt.plot( X4, Y4,                         
                  color='greenyellow',            
                  marker='.',                     
                  linestyle='' )                  
                  
    ## plot the moving grids

    if (input_data.moving_grid_level > 0):
        for k in range(input_data.puncture_number):
            Xk0, Yk0 = numpy.meshgrid( Moving_Grid_X[0,k], Moving_Grid_Y[0,k] )
            plt.plot( Xk0, Yk0,                       
                      color='cyan',  	              
                      marker='.',  	                  
                      linestyle='' )                  
 
    if (input_data.moving_grid_level > 1):
        for k in range(input_data.puncture_number):
            Xk1, Yk1 = numpy.meshgrid( Moving_Grid_X[1,k], Moving_Grid_Y[1,k] )
            plt.plot( Xk1, Yk1,                       
                      color='blue',                    
                      marker='.',                     
                      linestyle='' )                  
    
    if (input_data.moving_grid_level > 2):
        for k in range(input_data.puncture_number):
            Xk2, Yk2 = numpy.meshgrid( Moving_Grid_X[2,k], Moving_Grid_Y[2,k] )
            plt.plot( Xk2, Yk2,                       
                      color='navy',                   
                      marker='.',                     
                      linestyle='' )                  

    if (input_data.moving_grid_level > 3):
        for k in range(input_data.puncture_number):
            Xk3, Yk3 = numpy.meshgrid( Moving_Grid_X[3,k], Moving_Grid_Y[3,k] )
            plt.plot( Xk3, Yk3,                       
                      color='gray',                  
                      marker='.',                     
                      linestyle='' )                  
    
    if (input_data.moving_grid_level > 4):
        for k in range(input_data.puncture_number):
            Xk4, Yk4 = numpy.meshgrid( Moving_Grid_X[4,k], Moving_Grid_Y[4,k] )
            plt.plot( Xk4, Yk4,                       
                      color='black',                   
                      marker='.',                     
                      linestyle='' )                  
    
    plt.grid(True)
    ## plt.show()
    plt.savefig( os.path.join(input_data.File_directory, "Initial_Grid.jpeg") )
    plt.savefig( os.path.join(input_data.File_directory, "Initial_Grid.pdf")  )

#################################################


#################################################
    
## putting the grid setting into AMSS-NCKU input file
    
def append_AMSSNCKU_cgh_input(): 

    file1 = open( os.path.join(input_data.File_directory, "AMSS-NCKU.input"), "a")  
    # "a" for append mode

    ## output the setting of cgh

    print( file=file1 )
    print( "cgh::moving levels start from = ", input_data.static_grid_level, file=file1 )
    print( "cgh::levels = ",                   input_data.grid_level,        file=file1)

    ## output the setting of static grids

    for i in range(input_data.static_grid_level): 

        print( f"cgh::grids[{i}]       = 1",                                file=file1 )
        
        if ( input_data.Symmetry == "octant-symmetry" ):
            print( f"cgh::shape[{i}][0][0] = { static_grid_number_x//2 } ", file=file1 )
            print( f"cgh::shape[{i}][0][1] = { static_grid_number_y//2 } ", file=file1 )
        else:
            print( f"cgh::shape[{i}][0][0] = { static_grid_number_x } ",    file=file1 )
            print( f"cgh::shape[{i}][0][1] = { static_grid_number_y } ",    file=file1 )

        if ( input_data.Symmetry == "octant-symmetry" ):
            print( f"cgh::shape[{i}][0][2] = { static_grid_number_z//2 } ", file=file1 )
        elif ( input_data.Symmetry == "equatorial-symmetry" ):
            print( f"cgh::shape[{i}][0][2] = { static_grid_number_z//2 } ", file=file1 )
        elif ( input_data.Symmetry == "no-symmetry" ):
            print( f"cgh::shape[{i}][0][2] = { static_grid_number_z } ",    file=file1 )
        else:
            print( " Symmetry Setting Error " )

        if ( input_data.Symmetry == "octant-symmetry" ):
            print( f"cgh::bbox[{i}][0][0]  = 0.0 ",                       file=file1 )
            print( f"cgh::bbox[{i}][0][1]  = 0.0 ",                       file=file1 )
        else:
            print( f"cgh::bbox[{i}][0][0]  = { Grid_X_Min[i] } ",         file=file1 )
            print( f"cgh::bbox[{i}][0][1]  = { Grid_Y_Min[i] } ",         file=file1 )

        if ( input_data.Symmetry == "octant-symmetry" ):
            print( f"cgh::bbox[{i}][0][2]  = 0.0 ",                       file=file1 )
        elif ( input_data.Symmetry == "equatorial-symmetry" ):
            print( f"cgh::bbox[{i}][0][2]  = 0.0 ",                       file=file1 )
        elif ( input_data.Symmetry == "no-symmetry" ):
            print( f"cgh::bbox[{i}][0][2]  = { Grid_Z_Min[i] } ",         file=file1 )
        else:
            print( " Symmetry Setting Error " )

        print( f"cgh::bbox[{i}][0][3]  = { Grid_X_Max[i] } ",             file=file1 )
        print( f"cgh::bbox[{i}][0][4]  = { Grid_Y_Max[i] } ",             file=file1 )
        print( f"cgh::bbox[{i}][0][5]  = { Grid_Z_Max[i] } ",             file=file1 )

    ## output the setting of moving grids
       
    ## circle over moving grid levels
    
    for i in range(input_data.moving_grid_level):

        j = i + input_data.static_grid_level
        print( f"cgh::grids[{j}]       = { input_data.puncture_number }",        file=file1 )

        ## circle over puncture number
        for k in range(input_data.puncture_number):

            if ( input_data.Symmetry == "octant-symmetry" ): 
                print( f"cgh::shape[{j}][{k}][0] = { moving_grid_number//2 } ",  file=file1 )
                print( f"cgh::shape[{j}][{k}][1] = { moving_grid_number//2 } ",  file=file1 )
                print( f"cgh::shape[{j}][{k}][2] = { moving_grid_number//2 } ",  file=file1 )
            elif ( input_data.Symmetry == "equatorial-symmetry" ): 
                print( f"cgh::shape[{j}][{k}][0] = { moving_grid_number } ",     file=file1 )
                print( f"cgh::shape[{j}][{k}][1] = { moving_grid_number } ",     file=file1 )
                print( f"cgh::shape[{j}][{k}][2] = { moving_grid_number//2 } ",  file=file1 )
            elif ( input_data.Symmetry == "no-symmetry" ):
                print( f"cgh::shape[{j}][{k}][0] = { moving_grid_number } ",     file=file1 )
                print( f"cgh::shape[{j}][{k}][1] = { moving_grid_number } ",     file=file1 )
                print( f"cgh::shape[{j}][{k}][2] = { moving_grid_number } ",     file=file1 )   
            else:
                print( " Symmetry Setting Error" )

            print( f"cgh::bbox[{j}][{k}][0]  = { Moving_Grid_X_Min[i,k] } ",     file=file1 )
            print( f"cgh::bbox[{j}][{k}][1]  = { Moving_Grid_Y_Min[i,k] } ",     file=file1 )

            if ( input_data.Symmetry == "octant-symmetry" ):
                print( f"cgh::bbox[{j}][{k}][0]  = { max(0.0, Moving_Grid_X_Min[i,k]) } ", file=file1 )
                print( f"cgh::bbox[{j}][{k}][1]  = { max(0.0, Moving_Grid_Y_Min[i,k]) } ", file=file1 )
                print( f"cgh::bbox[{j}][{k}][2]  = { max(0.0, Moving_Grid_Z_Min[i,k]) } ", file=file1 )
            elif ( input_data.Symmetry == "equatorial-symmetry" ):
                print( f"cgh::bbox[{j}][{k}][0]  = { Moving_Grid_X_Min[i,k] } ",           file=file1 )
                print( f"cgh::bbox[{j}][{k}][1]  = { Moving_Grid_Y_Min[i,k] } ",           file=file1 )
                print( f"cgh::bbox[{j}][{k}][2]  = { max(0.0, Moving_Grid_Z_Min[i,k]) } ", file=file1 )
            elif ( input_data.Symmetry == "no-symmetry" ):
                print( f"cgh::bbox[{j}][{k}][0]  = { Moving_Grid_X_Min[i,k] } ", file=file1 )
                print( f"cgh::bbox[{j}][{k}][1]  = { Moving_Grid_Y_Min[i,k] } ", file=file1 )
                print( f"cgh::bbox[{j}][{k}][2]  = { Moving_Grid_Z_Min[i,k] } ", file=file1 )
            else:
                print( " Symmetry Setting Error" )

            print( f"cgh::bbox[{j}][{k}][3]  = { Moving_Grid_X_Max[i,k] } ",     file=file1 )
            print( f"cgh::bbox[{j}][{k}][4]  = { Moving_Grid_Y_Max[i,k] } ",     file=file1 )
            print( f"cgh::bbox[{j}][{k}][5]  = { Moving_Grid_Z_Max[i,k] } ",     file=file1 )

    ## output the setting of BSSN
    
    print(                                                                         file=file1 )
    print( "############ for shell-box coupling set this exactly to box boundary", file=file1 )
    print( "BSSN::Shell shape[0]   = ", input_data.shell_grid_number[0],           file=file1 )
    print( "BSSN::Shell shape[1]   = ", input_data.shell_grid_number[1],           file=file1 )
    print( "BSSN::Shell shape[2]   = ", input_data.shell_grid_number[2],           file=file1 )
    print( "BSSN::Shell R range[0] = ", Shell_R_Min,                               file=file1 )
    print( "BSSN::Shell R range[1] = ", Shell_R_Max,                               file=file1 )
    print(                                                                         file=file1 )
            
    file1.close()

    return file1
    
#################################################


    
