
#################################################
##
## Plotting utilities for AMSS-NCKU numerical relativity outputs
## Author: Xiaoqu
## 2024/10/01 --- 2025/09/14
##
#################################################

import numpy                               ## numpy for array operations
import matplotlib.pyplot    as     plt     ## matplotlib for plotting
from   mpl_toolkits.mplot3d import Axes3D  ## needed for 3D plots
import os                                  ## operating system utilities

import AMSS_NCKU_Input as input_data

# plt.rcParams['text.usetex'] = True  ## enable LaTeX fonts in plots



####################################################################################

## Plot black-hole puncture trajectories (2D)

def generate_puncture_orbit_plot( outdir, figure_outdir ):

    print(                                                    )
    print( " Plotting the black holes' trajectory (2D plot)" )
    print(                                                   )
    
    # path to data file
    file0 = os.path.join(outdir, "bssn_BH.dat")
    
    print( " Corresponding data file = ", file0 )

    # load the full data file (assumed whitespace-separated floats)
    data = numpy.loadtxt(file0)

    # print(data[:,0])
    # print(data[:,2])

    # initialize min/max arrays for black-hole coordinates
    BH_Xmin = numpy.zeros(input_data.puncture_number)
    BH_Xmax = numpy.zeros(input_data.puncture_number)
    BH_Ymin = numpy.zeros(input_data.puncture_number)
    BH_Ymax = numpy.zeros(input_data.puncture_number)
    BH_Zmin = numpy.zeros(input_data.puncture_number)
    BH_Zmax = numpy.zeros(input_data.puncture_number)
    
    # --------------------------
    
    # Plot black-hole displacement trajectory (XY)
    
    plt.figure( figsize=(8,8)                         )   ## figsize sets the figure size
    plt.title( " Black Hole Trajectory ", fontsize=18 )   ## fontsize sets the title size
    
    for i in range(input_data.puncture_number):
        BH_x       = data[:, 3*i+1]
        BH_y       = data[:, 3*i+2]
        BH_z       = data[:, 3*i+3]
        BH_Xmin[i] = min( BH_x )
        BH_Xmax[i] = max( BH_x )
        BH_Ymin[i] = min( BH_y )
        BH_Ymax[i] = max( BH_y )
        if i==0:
            plt.plot( BH_x, BH_y, color='red',   label="BH"+str(i+1), linewidth=2 )
        elif i==1:
            plt.plot( BH_x, BH_y, color='green', label="BH"+str(i+1), linewidth=2 )
        elif i==2:
            plt.plot( BH_x, BH_y, color='blue',  label="BH"+str(i+1), linewidth=2 )
        elif i==3:
            plt.plot( BH_x, BH_y, color='gray',  label="BH"+str(i+1), linewidth=2 )
            
    plt.xlabel( "X [M]",          fontsize=16 )
    plt.ylabel( "Y [M]",          fontsize=16 )
    plt.legend( loc='upper right'             )

    # set axis ranges
    Xmin0 = min( BH_Xmin )
    Xmax0 = max( BH_Xmax )
    Ymin0 = min( BH_Ymin )
    Ymax0 = max( BH_Ymax )
    Xmin  = min( Xmin0-2.0, -5.0 )
    Xmax  = max( Xmax0+2.0, +5.0 )
    Ymin  = min( Ymin0-2.0, -5.0 )
    Ymax  = max( Ymax0+2.0, +5.0 )
    plt.xlim( Xmin, Xmax )          # x axis range from Xmin to Xmax
    plt.ylim( Ymin, Ymax )          # y axis range from Ymin to Ymax
    
    plt.grid( color='gray', linestyle='--', linewidth=0.5 )  # display grid lines

    # plt.show(                                                      )
    plt.savefig( os.path.join(figure_outdir, "BH_Trajectory_XY.pdf") )
    plt.close(                                                       )
    
    # --------------------------
    
    # Plot black-hole displacement trajectory (XZ)
    
    plt.figure( figsize=(8,8)                         )   ## figsize sets the figure size
    plt.title( " Black Hole Trajectory ", fontsize=18 )   ## fontsize sets the title size
    
    for i in range(input_data.puncture_number):
        BH_x       = data[:, 3*i+1]
        BH_y       = data[:, 3*i+2]
        BH_z       = data[:, 3*i+3]
        BH_Xmin[i] = min( BH_x )
        BH_Xmax[i] = max( BH_x )
        BH_Zmin[i] = min( BH_z )
        BH_Zmax[i] = max( BH_z )
        if i==0:
            plt.plot( BH_x, BH_z, color='red',   label="BH"+str(i+1), linewidth=2 )
        elif i==1:
            plt.plot( BH_x, BH_z, color='green', label="BH"+str(i+1), linewidth=2 )
        elif i==2:
            plt.plot( BH_x, BH_z, color='blue',  label="BH"+str(i+1), linewidth=2 )
        elif i==3:
            plt.plot( BH_x, BH_z, color='gray',  label="BH"+str(i+1), linewidth=2 )
            
    plt.xlabel( "X [M]",          fontsize=16 )
    plt.ylabel( "Z [M]",          fontsize=16 )
    plt.legend( loc='upper right'             )

    # set axis ranges
    Xmin0 = min( BH_Xmin )
    Xmax0 = max( BH_Xmax )
    Zmin0 = min( BH_Zmin )
    Zmax0 = max( BH_Zmax )
    Xmin  = min( Xmin0-2.0, -5.0 )
    Xmax  = max( Xmax0+2.0, +5.0 )
    Zmin  = min( Zmin0-2.0, -5.0 )
    Zmax  = max( Zmax0+2.0, +5.0 )
    plt.xlim( Xmin, Xmax )         # x axis range from Xmin to Xmax
    plt.ylim( Zmin, Zmax )         # z axis range from Zmin to Zmax
    
    plt.grid( color='gray', linestyle='--', linewidth=0.5 )  # display grid lines

    # plt.show(                                                      )
    plt.savefig( os.path.join(figure_outdir, "BH_Trajectory_XZ.pdf") )
    plt.close(                                                       )
    
    # --------------------------
    
    # Plot black-hole displacement trajectory (YZ)
    
    plt.figure( figsize=(8,8)                         )   ## figsize sets the figure size
    plt.title( " Black Hole Trajectory ", fontsize=18 )   ## fontsize sets the title size
    
    for i in range(input_data.puncture_number):
        BH_x       = data[:, 3*i+1]
        BH_y       = data[:, 3*i+2]
        BH_z       = data[:, 3*i+3]
        BH_Ymin[i] = min( BH_y )
        BH_Ymax[i] = max( BH_y )
        BH_Zmin[i] = min( BH_z )
        BH_Zmax[i] = max( BH_z )
        if i==0:
            plt.plot( BH_y, BH_z, color='red',   label="BH"+str(i+1), linewidth=2 )
        elif i==1:
            plt.plot( BH_y, BH_z, color='green', label="BH"+str(i+1), linewidth=2 )
        elif i==2:
            plt.plot( BH_y, BH_z, color='blue',  label="BH"+str(i+1), linewidth=2 )
        elif i==3:
            plt.plot( BH_y, BH_z, color='gray',  label="BH"+str(i+1), linewidth=2 )
            
    plt.xlabel( "Y [M]",          fontsize=16 )
    plt.ylabel( "Z [M]",          fontsize=16 )
    plt.legend( loc='upper right'             )

    # set axis ranges
    Ymin0 = min( BH_Ymin )
    Ymax0 = max( BH_Ymax )
    Zmin0 = min( BH_Zmin )
    Zmax0 = max( BH_Zmax )
    Ymin  = min( Ymin0-2.0, -5.0 )
    Ymax  = max( Ymax0+2.0, +5.0 )
    Zmin  = min( Zmin0-2.0, -5.0 )
    Zmax  = max( Zmax0+2.0, +5.0 )
    plt.xlim( Ymin, Ymax )          # x axis range from Ymin to Ymax
    plt.ylim( Zmin, Zmax )          # z axis range from Zmin to Zmax
    
    plt.grid( color='gray', linestyle='--', linewidth=0.5 )  # display grid lines

    # plt.show(                                                      )
    plt.savefig( os.path.join(figure_outdir, "BH_Trajectory_YZ.pdf") )
    plt.close(                                                       )
    
    # --------------------------
    
    # extract coordinates for BH1 and BH2
    BH_x1 = data[:, 1]
    BH_y1 = data[:, 2]
    BH_z1 = data[:, 3]
    BH_x2 = data[:, 4]
    BH_y2 = data[:, 5]
    BH_z2 = data[:, 6]
    
    # --------------------------
    
    # Plot relative trajectory: (X2-X1) vs (Y2-Y1)

    plt.figure( figsize=(8,8)                                           )                          
    plt.title(  " Black Hole Trajectory ",                  fontsize=18 )   
    plt.plot(   (BH_x2-BH_x1), (BH_y2-BH_y1), color='blue', linewidth=2 )
    plt.xlabel( " $X_{2}$ - $X_{1}$ [M] ",                  fontsize=16 )
    plt.ylabel( " $Y_{2}$ - $Y_{1}$ [M] ",                  fontsize=16 )
    plt.legend( loc='upper right'                                       )

    # set axis ranges
    Xmin0 = min( (BH_x2 - BH_x1) )
    Xmax0 = max( (BH_x2 - BH_x1) ) 
    Ymin0 = min( (BH_y2 - BH_y1) )
    Ymax0 = max( (BH_y2 - BH_y1) ) 
    Xmin  = min( Xmin0-2.0, -5.0 )
    Xmax  = max( Xmax0+2.0, +5.0 )
    Ymin  = min( Ymin0-2.0, -5.0 )
    Ymax  = max( Ymax0+2.0, +5.0 )
    plt.xlim( Xmin, Xmax )          # x axis range from Xmin to Xmax
    plt.ylim( Ymin, Ymax )          # y axis range from Ymin to Ymax
    
    plt.grid( color='gray', linestyle='--', linewidth=0.5 )  # show grid lines

    plt.savefig( os.path.join(figure_outdir, "BH_Trajectory_21_XY.pdf")  )
    plt.close(                                                           )
    
    # --------------------------
    
    # plot BH displacement trajectory (X2-X1 Z2-Z1)
    
    plt.figure( figsize=(8,8)                                           )                          
    plt.title(  " Black Hole Trajectory ",                  fontsize=18 )   
    plt.plot(   (BH_x2-BH_x1), (BH_z2-BH_z1), color='blue', linewidth=2 )
    plt.xlabel( " $X_{2}$ - $X_{1}$ [M] ",                  fontsize=16 )
    plt.ylabel( " $Z_{2}$ - $Z_{1}$ [M] ",                  fontsize=16 )
    plt.legend( loc='upper right'                                       )

    # set axis ranges
    Xmin0 = min( (BH_x2 - BH_x1) )
    Xmax0 = max( (BH_x2 - BH_x1) ) 
    Zmin0 = min( (BH_z2 - BH_z1) )
    Zmax0 = max( (BH_z2 - BH_z1) ) 
    Xmin  = min( Xmin0-2.0, -5.0 )
    Xmax  = max( Xmax0+2.0, +5.0 )
    Zmin  = min( Zmin0-2.0, -5.0 )
    Zmax  = max( Zmax0+2.0, +5.0 )
    plt.xlim( Xmin, Xmax )          # x axis range from Xmin to Xmax
    plt.ylim( Zmin, Zmax )          # z axis range from Zmin to Zmax
    
    plt.grid( color='gray', linestyle='--', linewidth=0.5 )  # show grid lines

    plt.savefig( os.path.join(figure_outdir, "BH_Trajectory_21_XZ.pdf")  )
    plt.close(                                                           )
    
    # --------------------------
    
    # plot BH displacement trajectory (Y2-Y1 Z2-Z1)
    
    plt.figure( figsize=(8,8)                                           )                          
    plt.title(  " Black Hole Trajectory ",                  fontsize=18 )   
    plt.plot(   (BH_y2-BH_y1), (BH_z2-BH_z1), color='blue', linewidth=2 )
    plt.xlabel( " $Y_{2}$ - $Y_{1}$ [M] ",                  fontsize=16 )
    plt.ylabel( " $Z_{2}$ - $Z_{1}$ [M] ",                  fontsize=16 )
    plt.legend( loc='upper right'                                       )

    # set axis ranges
    Ymin0 = min( (BH_y2 - BH_y1) )
    Ymax0 = max( (BH_y2 - BH_y1) ) 
    Zmin0 = min( (BH_z2 - BH_z1) )
    Zmax0 = max( (BH_z2 - BH_z1) ) 
    Ymin  = min( Ymin0-2.0, -5.0 )
    Ymax  = max( Ymax0+2.0, +5.0 )
    Zmin  = min( Zmin0-2.0, -5.0 )
    Zmax  = max( Zmax0+2.0, +5.0 )
    plt.xlim( Ymin, Ymax )          # x axis range from Ymin to Ymax
    plt.ylim( Zmin, Zmax )          # z axis range from Zmin to Zmax
    
    plt.grid( color='gray', linestyle='--', linewidth=0.5 )  # show grid lines

    plt.savefig( os.path.join(figure_outdir, "BH_Trajectory_21_YZ.pdf")  )
    plt.close(                                                           )
    
    # --------------------------
    
    # NOTE: file0 is only a filename string here; no file object to close
    
    print(                      )
    print( " Black holes' trajectory plot has been finished (2D plot)" )
    print(                                                             )

    return

####################################################################################



####################################################################################

## Plot relative distances between black holes

def generate_puncture_distence_plot( outdir, figure_outdir ):

    print(                                                )
    print( " Plotting the black hole relative distance " )
    print(                                               )
    
    # path to data file
    file0 = os.path.join(outdir, "bssn_BH.dat")
    
    print( " Corresponding data file = ", file0 )

    # load the full data file (assumed whitespace-separated floats)
    data = numpy.loadtxt(file0)
    
    # --------------------------
    
    # --------------------------

    # Plot each black hole's distance R from the origin as a function of time

    # initialize min/max arrays for BH distances
    BH_Rmin = numpy.zeros(input_data.puncture_number)
    BH_Rmax = numpy.zeros(input_data.puncture_number)

    # create a new figure
    fig = plt.figure( figsize=(8,8) )
    plt.title( " Black Hole Position R ", fontsize=18 )   # title

    BH_time = data[:, 0]
    
    for i in range(input_data.puncture_number):
        BH_x = data[:, 3*i+1]
        BH_y = data[:, 3*i+2]
        BH_z = data[:, 3*i+3]
        BH_R = (BH_x*BH_x + BH_y*BH_y + BH_z*BH_z)**0.5
        # compute distance R using numpy
        BH_Rmin[i] = min( BH_R )
        BH_Rmax[i] = max( BH_R )
        if i==0:
            plt.plot( BH_time, BH_R, color='red',   label="BH"+str(i+1), linewidth=2 )
        elif i==1:
            plt.plot( BH_time, BH_R, color='green', label="BH"+str(i+1), linewidth=2 )
        elif i==2:
            plt.plot( BH_time, BH_R, color='blue',  label="BH"+str(i+1), linewidth=2 )
        elif i==3:
            plt.plot( BH_time, BH_R, color='gray',  label="BH"+str(i+1), linewidth=2 )

    # set axis labels
    plt.xlabel( " $T$ [M] ",      fontsize=16 )
    plt.ylabel( " $R$ [M] ",      fontsize=16 )
    plt.legend( loc='upper right'             )

    # set axis ranges
    R_min0 = min( BH_Rmin ) 
    R_max0 = max( BH_Rmax )
    R_min  = max( R_min0-2.0,  0.0 )
    R_max  = max( R_max0+2.0, +5.0 )
    plt.ylim( R_min, R_max )             # y axis range from R_min to R_max
    
    plt.grid( color='gray', linestyle='--', linewidth=0.5 )  # display grid lines

    # plt.show(                                                   )
    plt.savefig( os.path.join(figure_outdir, "BH_Position_R.pdf") )
    plt.close(                                                    )
    
    # --------------------------
    
    # extract coordinates for BH1 and BH2
    BH_x1  = data[:, 1]
    BH_y1  = data[:, 2]
    BH_z1  = data[:, 3]
    BH_x2  = data[:, 4]
    BH_y2  = data[:, 5]
    BH_z2  = data[:, 6]
    
    # compute relative distance R12 between BH1 and BH2
    BH_R12 = ( (BH_x2-BH_x1)**2 + (BH_y2-BH_y1)**2 + (BH_z2-BH_z1)**2 )**0.5
    
    # --------------------------
    
    # plot relative distance R12 between BH1 and BH2 as a function of time

    plt.figure( figsize=(8,8)                              )                          
    plt.title(  " Black Hole Distance ",       fontsize=18 )   
    plt.plot(   BH_time, BH_R12, color='blue', linewidth=2 )
    plt.xlabel( " $T$ [M] ",                   fontsize=16 )
    plt.ylabel( " $R_{12}$ [M] ",              fontsize=16 )
    plt.legend( loc='upper right'                          )

    # set axis ranges
    R12_min0 = min( BH_R12 )
    R12_max0 = max( BH_R12 ) 
    R12_min  = max( R12_min0-2.0,  0.0 )
    R12_max  = max( R12_max0+2.0, +5.0 )
    plt.ylim( R12_min, R12_max )             # y axis range from R12_min to R12_max
    
    plt.grid( color='gray', linestyle='--', linewidth=0.5 )  # show grid lines

    plt.savefig( os.path.join(figure_outdir, "BH_Distance_21.pdf")  )
    plt.close(                                                      )
    
    print(                           )
    print( " black hole relative distance plot has been finished " )
    print(                                                         )
    
    # --------------------------
 
    return

####################################################################################



####################################################################################

## Plot black-hole puncture trajectories (3D)

def generate_puncture_orbit_plot3D( outdir, figure_outdir ):

    print(                               )
    print( " Plotting the black holes' trajectory (3D plot) " )
    print(                               )
    
    # path to data file
    file0 = os.path.join(outdir, "bssn_BH.dat")
    
    print( " Corresponding data file = ", file0 )

    # load the full data file (assumed whitespace-separated floats)
    data = numpy.loadtxt(file0)

    # initialize min/max arrays for black-hole coordinates
    BH_Xmin = numpy.zeros(input_data.puncture_number)
    BH_Xmax = numpy.zeros(input_data.puncture_number)
    BH_Ymin = numpy.zeros(input_data.puncture_number)
    BH_Ymax = numpy.zeros(input_data.puncture_number)
    BH_Zmin = numpy.zeros(input_data.puncture_number)
    BH_Zmax = numpy.zeros(input_data.puncture_number)
    
    # create a new figure
    fig = plt.figure( figsize=(8,8) )
 
    # create a 3D axes
    ax = fig.add_subplot(111, projection='3d')
    # set title
    ax.set_title( " Black Hole Trajectory ", fontsize=18 )
    
    for i in range(input_data.puncture_number):
        BH_x = data[:, 3*i+1]
        BH_y = data[:, 3*i+2]
        BH_z = data[:, 3*i+3]
        BH_Xmin[i] = min( BH_x )
        BH_Xmax[i] = max( BH_x )
        BH_Ymin[i] = min( BH_y )
        BH_Ymax[i] = max( BH_y )
        BH_Zmin[i] = min( BH_z )
        BH_Zmax[i] = max( BH_z )
        if i==0:
            ax.plot( BH_x, BH_y, BH_z, color='red',   label="BH"+str(i+1), linewidth=2 )
        elif i==1:
            ax.plot( BH_x, BH_y, BH_z, color='green', label="BH"+str(i+1), linewidth=2 )
        elif i==2:
            ax.plot( BH_x, BH_y, BH_z, color='blue',  label="BH"+str(i+1), linewidth=2 )
        elif i==3:
            ax.plot( BH_x, BH_y, BH_z, color='gray',  label="BH"+str(i+1), linewidth=2 )

    # set axis labels
    ax.set_xlabel( "X [M]",          fontsize=16 )
    ax.set_ylabel( "Y [M]",          fontsize=16 )
    ax.set_zlabel( "Z [M]",          fontsize=16 )
    plt.legend(    loc='upper right'             )

    # set axis ranges
    Xmin0 = min( BH_Xmin )
    Xmax0 = max( BH_Xmax )
    Ymin0 = min( BH_Ymin )
    Ymax0 = max( BH_Ymax )
    Zmin0 = min( BH_Zmin )
    Zmax0 = max( BH_Zmax )
    Xmin  = min( Xmin0-2.0, -5.0 )
    Xmax  = max( Xmax0+2.0, +5.0 )
    Ymin  = min( Ymin0-2.0, -5.0 )
    Ymax  = max( Ymax0+2.0, +5.0 )
    Zmin  = min( Zmin0-2.0, -5.0 )
    Zmax  = max( Zmax0+2.0, +5.0 )
    ax.set_xlim( [Xmin, Xmax] )      
    ax.set_ylim( [Ymin, Ymax] )      
    ax.set_zlim( [Zmin, Zmax] )     

    plt.savefig( os.path.join(figure_outdir, "BH_Trajectory_3D.pdf") )
    plt.close(                                                       )
    
    print(                                                             )
    print( " Black holes' trajectory plot has been finished (3D plot)" )
    print(                                                             )
 
    return


####################################################################################



####################################################################################

## Plot gravitational-wave waveform Psi4

def generate_gravitational_wave_psi4_plot( outdir, figure_outdir, detector_number_i ):
    

    # path to data file
    file0 = os.path.join(outdir, "bssn_psi4.dat")

    if ( detector_number_i == 0 ):
        print(                                                )
        print( " Plotting the Weyl conformal component Psi4 " )
        print(                                                )
        print( " corresponding data file = ", file0 )
        print(                                      )

    print( " Begin the Weyl conformal Psi4 plot for detector number = ", detector_number_i )
    
    # load the full data file (assumed whitespace-separated floats)
    data = numpy.loadtxt(file0)
    
    # extract columns from the Phi4 file
    time                 = data[:,0]
    psi4_l2m2m_real      = data[:,1]
    psi4_l2m2m_imaginary = data[:,2]
    psi4_l2m1m_real      = data[:,3]
    psi4_l2m1m_imaginary = data[:,4]
    psi4_l2m0_real       = data[:,5]
    psi4_l2m0_imaginary  = data[:,6]
    psi4_l2m1_real       = data[:,7]
    psi4_l2m1_imaginary  = data[:,8]
    psi4_l2m2_real       = data[:,9]
    psi4_l2m2_imaginary  = data[:,10]
    
    # NOTE: file0 is only a filename string here; no file object to close

    # In Python division returns float; use integer division here
    length = len(time) // input_data.Detector_Number 
    
    time2                 = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m2m_real2      = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m2m_imaginary2 = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m1m_real2      = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m1m_imaginary2 = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m0_real2       = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m0_imaginary2  = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m1_real2       = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m1_imaginary2  = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m2_real2       = numpy.zeros( (input_data.Detector_Number, length) )
    psi4_l2m2_imaginary2  = numpy.zeros( (input_data.Detector_Number, length) )
    
    # split data into arrays corresponding to each detector radius
    for i in range(input_data.Detector_Number):
        for j in range(length):
            time2[i,j]                 = time[                 j*input_data.Detector_Number + i ]
            psi4_l2m2m_real2[i,j]      = psi4_l2m2m_real[      j*input_data.Detector_Number + i ]
            psi4_l2m2m_imaginary2[i,j] = psi4_l2m2m_imaginary[ j*input_data.Detector_Number + i ]
            psi4_l2m1m_real2[i,j]      = psi4_l2m1m_real[      j*input_data.Detector_Number + i ]
            psi4_l2m1m_imaginary2[i,j] = psi4_l2m1m_imaginary[ j*input_data.Detector_Number + i ]
            psi4_l2m0_real2[i,j]       = psi4_l2m0_real[       j*input_data.Detector_Number + i ]
            psi4_l2m0_imaginary2[i,j]  = psi4_l2m0_imaginary[  j*input_data.Detector_Number + i ]
            psi4_l2m1_real2[i,j]       = psi4_l2m1_real[       j*input_data.Detector_Number + i ]
            psi4_l2m1_imaginary2[i,j]  = psi4_l2m1_imaginary[  j*input_data.Detector_Number + i ]
            psi4_l2m2_real2[i,j]       = psi4_l2m2_real[       j*input_data.Detector_Number + i ]
            psi4_l2m2_imaginary2[i,j]  = psi4_l2m2_imaginary[  j*input_data.Detector_Number + i ]
            
    # compute detector distance from input parameters
    Detector_Interval   = ( input_data.Detector_Rmax - input_data.Detector_Rmin ) / ( input_data.Detector_Number - 1 )
    Detector_Distance_R = input_data.Detector_Rmax - Detector_Interval * detector_number_i
    
    plt.figure( figsize=(8,8) )                                   ## figsize sets the figure size
    plt.title( f" Gravitational Wave $\Psi_{4}$   Detector Distance =  { Detector_Distance_R } ", fontsize=18 )   ## fontsize sets the title size
    plt.plot( time2[detector_number_i], psi4_l2m0_real2[detector_number_i],      \
              color='red',    label="l=2 m=0 real",                       linewidth=2 )
    plt.plot( time2[detector_number_i], psi4_l2m0_imaginary2[detector_number_i], \
              color='orange', label="l=2 m=0 imaginary",  linestyle='--', linewidth=2 )
    plt.plot( time2[detector_number_i], psi4_l2m1_real2[detector_number_i],      \
              color='green',  label="l=2 m=1 real",                       linewidth=2 )
    plt.plot( time2[detector_number_i], psi4_l2m1_imaginary2[detector_number_i], \
              color='cyan',   label="l=2 m=1 imaginary",  linestyle='--', linewidth=2 )
    plt.plot( time2[detector_number_i], psi4_l2m2_real2[detector_number_i],      \
              color='black',  label="l=2 m=2 real",                       linewidth=2 )
    plt.plot( time2[detector_number_i], psi4_l2m2_imaginary2[detector_number_i], \
              color='gray',   label="l=2 m=2 imaginary",  linestyle='--', linewidth=2 )
    plt.xlabel( "T [M]",          fontsize=16 )
    plt.ylabel( r"$R*\Psi$",      fontsize=16 )
    plt.legend( loc='upper right'             )
    plt.grid(   color='gray', linestyle='--', linewidth=0.5 )  # display grid lines
    plt.savefig( os.path.join(figure_outdir, "Gravitational_Psi4_Detector_" + str(detector_number_i) + ".pdf") )
    
    
    print( " The Weyl Conformal component Psi4 plot has been finished ", " detector number ", detector_number_i )
    print(                                                                                            )

    if ( detector_number_i == (input_data.Detector_Number-1) ):
        print(                                    )
        print( " The Weyl conformal component Psi4 plots have been finished " )
        print(                                                                 )


    return

####################################################################################



####################################################################################

## Plot ADM mass and angular momentum

def generate_ADMmass_plot( outdir, figure_outdir, detector_number_i ):

    
    # path to data file
    file0 = os.path.join(outdir, "bssn_ADMQs.dat")

    if ( detector_number_i == 0 ):
        print(                                                )
        print( " Plotting the ADM mass and angular momentum " )
        print(                                                )
        print( " corresponding data file = ", file0 )
        print(                                      )
    
    print( " Begin the ADM momentum plot for detector number =  ", detector_number_i )


    # load the full data file (assumed whitespace-separated floats)
    data = numpy.loadtxt(file0)
    
    # extract columns from the ADM momentum file
    time     = data[:,0]
    ADM_mass = data[:,1]
    ADM_Px   = data[:,2]
    ADM_Py   = data[:,3]
    ADM_Pz   = data[:,4]
    ADM_Jx   = data[:,5]
    ADM_Jy   = data[:,6]
    ADM_Jz   = data[:,7]
    
    # NOTE: file0 is only a filename string here; no file object to close

    # In Python division returns a float; use integer division here
    length = len(time) // input_data.Detector_Number
    
    '''
    # split data into arrays corresponding to each detector radius (disabled)
    # time2     = time.reshape( (input_data.Detector_Number, length) )
    # ADM_mass2 = ADM_mass.reshape( (input_data.Detector_Number, length) )
    # ADM_Px2   = ADM_Px.reshape( (input_data.Detector_Number, length) )
    # ADM_Py2   = ADM_Py.reshape( (input_data.Detector_Number, length) )
    # ADM_Pz2   = ADM_Pz.reshape( (input_data.Detector_Number, length) )
    # ADM_Jx2   = ADM_Jx.reshape( (input_data.Detector_Number, length) )
    # ADM_Jy2   = ADM_Jy.reshape( (input_data.Detector_Number, length) )
    # ADM_Jz2   = ADM_Jz.reshape( (input_data.Detector_Number, length) )
    '''
    # Rows/cols in reshape were unclear; use straightforward indexing instead
    time2     = numpy.zeros( (input_data.Detector_Number, length) )
    ADM_mass2 = numpy.zeros( (input_data.Detector_Number, length) )
    ADM_Px2   = numpy.zeros( (input_data.Detector_Number, length) )
    ADM_Py2   = numpy.zeros( (input_data.Detector_Number, length) )
    ADM_Pz2   = numpy.zeros( (input_data.Detector_Number, length) )
    ADM_Jx2   = numpy.zeros( (input_data.Detector_Number, length) )
    ADM_Jy2   = numpy.zeros( (input_data.Detector_Number, length) )
    ADM_Jz2   = numpy.zeros( (input_data.Detector_Number, length) )
    
    # split data into arrays corresponding to each detector radius
    for i in range(input_data.Detector_Number):
        for j in range(length):
            time2[i,j]     = time[     j*input_data.Detector_Number + i ]
            ADM_mass2[i,j] = ADM_mass[ j*input_data.Detector_Number + i ]
            ADM_Px2[i,j]   = ADM_Px[   j*input_data.Detector_Number + i ]
            ADM_Py2[i,j]   = ADM_Py[   j*input_data.Detector_Number + i ]
            ADM_Pz2[i,j]   = ADM_Pz[   j*input_data.Detector_Number + i ]
            ADM_Jx2[i,j]   = ADM_Jx[   j*input_data.Detector_Number + i ]
            ADM_Jy2[i,j]   = ADM_Jy[   j*input_data.Detector_Number + i ]
            ADM_Jz2[i,j]   = ADM_Jz[   j*input_data.Detector_Number + i ]
            
    # compute detector distance from input parameters
    Detector_Interval   = ( input_data.Detector_Rmax - input_data.Detector_Rmin ) / ( input_data.Detector_Number - 1 )
    Detector_Distance_R = input_data.Detector_Rmax - Detector_Interval * detector_number_i
            
    # Plot ADM momentum for the current detector radius
    plt.figure( figsize=(8,8) )                  
    plt.title(f" ADM Momentum    Detector Distence = {Detector_Distance_R}", fontsize=18 )   
    plt.plot( time2[detector_number_i], ADM_mass2[detector_number_i], color='red',   label="ADM Mass", linewidth=2 )
    plt.plot( time2[detector_number_i], ADM_Px2[detector_number_i],   color='green', label="ADM Px",   linewidth=2 )
    plt.plot( time2[detector_number_i], ADM_Py2[detector_number_i],   color='cyan',  label="ADM Py",   linewidth=2 )
    plt.plot( time2[detector_number_i], ADM_Pz2[detector_number_i],   color='blue',  label="ADM Pz",   linewidth=2 )
    plt.xlabel( "T [M]",            fontsize=16 )
    plt.ylabel( "ADM Momentum [M]", fontsize=16 )
    plt.legend( loc='upper right'               )
    plt.grid(   color='gray', linestyle='--', linewidth=0.5 )  # display grid lines
    plt.savefig( os.path.join(figure_outdir, "ADM_Mass_Dector_" + str(detector_number_i) + ".pdf") )
    
    # Plot ADM angular momentum for the current detector radius
    plt.figure( figsize=(8,8) )                  
    plt.title(f" ADM Angular Momentum    Detector Distence = {Detector_Distance_R}", fontsize=18 )   
    # plt.plot( time2[detector_number_i], ADM_mass2[detector_number_i], color='red',   label="ADM Mass", linewidth=2 )
    plt.plot( time2[detector_number_i], ADM_Jx2[detector_number_i],   color='green', label="ADM Jx",   linewidth=2 )
    plt.plot( time2[detector_number_i], ADM_Jy2[detector_number_i],   color='cyan',  label="ADM Jy",   linewidth=2 )
    plt.plot( time2[detector_number_i], ADM_Jz2[detector_number_i],   color='blue',  label="ADM Jz",   linewidth=2 )
    plt.xlabel( "T [M]",                        fontsize=16 )
    plt.ylabel( "ADM Angular Momentum [$M^2$]", fontsize=16 )
    plt.legend( loc='upper right'                           )
    plt.grid(   color='gray', linestyle='--', linewidth=0.5 )  # display grid lines
    plt.savefig( os.path.join(figure_outdir, "ADM_Angular_Momentum_Dector_" + str(detector_number_i) + ".pdf") )
    

    print( " ADM momentum plot has been finished, detector number =  ", detector_number_i )
    print(                                                                                )

    if ( detector_number_i == (input_data.Detector_Number-1) ):
        print( " The ADM mass and augular momentum plots have been finished " )
        print(                                                                )

    return
    
####################################################################################



####################################################################################

## Plot constraint violation for each grid level

def generate_constraint_check_plot( outdir, figure_outdir, input_level_number ):

    # path to data file
    file0 = os.path.join(outdir, "bssn_constraint.dat")

    if ( input_level_number == 0 ):
        print(                                                   )
        print( " Plotting the constraint violation for each grid level" )
        print(                                                          )
        print( " corresponding data file = ", file0 )
        print(                                      )

    print( " Begin the constraint violation plot for grid level number =  ", input_level_number )
    
    # load the full data file (assumed whitespace-separated floats)
    data = numpy.loadtxt(file0)
    
    # extract columns from the constraint data file
    time          = data[:,0]
    Constraint_H  = data[:,1]
    Constraint_Px = data[:,2]
    Constraint_Py = data[:,3]
    Constraint_Pz = data[:,4]
    Constraint_Gx = data[:,5]
    Constraint_Gy = data[:,6]
    Constraint_Gz = data[:,7]
    
    # NOTE: file0 is only a filename string here; no file object to close

    # initialize arrays for different quantities
    
    level_number = input_level_number
    length0      = input_data.grid_level
    length1      = len(time) // length0
    
    time2          = numpy.zeros( (length0, length1) )
    Constraint_H2  = numpy.zeros( (length0, length1) )
    Constraint_Px2 = numpy.zeros( (length0, length1) )
    Constraint_Py2 = numpy.zeros( (length0, length1) )
    Constraint_Pz2 = numpy.zeros( (length0, length1) )
    Constraint_Gx2 = numpy.zeros( (length0, length1) )
    Constraint_Gy2 = numpy.zeros( (length0, length1) )
    Constraint_Gz2 = numpy.zeros( (length0, length1) )
    
    # split data into arrays corresponding to each grid level
    for i in range(length0):
        for j in range(length1):
            time2[i,j]          = time[          j*length0 + i ]
            Constraint_H2[i,j]  = Constraint_H[  j*length0 + i ]
            Constraint_Px2[i,j] = Constraint_Px[ j*length0 + i ]
            Constraint_Py2[i,j] = Constraint_Py[ j*length0 + i ]
            Constraint_Pz2[i,j] = Constraint_Pz[ j*length0 + i ]
    
    # Plot constraint violation for the outermost grid level
    plt.figure( figsize=(8,8) )                    
    plt.title( f" ADM Constraint  Grid Level = {input_level_number}", fontsize=18 )   
    plt.plot( time2[level_number], Constraint_H2[level_number],  color='red',   label="ADM Constraint H",  linewidth=2 )
    plt.plot( time2[level_number], Constraint_Px2[level_number], color='green', label="ADM Constraint Px", linewidth=2 )
    plt.plot( time2[level_number], Constraint_Py2[level_number], color='cyan',  label="ADM Constraint Py", linewidth=2 )
    plt.plot( time2[level_number], Constraint_Pz2[level_number], color='blue',  label="ADM Constraint Pz", linewidth=2 )
    plt.xlabel( "T [M]",          fontsize=16 )
    plt.ylabel( "ADM Constraint", fontsize=16 )
    plt.legend( loc='upper right'             )
    plt.grid(   color='gray', linestyle='--', linewidth=0.5 )  # display grid lines
    plt.savefig( os.path.join(figure_outdir, "ADM_Constraint_Grid_Level_" + str(input_level_number) + ".pdf") )
    

    print( " Constraint violation plot has been finished, grid level number = ", input_level_number )
    print(                                                                                          )
    
    if ( input_level_number == (input_data.grid_level-1) ):
        print( " Constraint violation plot has been finished " )
        print(                                                 )

    return

####################################################################################



####################################################################################

# Standalone examples
'''
outdir = "./BBH_q=1"

generate_puncture_orbit_plot(    outdir, outdir )
generate_puncture_orbit_plot3D(  outdir, outdir )
generate_puncture_distence_plot( outdir, outdir )

for i in range(input_data.grid_level):
    generate_constraint_check_plot( outdir, outdir, i )

for i in range(input_data.Detector_Number):
    generate_ADMmass_plot( outdir, outdir, i )

for i in range(input_data.Detector_Number):
    generate_gravitational_wave_psi4_plot( outdir, outdir, i )
'''
####################################################################################


