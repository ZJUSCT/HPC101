
#################################################
##
## This file extracts gravitational-wave strain amplitude
## from AMSS-NCKU numerical-relativity outputs and plots it.
## Author: Xiaoqu
## Dates: 2024/10/01 --- 2025/11/20
##
#################################################

import numpy                               ## numpy for array operations
import scipy                               ## scipy for interpolation and signal processing
import math
import matplotlib.pyplot    as     plt     ## matplotlib for plotting
import os                                  ## os for system/file operations

import AMSS_NCKU_Input as input_data

# plt.rcParams['text.usetex'] = True  ## enable LaTeX fonts in plots


####################################################################################



####################################################################################

## Compute Fourier transform for discrete time data in [t0, t1].
## Adapted from deepseek examples.

## Parameters:
## time : time series [0, t0]
## f : function value series
## apply_window : apply windowing (recommended True)
## zero_pad_factor : zero-padding factor (recommended 2-8)

## Returns:
## frequency : frequency axis (Hz)
## frequency_spectrum : complex frequency spectrum

def compute_frequency_spectrum(time, signal_f, apply_window=True, zero_pad_factor=4):

    ## Sampling parameters
    N  = len(time)
    dt = time[1] - time[0]        ## sample interval
    fs = 1/dt                     ## sampling frequency
    omega_s = fs * 2.0 * math.pi  ## sampling angular frequency

    ## Data preprocessing
    f_detrended = signal_f - numpy.mean(signal_f)  ## remove DC offset

    ## Windowing (reduce spectral leakage)
    if apply_window:
        # window = scipy.signal.windows.hann(N)  # Hann window
        window = scipy.signal.windows.tukey(N, alpha=0.1)  # or Tukey window
        f_windowed = f_detrended * window
    else:
        f_windowed = f_detrended
    
    # Zero-padding (improve frequency resolution)
    M = zero_pad_factor * N
    f_padded = numpy.zeros(M)
    f_padded[:N] = f_windowed
    
    # Execute FFT to obtain complex frequency spectrum

    # Use numpy.fft.fft to perform the fast Fourier transform
    # Basic signature: numpy.fft.fft(a, n=None, axis=-1, norm=None)
    # a: input 1-D array to be transformed
    # n: transform length; if n>len(a) the input is zero-padded, if n<len(a) it is truncated
    # axis: axis along which to compute the FFT (default: last axis)
    # norm: normalization mode (None, 'ortho', etc.)
    # numpy.fft.fft returns a complex array; the corresponding frequency axis spans -fs/2 to fs/2
    
    frequency_spectrum = numpy.fft.fft(f_padded, norm='ortho')    
    # frequency_spectrum = numpy.fft.fft(f_detrended, norm='ortho')  
    # frequency_spectrum = numpy.fft.fft(f_windowed, norm='ortho')     
    
    # Frequency axis generation

    # Use numpy.fft.fftfreq to build the discrete frequency axis:
    # numpy.fft.fftfreq(n, d=1.0)
    # n: signal length (number of samples)
    # d: sample spacing (time between samples)
    frequency = numpy.fft.fftfreq(M, dt)
    # frequency = numpy.fft.fftfreq(N, dt)

    # The above gives frequency in Hz; convert to angular frequency
    frequency_omega = 2.0 * math.pi * frequency
    
    # Amplitude correction (compensate for window loss)
    if apply_window:
        window_gain = numpy.mean(window)  # window gain
        frequency_spectrum /= window_gain
    
    ## return frequency_omega, frequency_spectrum
    return frequency, frequency_omega, frequency_spectrum

    ## Window types, properties and typical use cases:
    ## Hann    : wide main lobe, fast sidelobe decay       -> black-hole quasi-periodic oscillations (QPO)
    ## Tukey   : adjustable flat-top region               -> gravitational-wave chirp signals
    ## Blackman: optimal sidelobe suppression              -> high dynamic-range data

####################################################################################


####################################################################################

def frequency_filter_integration(omega, frequency_spectrum, omega0):

    '''
    ## modifiy the part |omega| < omega0
    omega_filter = numpy.where(omega < omega0, omega0, omega)
    '''

    ## Replace region |omega| < omega0 with signed omega0
    # Build replacements: positive omegas -> +omega0, negative -> -omega0
    replacements = numpy.where(omega >= 0, omega0, -omega0)

    # Apply replacement where abs(omega) < omega0
    omega_filter = numpy.where( numpy.abs(omega) < omega0, replacements, omega )

    ## Integrand for frequency-domain integration
    ## Note: convolution in time corresponds to multiplication in frequency
    frequency_integration = - frequency_spectrum / (omega_filter)**2

    return frequency_integration

####################################################################################


####################################################################################

## This function replaces |omega| < omega0 with signed omega0

def omega_filter(omega, omega0):

    ## Replace region |omega| < omega0 with signed omega0
    # Build replacements: positive omegas -> +omega0, negative -> -omega0
    replacements = numpy.where(omega >= 0, omega0, -omega0)
    
    # Apply replacements where |omega| < omega0
    # omega_filter = numpy.where(mask, replacements, omega)
    omega_filter = numpy.where( numpy.abs(omega) < omega0, replacements, omega )

    return omega_filter
    
####################################################################################


####################################################################################

## Inverse Fourier transform utility

## Inputs:
## omega : frequency axis (Hz)
## F_omega : complex frequency-domain data
## sampling_factor : sampling multiplier for reconstruction
## original_zero_pad_factor : original zero-padding factor used in forward FFT

## Returns:
## t : time axis
## reconstructed : reconstructed time-domain signal


def inverse_fourier_transform(omega, F_omega, sampling_factor=2, original_zero_pad_factor=4):
    
    # Compute sampling parameters
    N = len(F_omega)
    domega = omega[1] - omega[0]  # frequency resolution

    # Determine sampling rate
    # To avoid aliasing, ensure sampling rate >= 2 * max frequency (Nyquist)
    if sampling_factor > 2: 
        sampling_rate_omega = sampling_factor * omega.max()
    else:
        sampling_rate_omega = 2.0 * omega.max()
    # The input omega is angular frequency; convert to ordinary frequency
    ## dt = 1.0 / sampling_rate_omega
    frequency = omega / (2.0*math.pi)
    dt = 2 * math.pi / sampling_rate_omega  
    
    '''
    # DC component check
    if not numpy.isclose(omega[0], 0):
        warnings.warn("Frequency-domain data does not include zero-frequency component; DC offset may result")
    '''
    
    # Perform inverse FFT (use same normalization as forward transform)
    reconstructed_signal = numpy.fft.ifft(F_omega, norm='ortho') 
    # Note: numpy.fft.ifft already handles normalization for 'ortho'

    ## Build time axis
    ## If zero-padding was used originally, recover the unpadded length
    if (original_zero_pad_factor > 1):
        N0 = N // original_zero_pad_factor
        t = numpy.arange(0, N0*dt, dt)
        reconstructed_signal2 = reconstructed_signal[:N0]
    ## If no zero-padding
    else:
        t = numpy.arange(0, N*dt, dt)
        reconstructed_signal2 = reconstructed_signal
    
    # Handle real signals
    if numpy.allclose(numpy.imag(reconstructed_signal2), 0):
        reconstructed_signal3 = numpy.real(reconstructed_signal2)
    
    return t[:len(reconstructed_signal3)], reconstructed_signal3


####################################################################################


####################################################################################

# Instantaneous frequency estimation using analytic signal (Hilbert transform)

def instantaneous_frequency(signal, sampling_rate):
    """
    Compute instantaneous frequency of a signal.
    :param signal: input time-domain sampled signal
    :param sampling_rate: sampling rate
    :return: time array and instantaneous frequency array
    """
    analytic_signal = scipy.signal.hilbert(signal)
    phase = numpy.unwrap(numpy.angle(analytic_signal))
    time = numpy.arange(len(signal)) / sampling_rate
    frequency = numpy.gradient(phase, time) / (2 * numpy.pi)
    return time, frequency

def get_frequency_at_t1(signal, sampling_rate, t1):
    """
    Get instantaneous frequency at time t1
    :param signal: input time-domain sampled signal
    :param sampling_rate: sampling rate
    :param t1: target time
    :return: instantaneous frequency at t1
    """
    time, freq = instantaneous_frequency(signal, sampling_rate)
    index = numpy.argmin(numpy.abs(time - t1))
    return freq[index]
    
    
####################################################################################



####################################################################################

## Function to plot gravitational-wave waveform h

## Inputs:
## outdir             path to data directory
## figure_outdir      path to figure output directory
## detector_number_i  detector index
## total_mass         total system mass

def generate_gravitational_wave_amplitude_plot( outdir, figure_outdir, detector_number_i ):


    # build file path
    file0 = os.path.join(outdir, "bssn_psi4.dat")

    if ( detector_number_i == 0 ):
        print()
        print("Plotting the gravitational-wave strain amplitude h")
        print()
        print("The corresponding data file is", file0)
        print()

    print()
    print( "Plotting gravitational-wave data for detector no.", detector_number_i )

    
    # read entire data file, assume whitespace-delimited floats
    data = numpy.loadtxt(file0)
    
    # extract columns from psi4 file
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
    
    # Note: file0 is just a filename; no file.open() was used, so nothing to close
    # file0.close()
    
    # Use integer division to compute length per detector
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
    
    # Split data into per-detector-radius series
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

    
    ## Compute discrete Fourier transforms of Psi4 data
    ## l=2 m=-2 spectrum
    psi4_l2m2m_real_frequency, psi4_l2m2m_real_omega, psi4_l2m2m_real_omega_spectrem                                                             \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m2m_real2[detector_number_i],      apply_window=True, zero_pad_factor=4 )
    psi4_l2m2m_imaginary_frequency, psi4_l2m2m_imaginary_omega, psi4_l2m2m_imaginary_omega_spectrem                                              \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m2m_imaginary2[detector_number_i], apply_window=True, zero_pad_factor=4 )
    ## l=2 m=-1 spectrum
    psi4_l2m1m_real_frequency, psi4_l2m1m_real_omega, psi4_l2m1m_real_omega_spectrem                                                             \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m1m_real2[detector_number_i],      apply_window=True, zero_pad_factor=4 )
    psi4_l2m1m_imaginary_frequency, psi4_l2m1m_imaginary_omega, psi4_l2m1m_imaginary_omega_spectrem                                              \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m1m_imaginary2[detector_number_i], apply_window=True, zero_pad_factor=4 )
    ## l=2 m=0 spectrum
    psi4_l2m0_real_frequency, psi4_l2m0_real_omega, psi4_l2m0_real_omega_spectrem                                                                \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m0_real2[detector_number_i],       apply_window=True, zero_pad_factor=4 )
    psi4_l2m0_imaginary_frequency, psi4_l2m0_imaginary_omega, psi4_l2m0_imaginary_omega_spectrem                                                 \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m0_imaginary2[detector_number_i],  apply_window=True, zero_pad_factor=4 )
    ## l=2 m=1 spectrum
    psi4_l2m1_real_frequency, psi4_l2m1_real_omega, psi4_l2m1_real_omega_spectrem                                                                \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m1_real2[detector_number_i],       apply_window=True, zero_pad_factor=4 )
    psi4_l2m1_imaginary_frequency, psi4_l2m1_imaginary_omega, psi4_l2m1_imaginary_omega_spectrem                                                 \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m1_imaginary2[detector_number_i],  apply_window=True, zero_pad_factor=4 )
    ## l=2 m=2 spectrum
    psi4_l2m2_real_frequency, psi4_l2m2_real_omega, psi4_l2m2_real_omega_spectrem                                                                \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m2_real2[detector_number_i],       apply_window=True, zero_pad_factor=4 )
    psi4_l2m2_imaginary_frequency, psi4_l2m2_imaginary_omega, psi4_l2m2_imaginary_omega_spectrem                                                 \
        = compute_frequency_spectrum( time2[detector_number_i], psi4_l2m2_imaginary2[detector_number_i],  apply_window=True, zero_pad_factor=4 )


    # Compute detector distance from input parameters
    Detector_Interval   = ( input_data.Detector_Rmax - input_data.Detector_Rmin ) / ( input_data.Detector_Number - 1 )
    Detector_Distance_R = input_data.Detector_Rmax - Detector_Interval * detector_number_i
    
    #################################################

    ## Set minimum cutoff frequency for frequency-domain integration

    ## Create output file to record frequency-domain cutoff values

    file_cut_path = os.path.join( figure_outdir, "frequency_cut.txt" )
    file_cut      = open( file_cut_path, "w" )
    
    ## Compute total mass of the system and output
    
    total_mass    = 0.0
    puncture_mass = numpy.zeros( input_data.puncture_number )
    
    ## For 'Ansorg-TwoPuncture' initial data: normalize masses of the first two black holes
    if ( input_data.Initial_Data_Method == "Ansorg-TwoPuncture" ):
        mass_ratio_Q = input_data.parameter_BH[0,0] / input_data.parameter_BH[1,0]
        BBH_M1 = mass_ratio_Q / ( 1.0 + mass_ratio_Q )
        BBH_M2 = 1.0          / ( 1.0 + mass_ratio_Q )
        for k in range( input_data.puncture_number ):
            if ( k == 0 ):
                puncture_mass[k] = BBH_M1 
            elif( k == 1 ):
                puncture_mass[k] = BBH_M2 
            else: 
                puncture_mass[k] = input_data.parameter_BH[k,0]
            total_mass += puncture_mass[k]
     
    ## For other initial-data methods: read puncture masses from input
    else:
        for k in range( input_data.puncture_number ):
            puncture_mass[k] = input_data.parameter_BH[k,0]
            total_mass += puncture_mass[k]
            
    ## Output total mass
    print( file=file_cut )
    
    for k in range( input_data.puncture_number ):
        print( f" mass[{k}] = {puncture_mass[k]} ", file=file_cut )
 
    print( file=file_cut )
        
    print( f" total mass = {total_mass} ", file=file_cut )
    print(                                 file=file_cut )

    ## Compute and output pairwise puncture distances
    puncture_distance = numpy.zeros( (input_data.puncture_number, input_data.puncture_number) )
    puncture_position = input_data.position_BH

    ## Compute pairwise puncture separations
    for k1 in range(input_data.puncture_number):
        for k2 in range(input_data.puncture_number):
            if (k1 != k2):
                puncture_distance[k1,k2] = (   ( puncture_position[k1,0] - puncture_position[k2,0] )**2           \
                                             + ( puncture_position[k1,1] - puncture_position[k2,1] )**2           \
                                             + ( puncture_position[k1,2] - puncture_position[k2,2] )**2  )**(0.5)
                print( f" puncture distance r[{k1,k2}] = {puncture_distance[k1,k2]} ", file=file_cut )
                print(                                                                 file=file_cut )
            ## If k1 == k2, avoid zero-distance artifacts on the diagonal
            else:
                puncture_distance[k1,k2] = (   ( puncture_position[0,0] - puncture_position[0,0] )**2           \
                                             + ( puncture_position[0,1] - puncture_position[0,1] )**2           \
                                             + ( puncture_position[0,2] - puncture_position[0,2] )**2  )**(0.5)

    print( file=file_cut )

    ## Estimate orbital periods and frequencies using Newtonian approximation
    orbital_period    = numpy.zeros( (input_data.puncture_number, input_data.puncture_number) )
    orbital_frequency = numpy.zeros( (input_data.puncture_number, input_data.puncture_number) )

    ## Estimate maximum orbital frequency using Newtonian approximation
    frequency_max = ( numpy.max(puncture_distance) / numpy.min( puncture_mass ) )**(0.5)

    ## Estimate orbital period and frequency for each pair using Newtonian mechanics
    for k1 in range(input_data.puncture_number):
        for k2 in range(input_data.puncture_number):
            if (k1 != k2):
                orbital_period[k1,k2]    = 2.0 * math.pi * ( puncture_distance[k1,k2]**3 / ( puncture_mass[k1] + puncture_mass[k2] ) )**(0.5)
                orbital_frequency[k1,k2] = 1.0 / orbital_period[k1,k2]
                print( f" orbital period estimate:    T_orbital[{k1,k2}] = {orbital_period[k1,k2]} ",    file=file_cut )
                print( f" orbital frequency estimate: f_orbital[{k1,k2}] = {orbital_frequency[k1,k2]} ", file=file_cut )
                print(                                                                                   file=file_cut )
            else:
                orbital_frequency[k1,k2] = frequency_max
                orbital_period[k1,k2]    = 1.0 / orbital_frequency[k1,k2]

    print( file=file_cut )

    ## Set minimum frequency cutoff based on orbital estimate
    orbital_frequency_min       = numpy.min( orbital_frequency )
    gravitational_frequency_min = 2.0 * orbital_frequency_min     ## GW frequency ~ 2 * orbital frequency for quadrupole
    print( " Orbital frequency estimate:            f_orbital_min =", orbital_frequency_min, file=file_cut )
    print( " Gravitational Wave frequency estimate: f_GW_min      =", orbital_frequency_min, file=file_cut )
    print(                                                                                   file=file_cut )

    ## Set minimum frequency cutoff based on orbital estimate
    frequency_cut = gravitational_frequency_min
    omega_cut     = 2.0 * math.pi * frequency_cut
    print( " Frequency Cut estimate: frequency_cut =", frequency_cut, file=file_cut )
    print( " Omega Cut estimate:     omega_cut     =", omega_cut,     file=file_cut )
    print(                                                            file=file_cut )

    ## Manual cutoff setting (deprecated)
    ## omega_cut = 2.0 * math.pi / 100.0

    #################################################

    ## Set tortoise coordinate (r*) for waveform retarded-time correction
    tortoise_R = Detector_Distance_R + 2.0 * total_mass * math.log( Detector_Distance_R / (2.0*total_mass) - 1.0)
    
    ## For more than two punctures, tortoise coordinate is ambiguous; use detector radius
    if ( input_data.puncture_number > 2 ):
        tortoise_R = Detector_Distance_R

    ## Set cutoff based on instantaneous frequency of the Psi4 signal
    ## Abandoned due to large errors
    '''
    ## Set initial time
    t1 = tortoise_R

    ## Compute instantaneous frequency of Psi4 signals
    ## instantaneous_frequency_psi4_l2m2_real = instantaneous_frequency( psi4_l2m2_real2[detector_number_i], len(psi4_l2m2_real2[detector_number_i]) )
    instantaneous_frequency_psi4_l2m2m_real = get_frequency_at_t1( psi4_l2m2m_real2[detector_number_i], len(psi4_l2m2m_real2[detector_number_i]), t1 ) / (2.0*math.pi)
    instantaneous_frequency_psi4_l2m1m_real = get_frequency_at_t1( psi4_l2m1m_real2[detector_number_i], len(psi4_l2m1m_real2[detector_number_i]), t1 ) / (2.0*math.pi)
    instantaneous_frequency_psi4_l2m0_real  = get_frequency_at_t1( psi4_l2m0_real2[detector_number_i],  len(psi4_l2m0_real2[detector_number_i]),  t1 ) / (2.0*math.pi)
    instantaneous_frequency_psi4_l2m1_real  = get_frequency_at_t1( psi4_l2m1_real2[detector_number_i],  len(psi4_l2m1_real2[detector_number_i]),  t1 ) / (2.0*math.pi)
    instantaneous_frequency_psi4_l2m2_real  = get_frequency_at_t1( psi4_l2m2_real2[detector_number_i],  len(psi4_l2m2_real2[detector_number_i]),  t1 ) / (2.0*math.pi)
    print( f" Instantaneous frequency at t - r* = 0, l=2 m=-2 psi4_real = {instantaneous_frequency_psi4_l2m2m_real:.2f} 1/M" )
    print( f" Instantaneous frequency at t - r* = 0, l=2 m=-1 psi4_real = {instantaneous_frequency_psi4_l2m1m_real:.2f} 1/M" )
    print( f" Instantaneous frequency at t - r* = 0, l=2 m=0  psi4_real = {instantaneous_frequency_psi4_l2m0_real:.2f}  1/M" )
    print( f" Instantaneous frequency at t - r* = 0, l=2 m=1  psi4_real = {instantaneous_frequency_psi4_l2m1_real:.2f}  1/M" )
    print( f" Instantaneous frequency at t - r* = 0, l=2 m=2  psi4_real = {instantaneous_frequency_psi4_l2m2_real:.2f}  1/M" )

    ## Add frequency cutoffs based on instantaneous frequency
    frequency_cut_l2m2m = abs( instantaneous_frequency_psi4_l2m2m_real ) * 1.5
    frequency_cut_l2m1m = abs( instantaneous_frequency_psi4_l2m1m_real ) * 1.5
    frequency_cut_l2m0  = abs( instantaneous_frequency_psi4_l2m0_real  ) * 1.5
    frequency_cut_l2m1  = abs( instantaneous_frequency_psi4_l2m1_real  ) * 1.5
    frequency_cut_l2m2  = abs( instantaneous_frequency_psi4_l2m2_real  ) * 1.5
    
    ## Add frequency-domain filter conditions
    omega_cut_l2m2m = 2.0 * math.pi / frequency_cut_l2m2m
    omega_cut_l2m1m = 2.0 * math.pi / frequency_cut_l2m1m
    omega_cut_l2m0  = 2.0 * math.pi / frequency_cut_l2m0
    omega_cut_l2m1  = 2.0 * math.pi / frequency_cut_l2m1
    omega_cut_l2m2  = 2.0 * math.pi / frequency_cut_l2m2

    if (omega_cut_l2m0 < omega_cut_l2m2):
        omega_cut_l2m0 = omega_cut_l2m2
    if (omega_cut_l2m1 < omega_cut_l2m2):
        omega_cut_l2m1 = omega_cut_l2m2
    if (omega_cut_l2m1m < omega_cut_l2m2):
        omega_cut_l2m1m = omega_cut_l2m2
    if (omega_cut_l2m2m < omega_cut_l2m2):
        omega_cut_l2m2m = omega_cut_l2m2
    '''
    
    ## Obtain integrand for inverse Fourier transform
    psi4_l2m2m_real_omega_integration      = frequency_filter_integration( psi4_l2m2m_real_omega,      psi4_l2m2m_real_omega_spectrem,      omega_cut )
    psi4_l2m2m_imaginary_omega_integration = frequency_filter_integration( psi4_l2m2m_imaginary_omega, psi4_l2m2m_imaginary_omega_spectrem, omega_cut )
    psi4_l2m1m_real_omega_integration      = frequency_filter_integration( psi4_l2m1m_real_omega,      psi4_l2m1m_real_omega_spectrem,      omega_cut )
    psi4_l2m1m_imaginary_omega_integration = frequency_filter_integration( psi4_l2m1m_imaginary_omega, psi4_l2m1m_imaginary_omega_spectrem, omega_cut )
    psi4_l2m0_real_omega_integration       = frequency_filter_integration( psi4_l2m0_real_omega,       psi4_l2m0_real_omega_spectrem,       omega_cut )
    psi4_l2m0_imaginary_omega_integration  = frequency_filter_integration( psi4_l2m0_imaginary_omega,  psi4_l2m0_imaginary_omega_spectrem,  omega_cut )
    psi4_l2m1_real_omega_integration       = frequency_filter_integration( psi4_l2m1_real_omega,       psi4_l2m1_real_omega_spectrem,       omega_cut )
    psi4_l2m1_imaginary_omega_integration  = frequency_filter_integration( psi4_l2m1_imaginary_omega,  psi4_l2m1_imaginary_omega_spectrem,  omega_cut )
    psi4_l2m2_real_omega_integration       = frequency_filter_integration( psi4_l2m2_real_omega,       psi4_l2m2_real_omega_spectrem,       omega_cut )
    psi4_l2m2_imaginary_omega_integration  = frequency_filter_integration( psi4_l2m2_imaginary_omega,  psi4_l2m2_imaginary_omega_spectrem,  omega_cut )

    ## Perform inverse Fourier transform in frequency domain to obtain gravitational-wave strain amplitudes
    ## l=2 m=-2 amplitude
    time_grid_h_plus_l2m2m, GW_h_plus_l2m2m \
        = inverse_fourier_transform( psi4_l2m2m_real_omega, psi4_l2m2m_real_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )  
    time_grid_h_cross_l2m2m, GW_h_cross_l2m2m \
        = inverse_fourier_transform( psi4_l2m2m_imaginary_omega, psi4_l2m2m_imaginary_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )
    ## l=2 m=-1 amplitude
    time_grid_h_plus_l2m1m, GW_h_plus_l2m1m \
        = inverse_fourier_transform( psi4_l2m1m_real_omega, psi4_l2m1m_real_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )  
    time_grid_h_cross_l2m1m, GW_h_cross_l2m1m \
        = inverse_fourier_transform( psi4_l2m1m_imaginary_omega, psi4_l2m1m_imaginary_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )
    ## l=2 m=0 amplitude
    time_grid_h_plus_l2m0, GW_h_plus_l2m0 \
        = inverse_fourier_transform( psi4_l2m0_real_omega, psi4_l2m0_real_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )  
    time_grid_h_cross_l2m0, GW_h_cross_l2m0 \
        = inverse_fourier_transform( psi4_l2m0_imaginary_omega, psi4_l2m0_imaginary_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )
    ## l=2 m=1 amplitude
    time_grid_h_plus_l2m1, GW_h_plus_l2m1 \
        = inverse_fourier_transform( psi4_l2m1_real_omega, psi4_l2m1_real_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )  
    time_grid_h_cross_l2m1, GW_h_cross_l2m1 \
        = inverse_fourier_transform( psi4_l2m1_imaginary_omega, psi4_l2m1_imaginary_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )
    ## l=2 m=2 amplitude
    time_grid_h_plus_l2m2, GW_h_plus_l2m2 \
        = inverse_fourier_transform( psi4_l2m2_real_omega, psi4_l2m2_real_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )  
    time_grid_h_cross_l2m2, GW_h_cross_l2m2 \
        = inverse_fourier_transform( psi4_l2m2_imaginary_omega, psi4_l2m2_imaginary_omega_integration, sampling_factor=2, original_zero_pad_factor=4 )  

    
    # Construct time grids for computing gravitational-wave strain h
    # time_max = max( time2[detector_number_i] )
    # time_grid = numpy.linspace( tortoise_R, time_max, 2000 )
    # time_grid_new = numpy.linspace( 0, time_max-tortoise_R, len(time_grid) )  ## subtract tortoise coordinate from times
    # l=2 m=-2
    time_grid_h_plus_l2m2m_new  = time_grid_h_plus_l2m2m - tortoise_R
    time_grid_h_cross_l2m2m_new = time_grid_h_cross_l2m2m - tortoise_R
    # l=2 m=-1
    time_grid_h_plus_l2m1m_new  = time_grid_h_plus_l2m1m - tortoise_R
    time_grid_h_cross_l2m1m_new = time_grid_h_cross_l2m1m - tortoise_R
    # l=2 m=0
    time_grid_h_plus_l2m0_new  = time_grid_h_plus_l2m0 - tortoise_R
    time_grid_h_cross_l2m0_new = time_grid_h_cross_l2m0 - tortoise_R
    # l=2 m=1
    time_grid_h_plus_l2m1_new  = time_grid_h_plus_l2m1 - tortoise_R
    time_grid_h_cross_l2m1_new = time_grid_h_cross_l2m1 - tortoise_R
    # l=2 m=2
    time_grid_h_plus_l2m2_new  = time_grid_h_plus_l2m2 - tortoise_R
    time_grid_h_cross_l2m2_new = time_grid_h_cross_l2m2 - tortoise_R

    plt.figure( figsize=(8,8) )                                   ## figsize controls figure size
    plt.title( f" Gravitational Wave h   Detector Distance = { Detector_Distance_R } ", fontsize=18 )   ## fontsize controls text size
    plt.plot( time_grid_h_plus_l2m0_new, GW_h_plus_l2m0,  \
              color='red',    label="l=2 m=0 h+",                  linewidth=2 )
    plt.plot( time_grid_h_cross_l2m0_new, GW_h_cross_l2m0, \
              color='orange', label="l=2 m=0 hx",  linestyle='--', linewidth=2 )
    plt.plot( time_grid_h_plus_l2m1_new, GW_h_plus_l2m1,  \
              color='green',  label="l=2 m=1 h+",                  linewidth=2 )
    plt.plot( time_grid_h_cross_l2m1_new, GW_h_cross_l2m1, \
              color='cyan',   label="l=2 m=1 hx",  linestyle='--', linewidth=2 )
    plt.plot( time_grid_h_plus_l2m2_new, GW_h_plus_l2m2,  \
              color='black',  label="l=2 m=2 h+",                  linewidth=2 )
    plt.plot( time_grid_h_cross_l2m2_new, GW_h_cross_l2m2, \
              color='gray',   label="l=2 m=2 hx",  linestyle='--', linewidth=2 )
    if ( input_data.puncture_number > 2 ):
        plt.xlabel( "T - R [M]",  fontsize=16     )
    else:
        plt.xlabel( "T - R* [M]", fontsize=16     )
    plt.ylabel( r"R*h",           fontsize=16     )
    plt.xlim( 0.0, max(time_grid_h_plus_l2m0_new) )
    plt.legend( loc='upper right'                 )
    plt.grid(   color='gray', linestyle='--', linewidth=0.5 )  # show grid lines
    plt.savefig( os.path.join(figure_outdir, "Gravitational_Wave_h_Detector_" + str(detector_number_i) + ".pdf") )
    

    print( "Gravitational-wave plot for detector no.", detector_number_i, "finished.")
    print()

    if ( detector_number_i == (input_data.Detector_Number-1) ):
        print( "All gravitational-wave strain amplitude plots finished." )
        print()
    
    '''
    # The following block performs direct time-domain integration of Psi4.
    # This method was deprecated because of insufficient accuracy.
    # h = int_{0}^{t} dt' int_{0}^{t"} Psi4(t") dt"

    # Interpolate per-detector data to obtain smooth functions
    # Use cubic spline interpolation
    psi4_l2m2m_real2_interpolation      = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m2m_real2[detector_number_i],      kind='cubic' )
    psi4_l2m2m_imaginary2_interpolation = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m2m_imaginary2[detector_number_i], kind='cubic' )
    psi4_l2m1m_real2_interpolation      = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m1m_real2[detector_number_i],      kind='cubic' )
    psi4_l2m1m_imaginary2_interpolation = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m1m_imaginary2[detector_number_i], kind='cubic' )
    psi4_l2m0_real2_interpolation       = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m0_real2[detector_number_i],       kind='cubic' )
    psi4_l2m0_imaginary2_interpolation  = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m0_imaginary2[detector_number_i],  kind='cubic' )
    psi4_l2m1_real2_interpolation       = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m1_real2[detector_number_i],       kind='cubic' )
    psi4_l2m1_imaginary2_interpolation  = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m1_imaginary2[detector_number_i],  kind='cubic' )
    psi4_l2m2_real2_interpolation       = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m2_real2[detector_number_i],       kind='cubic' )
    psi4_l2m2_imaginary2_interpolation  = scipy.interpolate.interp1d( time2[detector_number_i], psi4_l2m2_imaginary2[detector_number_i],  kind='cubic' )

    # Compute detector distance from input parameters
    Detector_Interval   = ( input_data.Detector_Rmax - input_data.Detector_Rmin ) / ( input_data.Detector_Number - 1 )
    Detector_Distance_R = input_data.Detector_Rmax - Detector_Interval * detector_number_i

    # Set tortoise coordinate
    tortoise_R = Detector_Distance_R + 2.0 * total_mass * math.log( Detector_Distance_R / (2.0*total_mass) - 1.0)
    
    # Construct time grid for gravitational-wave amplitude h
    time_max = max( time2[detector_number_i] )
    time_grid = numpy.linspace( tortoise_R, time_max, 2000 )
    time_grid_new = numpy.linspace( 0, time_max-tortoise_R, len(time_grid) )  # subtract tortoise coordinate

    GW_h_plus_l2m2m  = numpy.zeros( len(time_grid) )
    GW_h_cross_l2m2m = numpy.zeros( len(time_grid) )
    GW_h_plus_l2m1m  = numpy.zeros( len(time_grid) )
    GW_h_cross_l2m1m = numpy.zeros( len(time_grid) )
    GW_h_plus_l2m0   = numpy.zeros( len(time_grid) )
    GW_h_cross_l2m0  = numpy.zeros( len(time_grid) )
    GW_h_plus_l2m1   = numpy.zeros( len(time_grid) )
    GW_h_cross_l2m1  = numpy.zeros( len(time_grid) )
    GW_h_plus_l2m2   = numpy.zeros( len(time_grid) )
    GW_h_cross_l2m2  = numpy.zeros( len(time_grid) )

    # Solve for h by double numerical integration: h = int_{0}^{t} dt' int_{0}^{t"} Psi4(t") dt" 
    # The double integral can be reordered and simplified to h = int_{0}^{t} (t-t") Psi4(t") dt"
    def GW_h_plus_l2m2m_integrand(t, tmax):
        return psi4_l2m2m_real2_interpolation(t) * (tmax-t)
    def GW_h_cross_l2m2m_integrand(t, tmax):
        return psi4_l2m2m_imaginary2_interpolation(t) * (tmax-t)
    def GW_h_plus_l2m1m_integrand(t, tmax):
        return psi4_l2m1m_real2_interpolation(t) * (tmax-t)
    def GW_h_cross_l2m1m_integrand(t, tmax):
        return psi4_l2m1m_imaginary2_interpolation(t) * (tmax-t)
    def GW_h_plus_l2m0_integrand(t, tmax):
        return psi4_l2m0_real2_interpolation(t) * (tmax-t)
    def GW_h_cross_l2m0_integrand(t, tmax):
        return psi4_l2m0_imaginary2_interpolation(t) * (tmax-t)
    def GW_h_plus_l2m1_integrand(t, tmax):
        return psi4_l2m1_real2_interpolation(t) * (tmax-t)
    def GW_h_cross_l2m1_integrand(t, tmax):
        return psi4_l2m1_imaginary2_interpolation(t) * (tmax-t)
    def GW_h_plus_l2m2_integrand(t, tmax):
        return psi4_l2m2_real2_interpolation(t) * (tmax-t)
    def GW_h_cross_l2m2_integrand(t, tmax):
        return psi4_l2m2_imaginary2_interpolation(t) * (tmax-t)
    
    # Compute gravitational-wave strains h+ and hx
    # Redefine integrand with a lambda so it becomes a single-variable function for integration

    for j in range( len(time_grid) ):

        print( " j = ", j )

        GW_h_plus_l2m2m_integrand2 = lambda t: GW_h_plus_l2m2m_integrand(t, time_grid[j])
        ## Note: scipy.integrate.quad returns a tuple (value, error)
        GW_h_plus_l2m2m[j], err0 = scipy.integrate.quad( GW_h_plus_l2m2m_integrand2, 0.0, time_grid[j], limit=600 )
                               # epsabs=1e-8,  # absolute tolerance
                               # limit=600 )    # increase number of subintervals

        GW_h_cross_l2m2m_integrand2 = lambda t: GW_h_cross_l2m2m_integrand(t, time_grid[j])
        GW_h_cross_l2m2m[j], err0 = scipy.integrate.quad( GW_h_cross_l2m2m_integrand2, 0.0, time_grid[j], limit=600 )

        GW_h_plus_l2m1m_integrand2 = lambda t: GW_h_plus_l2m1m_integrand(t, time_grid[j])
        GW_h_plus_l2m1m[j], err0 = scipy.integrate.quad( GW_h_plus_l2m1m_integrand2, 0.0, time_grid[j], limit=600 )

        GW_h_cross_l2m1m_integrand2 = lambda t: GW_h_cross_l2m1m_integrand(t, time_grid[j])
        GW_h_cross_l2m1m[j], err0 = scipy.integrate.quad( GW_h_cross_l2m1m_integrand2, 0.0, time_grid[j], limit=600 )

        GW_h_plus_l2m0_integrand2 = lambda t: GW_h_plus_l2m0_integrand(t, time_grid[j])
        GW_h_plus_l2m0[j], err0 = scipy.integrate.quad( GW_h_plus_l2m0_integrand2, 0.0, time_grid[j], limit=600 )

        GW_h_cross_l2m0_integrand2 = lambda t: GW_h_cross_l2m0_integrand(t, time_grid[j])
        GW_h_cross_l2m0[j], err0 = scipy.integrate.quad( GW_h_cross_l2m0_integrand2, 0.0, time_grid[j], limit=600 )

        GW_h_plus_l2m1_integrand2 = lambda t: GW_h_plus_l2m1_integrand(t, time_grid[j])
        GW_h_plus_l2m1[j], err0 = scipy.integrate.quad( GW_h_plus_l2m1_integrand2, 0.0, time_grid[j], limit=600 )

        GW_h_cross_l2m1_integrand2 = lambda t: GW_h_cross_l2m1_integrand(t, time_grid[j])
        GW_h_cross_l2m1[j], err0 = scipy.integrate.quad( GW_h_cross_l2m1_integrand2, 0.0, time_grid[j], limit=600 )

        GW_h_plus_l2m2_integrand2 = lambda t: GW_h_plus_l2m2_integrand(t, time_grid[j])
        GW_h_plus_l2m2[j], err0 = scipy.integrate.quad( GW_h_plus_l2m2_integrand2, 0.0, time_grid[j], limit=600 )

        GW_h_cross_l2m2_integrand2 = lambda t: GW_h_cross_l2m2_integrand(t, time_grid[j])
        GW_h_cross_l2m2[j], err0 = scipy.integrate.quad( GW_h_cross_l2m2_integrand2, 0.0, time_grid[j], limit=600 )
            
    # Computation of gravitational-wave amplitudes h+ and hx complete

    # Now perform plotting
    plt.figure( figsize=(8,8) )                                   ## figsize controls figure size
    plt.title( f" Gravitational Wave h   Detector Distance = { Detector_Distance_R } ", fontsize=18 )   ## fontsize controls text size
    plt.plot( time_grid_new, GW_h_plus_l2m0,  \
              color='red',    label="l=2 m=0 h+",                  linewidth=2 )
    plt.plot( time_grid_new, GW_h_cross_l2m0, \
              color='orange', label="l=2 m=0 hx",  linestyle='--', linewidth=2 )
    plt.plot( time_grid_new, GW_h_plus_l2m1,  \
              color='green',  label="l=2 m=1 h+",                  linewidth=2 )
    plt.plot( time_grid_new, GW_h_cross_l2m1, \
              color='cyan',   label="l=2 m=1 hx",  linestyle='--', linewidth=2 )
    plt.plot( time_grid_new, GW_h_plus_l2m2,  \
              color='black',  label="l=2 m=2 h+",                  linewidth=2 )
    plt.plot( time_grid_new, GW_h_cross_l2m2, \
              color='gray',   label="l=2 m=2 hx",  linestyle='--', linewidth=2 )
    plt.xlabel( "T [M]",          fontsize=16 )
    plt.ylabel( r"R*h",           fontsize=16 )
    plt.legend( loc='upper right'             )
    plt.savefig( os.path.join(figure_outdir, "Gravitational_Wave_h_Detector_" + str(detector_number_i) + ".pdf") )
    
    print()
    print( "Gravitational-wave plot for detector no.", detector_number_i, "finished.")
    print( "Plotting of gravitational-wave strain amplitude h completed.")
    print()
    '''

    return

####################################################################################



####################################################################################

## Standalone usage example
'''
## outdir = "./BBH_q=1"
outdir = "./3BH"
for i in range( input_data.Detector_Number ):
    generate_gravitational_wave_amplitude_plot(outdir, outdir, i)
'''
####################################################################################



