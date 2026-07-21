#!/usr/bin/env python3
##################################################################
##
## AMSS-NCKU Numerical Relativity -- run driver
##
## This script does NOT build anything. Compile first:
##     cmake -B build -S .
##     cmake --build build -j
## then run:
##     python3 AMSS_NCKU_Program.py
##
## It generates the input parfiles, runs the TwoPuncture initial-data
## solver and the ABE evolution, copies out the result files, and plots.
##
## Fast-debug option: set the environment variable AMSS_NCKU_TWOP_CACHE=1
## to cache the TwoPuncture initial data (keyed by the TwoPuncture input
## parfile) and skip re-running the solver when the inputs are unchanged.
## Default (unset): TwoPuncture always runs.
##
##################################################################

import hashlib
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")          # headless: write figures to files, no display

import AMSS_NCKU_Input as input_data

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
SRC_DIR   = os.path.join(REPO_ROOT, "src")
os.chdir(REPO_ROOT)

##################################################################
## This trimmed lab build supports both CPU (ABE) and GPU (ABEGPU) BSSN
## evolution with Ansorg-TwoPuncture initial data.

if input_data.Equation_Class != "BSSN":
    sys.exit(" This trimmed lab build only supports Equation_Class = 'BSSN'")
if input_data.Initial_Data_Method != "Ansorg-TwoPuncture":
    sys.exit(" This trimmed lab build only supports Initial_Data_Method = 'Ansorg-TwoPuncture'")

if input_data.GPU_Calculation == "no":
    abe_name = "ABE"
elif input_data.GPU_Calculation == "yes":
    abe_name = "ABEGPU"
else:
    sys.exit(" GPU_Calculation in AMSS_NCKU_Input.py must be 'no' or 'yes'")

abe_built  = os.path.join(BUILD_DIR, abe_name)
twop_built = os.path.join(BUILD_DIR, "TwoPunctureABE")
for exe in (abe_built, twop_built):
    if not os.path.isfile(exe):
        sys.exit(f" Missing executable: {exe}\n"
                 f" Build first:  cmake -B build -S .  &&  cmake --build build -j")

## OpenMP threads per MPI rank (inherited by the mpirun child process)
os.environ["OMP_NUM_THREADS"] = str(input_data.OMP_threads)

## TwoPuncture initial-data cache (opt-in, for fast debugging)
TWOP_CACHE   = os.environ.get("AMSS_NCKU_TWOP_CACHE", "") == "1"
CACHE_ROOT   = os.path.join(REPO_ROOT, "twopuncture_cache")
TWOP_OUTPUTS = ("Ansorg.psid", "puncture_parameters_new.txt")

##################################################################
## (Re)create the output directory tree

File_directory = input_data.File_directory
shutil.rmtree(File_directory, ignore_errors=True)
os.mkdir(File_directory)
shutil.copy("AMSS_NCKU_Input.py", File_directory)

output_directory         = os.path.join(File_directory, "AMSS_NCKU_output")
binary_results_directory = os.path.join(output_directory, input_data.Output_directory)
figure_directory         = os.path.join(File_directory, "figure")
os.mkdir(output_directory)
os.mkdir(binary_results_directory)
os.mkdir(figure_directory)
print(" Output directory has been generated\n")

##################################################################
## Generate parameter info and the ABE input parfile

import setup

setup.print_input_data(File_directory)
setup.generate_AMSSNCKU_input()
setup.print_puncture_information()

print("\n Generating the AMSS-NCKU input parfile for the ABE executable.\n")

import numerical_grid

numerical_grid.append_AMSSNCKU_cgh_input()
numerical_grid.plot_initial_grid()

##################################################################
## Stage the executables into the run directory

shutil.copy2(abe_built,  os.path.join(output_directory, abe_name))
shutil.copy2(twop_built, os.path.join(output_directory, "TwoPunctureABE"))

import makefile_and_run

##################################################################
## Run the TwoPuncture initial-data solver (or reuse a cached result)

start_time = time.time()

print("\n Initial data method: Ansorg-TwoPuncture\n")

import generate_TwoPuncture_input
generate_TwoPuncture_input.generate_AMSSNCKU_TwoPuncture_input()

twop_parfile = os.path.join(output_directory, "TwoPunctureinput.par")
shutil.copy2(os.path.join(File_directory, "AMSS-NCKU-TwoPuncture.input"),
             twop_parfile)

cache_dir = None
if TWOP_CACHE:
    with open(twop_parfile, "rb") as f:
        key = hashlib.sha1(f.read()).hexdigest()[:16]
    cache_dir = os.path.join(CACHE_ROOT, key)

cache_hit = (cache_dir is not None and
             all(os.path.isfile(os.path.join(cache_dir, f))
                 for f in TWOP_OUTPUTS))

if cache_hit:
    print(f" TwoPuncture cache hit ({key}); reusing cached initial data")
    for f in TWOP_OUTPUTS:
        shutil.copy2(os.path.join(cache_dir, f),
                     os.path.join(output_directory, f))
else:
    os.chdir(output_directory)
    makefile_and_run.run_TwoPunctureABE()
    os.chdir(REPO_ROOT)
    missing_twop = [f for f in TWOP_OUTPUTS
                    if not os.path.isfile(os.path.join(output_directory, f))]
    if missing_twop:
        sys.exit(" TwoPunctureABE did not produce expected output file(s): "
                 + ", ".join(missing_twop))
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        for f in TWOP_OUTPUTS:
            shutil.copy2(os.path.join(output_directory, f),
                         os.path.join(cache_dir, f))
        print(f" TwoPuncture output cached ({key})")

##################################################################
## Update puncture parameters from the TwoPuncture output, then
## assemble the final ABE input parfile

import renew_puncture_parameter

renew_puncture_parameter.append_AMSSNCKU_BSSN_input(File_directory, output_directory)

shutil.copy2(os.path.join(File_directory, "AMSS-NCKU.input"),
             os.path.join(output_directory, "input.par"))

##################################################################
## Run the ABE evolution

os.chdir(output_directory)
makefile_and_run.run_ABE()
os.chdir(REPO_ROOT)

elapsed_time = time.time() - start_time

##################################################################
## Copy key result files up one level for easy inspection

shutil.copy(os.path.join(binary_results_directory, "setting.par"),
            os.path.join(output_directory, "AMSSNCKU_setting_parameter"))
shutil.copy(os.path.join(binary_results_directory, "Error.log"),
            os.path.join(output_directory, "Error.log"))
for name in ("bssn_BH.dat", "bssn_ADMQs.dat", "bssn_psi4.dat", "bssn_constraint.dat"):
    shutil.copy(os.path.join(binary_results_directory, name),
                os.path.join(output_directory, name))

##################################################################
## Plot the results (non-fatal: the simulation data is already saved)

print("\n Plotting the AMSS-NCKU simulation results\n")
try:
    import plot_xiaoqu
    import plot_GW_strain_amplitude_xiaoqu

    plot_xiaoqu.generate_puncture_orbit_plot(binary_results_directory, figure_directory)
    plot_xiaoqu.generate_puncture_orbit_plot3D(binary_results_directory, figure_directory)
    plot_xiaoqu.generate_puncture_distence_plot(binary_results_directory, figure_directory)

    for i in range(input_data.Detector_Number):
        plot_xiaoqu.generate_gravitational_wave_psi4_plot(
            binary_results_directory, figure_directory, i)
        plot_GW_strain_amplitude_xiaoqu.generate_gravitational_wave_amplitude_plot(
            binary_results_directory, figure_directory, i)

    for i in range(input_data.Detector_Number):
        plot_xiaoqu.generate_ADMmass_plot(binary_results_directory, figure_directory, i)

    for i in range(input_data.grid_level):
        plot_xiaoqu.generate_constraint_check_plot(binary_results_directory, figure_directory, i)

except Exception as exc:                                  # noqa: BLE001
    print(f" WARNING: plotting failed ({type(exc).__name__}: {exc})")
    print(" The simulation result files are still available under "
          f"{output_directory}")

##################################################################

print(f"\n This Program Cost = {elapsed_time} Seconds\n")
print(" The AMSS-NCKU-Python simulation is successfully finished.\n")
