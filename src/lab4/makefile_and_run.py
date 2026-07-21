##################################################################
##
## Run-only helpers for the AMSS-NCKU lab driver.
##
## Both run_ABE and run_TwoPunctureABE invoke the executable via the
## shell with stdin redirected from /dev/null and stdout/stderr streamed
## through tee, so output is visible in the terminal and saved to the
## per-run log file. mpirun therefore owns its file descriptors directly;
## the parent Python just waits for it to fully exit (subprocess.call).
## Earlier code used Popen with stdout=PIPE and iterated the pipe, which
## under OpenMPI 5 caused the iterator to return after the first few
## timesteps with the ABE ranks orphaned.
##
##################################################################

import shlex
import subprocess

import AMSS_NCKU_Input as input_data


def _run_and_tee(cmd, log):
    """Run a shell command and stream output to both terminal and log."""
    quoted_log = shlex.quote(log)
    wrapped = f"set -o pipefail; {cmd} 2>&1 | tee {quoted_log}"
    rc = subprocess.call(["bash", "-c", wrapped])
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def run_TwoPunctureABE():
    """Run the TwoPuncture initial-data solver in the current directory."""
    print("\n Running the AMSS-NCKU executable file TwoPunctureABE\n")
    cmd = "./TwoPunctureABE < /dev/null"
    _run_and_tee(cmd, "TwoPunctureABE_out.log")
    print("\n The TwoPunctureABE simulation is finished\n")


def run_ABE():
    """Run the main ABE evolution executable via mpirun."""
    if input_data.GPU_Calculation == "no":
        exe, log = "./ABE", "ABE_out.log"
    elif input_data.GPU_Calculation == "yes":
        exe, log = "./ABEGPU", "ABEGPU_out.log"
    else:
        raise ValueError("GPU_Calculation must be 'no' or 'yes'")

    print(f"\n Running {exe} with {input_data.MPI_processes} MPI ranks "
          f"and OMP_NUM_THREADS={input_data.OMP_threads}\n")

    # -x OMP_NUM_THREADS exports the env var from the launcher's shell into
    # every MPI rank. Without it, OpenMPI does not forward env vars by
    # default and ranks would default OMP_NUM_THREADS=1 regardless of the
    # input file.
    # -x LD_LIBRARY_PATH exports compiler/MPI runtime library paths needed by
    # Intel-built executables on batch/non-login shells.
    cmd = (f"mpirun -np {input_data.MPI_processes} "
           f"-x OMP_NUM_THREADS -x LD_LIBRARY_PATH {exe} "
           f"< /dev/null")
    _run_and_tee(cmd, log)

    print(f"\n The {exe} simulation is finished\n")
