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

import os
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


def _mpiexec_argv():
    """Resolve the MPI launcher from AMSS_MPIEXEC (default: mpiexec).

    The value is split with shlex so users can pass extra flags, e.g.
    AMSS_MPIEXEC="mpiexec --oversubscribe"."""
    return shlex.split(os.environ.get("AMSS_MPIEXEC", "mpiexec"))


def _env_tokens():
    """Build env VAR=VALUE tokens forwarded to each MPI rank.

    Uses `mpiexec -n N env VAR=... exe` form, which works on both OpenMPI
    and MPICH (the -x flag was OpenMPI-only)."""
    tokens = []
    for var in ("OMP_NUM_THREADS", "LD_LIBRARY_PATH"):
        if var in os.environ:
            tokens.append(f"{var}={os.environ[var]}")
    return tokens


def run_TwoPunctureABE():
    """Run the TwoPuncture initial-data solver in the current directory."""
    print("\n Running the AMSS-NCKU executable file TwoPunctureABE\n")
    cmd = "./TwoPunctureABE < /dev/null"
    _run_and_tee(cmd, "TwoPunctureABE_out.log")
    print("\n The TwoPunctureABE simulation is finished\n")


def run_ABE():
    """Run the main ABE evolution executable via mpiexec."""
    if input_data.GPU_Calculation == "no":
        exe, log = "./ABE", "ABE_out.log"
    elif input_data.GPU_Calculation == "yes":
        exe, log = "./ABEGPU", "ABEGPU_out.log"
    else:
        raise ValueError("GPU_Calculation must be 'no' or 'yes'")

    print(f"\n Running {exe} with {input_data.MPI_processes} MPI ranks "
          f"and OMP_NUM_THREADS={input_data.OMP_threads}\n")

    # Build the launcher command. `env VAR=... exe` form works on both
    # OpenMPI and MPICH; the original -x flag was OpenMPI-only.
    argv = (_mpiexec_argv()
            + ["-n", str(input_data.MPI_processes)]
            + ["env"] + _env_tokens()
            + [exe])
    cmd = shlex.join(argv) + " < /dev/null"

    _run_and_tee(cmd, log)

    print(f"\n The {exe} simulation is finished\n")
