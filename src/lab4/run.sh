#!/bin/bash
# run.sh — run the AMSS-NCKU Lab 4 driver.
#
# Path layout (relative values resolve against the lab root):
#   AMSS_BUILD_DIR    build output          (default: <lab4>/build)
#   AMSS_OUTPUT_ROOT  run directory parent  (default: <lab4>)
#   AMSS_CACHE_DIR    TwoPuncture cache root (default: <lab4>/twopuncture_cache)
#   AMSS_MPIEXEC      MPI launcher          (default: mpiexec)
set -euo pipefail

# Ansorg-TwoPuncture allocates large Fortran automatic arrays.
ulimit -s unlimited

# HPC jobs run inside a root container even when submitted by a regular user.
# Open MPI 5.x (prterun) therefore needs its explicit container opt-in. These
# variables are ignored for non-root launches and preserve a caller-supplied
# AMSS_MPIEXEC.
export OMPI_ALLOW_RUN_AS_ROOT="${OMPI_ALLOW_RUN_AS_ROOT:-1}"
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="${OMPI_ALLOW_RUN_AS_ROOT_CONFIRM:-1}"
AMSS_MPIEXEC="${AMSS_MPIEXEC:-mpiexec --allow-run-as-root}"

ROOT_DIR="$(pwd)"
PYTHON="${PYTHON:-python3}"

resolve_under_root() {
  case "$1" in
    /*) printf '%s' "$1" ;;
     *) printf '%s/%s' "$ROOT_DIR" "$1" ;;
  esac
}

AMSS_BUILD_DIR="$(resolve_under_root "${AMSS_BUILD_DIR:-$ROOT_DIR/build}")"
AMSS_OUTPUT_ROOT="$(resolve_under_root "${AMSS_OUTPUT_ROOT:-$ROOT_DIR}")"
AMSS_CACHE_DIR="$(resolve_under_root "${AMSS_CACHE_DIR:-$ROOT_DIR/twopuncture_cache}")"
AMSS_MPIEXEC="${AMSS_MPIEXEC:-mpiexec}"
export AMSS_BUILD_DIR AMSS_OUTPUT_ROOT AMSS_CACHE_DIR AMSS_MPIEXEC

if [[ "${1:-}" == "--twop-cache" ]]; then
  export AMSS_NCKU_TWOP_CACHE=1
  shift
fi
if (( $# > 0 )); then
  echo "usage: ./run.sh [--twop-cache]" >&2
  exit 1
fi

echo "==> Build    : $AMSS_BUILD_DIR"
echo "==> Output   : $AMSS_OUTPUT_ROOT"
echo "==> Cache    : $AMSS_CACHE_DIR"
echo "==> MPI exec : $AMSS_MPIEXEC"

cd "$ROOT_DIR"
"$PYTHON" AMSS_NCKU_Program.py