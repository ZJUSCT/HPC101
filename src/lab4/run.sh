#!/usr/bin/env bash
set -euo pipefail

source /home/jjsnam/anaconda3/etc/profile.d/conda.sh
conda activate AMSS

# Ansorg-TwoPuncture initialization allocates large Fortran automatic arrays.
# case3's 128x128x64 blocks exceed the default 8 MB stack on many shells.
ulimit -s unlimited

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DEFAULT_MPI_BIN="/home/jjsnam/spack/opt/spack/linux-icelake/openmpi-5.0.9-nd3t2hacw7x7xkmrtqkzr4gxxr54isly/bin"
DEFAULT_RUNTIME_LIBS=(
  "/home/jjsnam/spack/opt/spack/linux-icelake/intel-oneapi-compilers-2025.3.2-4bn4aku7wecqcqv47wdlqxn2pkvtuqwu/compiler/2025.3/lib"
  "/home/jjsnam/spack/opt/spack/linux-icelake/openmpi-5.0.9-nd3t2hacw7x7xkmrtqkzr4gxxr54isly/lib"
  "/home/ckyasb/miniconda3/lib"
)

cd "$ROOT_DIR"

if ! command -v mpirun >/dev/null 2>&1 && [[ -x "$DEFAULT_MPI_BIN/mpirun" ]]; then
  export PATH="$DEFAULT_MPI_BIN:$PATH"
fi

for lib_dir in "${DEFAULT_RUNTIME_LIBS[@]}"; do
  if [[ -d "$lib_dir" && ":${LD_LIBRARY_PATH:-}:" != *":$lib_dir:"* ]]; then
    export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done

if [[ "${1:-}" == "--twop-cache" ]]; then
  export AMSS_NCKU_TWOP_CACHE=1
  shift
fi

if (( $# > 0 )); then
  echo "usage: ./run.sh [--twop-cache]" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import os
import sys
import AMSS_NCKU_Input as cfg

if cfg.GPU_Calculation == "yes":
    exe = "ABEGPU"
elif cfg.GPU_Calculation == "no":
    exe = "ABE"
else:
    print("error: GPU_Calculation must be 'yes' or 'no'", file=sys.stderr)
    sys.exit(1)
missing = [
    path for path in (
        os.path.join("build", "TwoPunctureABE"),
        os.path.join("build", exe),
    )
    if not os.path.isfile(path)
]
if missing:
    print("error: missing executable(s):", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    print("run ./compile.sh first", file=sys.stderr)
    sys.exit(1)

print(f"==> Run mode: GPU_Calculation={cfg.GPU_Calculation!r}, "
      f"MPI_processes={cfg.MPI_processes}, OMP_threads={cfg.OMP_threads}")
print(f"==> Output directory: {cfg.File_directory}/AMSS_NCKU_output/")
PY

"$PYTHON" AMSS_NCKU_Program.py
