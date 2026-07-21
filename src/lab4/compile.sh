#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(nproc)}"
CMAKE="${CMAKE:-cmake}"
DEFAULT_MPI_BIN="/home/jjsnam/spack/opt/spack/linux-icelake/openmpi-5.0.9-nd3t2hacw7x7xkmrtqkzr4gxxr54isly/bin"
DEFAULT_CUDA_NVCC="/usr/local/cuda-13.3/bin/nvcc"

cd "$ROOT_DIR"

if ! command -v "$CMAKE" >/dev/null 2>&1; then
  echo "error: cmake not found on PATH" >&2
  exit 1
fi

# if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] &&
#    { grep -q 'CMAKE_Fortran_COMPILER.*NOTFOUND' "$BUILD_DIR/CMakeCache.txt" ||
#      grep -q 'CMAKE_CXX_COMPILER:.*=/usr/bin/c++' "$BUILD_DIR/CMakeCache.txt"; }; then
#   echo "==> Removing stale CMake cache with invalid compiler settings"
#   rm -f "$BUILD_DIR/CMakeCache.txt"
#   rm -rf "$BUILD_DIR/CMakeFiles"
# fi

cmake_args=()
if [[ -z "${CXX:-}" && -x "$DEFAULT_MPI_BIN/mpicxx" ]]; then
  cmake_args+=("-DCMAKE_CXX_COMPILER=$DEFAULT_MPI_BIN/mpicxx")
fi
if [[ -z "${CC:-}" && -x "$DEFAULT_MPI_BIN/mpicc" ]]; then
  cmake_args+=("-DCMAKE_C_COMPILER=$DEFAULT_MPI_BIN/mpicc")
fi
if [[ -z "${FC:-}" && -x "$DEFAULT_MPI_BIN/mpifort" ]]; then
  cmake_args+=("-DCMAKE_Fortran_COMPILER=$DEFAULT_MPI_BIN/mpifort")
fi
if [[ -z "${CUDACXX:-}" && -x "$DEFAULT_CUDA_NVCC" ]]; then
  cmake_args+=("-DCMAKE_CUDA_COMPILER=$DEFAULT_CUDA_NVCC")
fi

echo "==> Configure: $CMAKE -B $BUILD_DIR -S . ${cmake_args[*]} $*"
"$CMAKE" -B "$BUILD_DIR" -S . "${cmake_args[@]}" "$@"

echo "==> Build: $CMAKE --build $BUILD_DIR -j $JOBS"
"$CMAKE" --build "$BUILD_DIR" -j "$JOBS"

echo "==> Built executables:"
ls -lh "$BUILD_DIR"/TwoPunctureABE "$BUILD_DIR"/ABE "$BUILD_DIR"/ABEGPU
