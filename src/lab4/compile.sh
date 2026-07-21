#!/usr/bin/env bash
# compile.sh — portable build driver for AMSS-NCKU Lab 4.
#
# Toolchain comes from environment (CC/CXX/FC/MPI_CXX_COMPILER/CUDACXX) or
# PATH (mpicxx/mpifort). AMSS_* vars are forwarded to CMake only when set.
# Build directory: AMSS_BUILD_DIR (default <lab-root>/build), relative to
# lab root. Extra args (e.g. -DAMSS_ENABLE_GPU=ON) override at the end.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAKE="${CMAKE:-cmake}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"

# Resolve build directory relative to the lab root.
BUILD_DIR="${AMSS_BUILD_DIR:-$ROOT_DIR/build}"
case "$BUILD_DIR" in
  /*) : ;;
   *) BUILD_DIR="$ROOT_DIR/$BUILD_DIR" ;;
esac

cmake_args=()
# MPI wrapper wins over plain CXX — the wrapper carries the right MPI flags
# and include paths. Setting both to different values is almost always a
# mistake, so warn instead of silently letting CXX override the wrapper.
if [[ -n "${MPI_CXX_COMPILER:-}" && -n "${CXX:-}" && "${MPI_CXX_COMPILER}" != "${CXX}" ]]; then
  echo "warning: both MPI_CXX_COMPILER ($MPI_CXX_COMPILER) and CXX ($CXX) are set;" >&2
  echo "         using MPI_CXX_COMPILER for the C++ compiler (MPI wrapper)." >&2
fi
if [[ -n "${MPI_CXX_COMPILER:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_COMPILER=$MPI_CXX_COMPILER")
elif [[ -n "${CXX:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_COMPILER=$CXX")
fi
[[ -n "${FC:-}" ]]                && cmake_args+=("-DCMAKE_Fortran_COMPILER=$FC")
[[ -n "${CUDACXX:-}" ]]           && cmake_args+=("-DCMAKE_CUDA_COMPILER=$CUDACXX")
[[ -n "${AMSS_ENABLE_GPU:-}" ]]    && cmake_args+=("-DAMSS_ENABLE_GPU=$AMSS_ENABLE_GPU")
[[ -n "${AMSS_CUDA_ARCHITECTURES:-}" ]] && cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$AMSS_CUDA_ARCHITECTURES")
[[ -n "${AMSS_ARCH_FLAGS:-}" ]]    && cmake_args+=("-DAMSS_ARCH_FLAGS=$AMSS_ARCH_FLAGS")
[[ -n "${AMSS_ENABLE_OPENMP:-}" ]] && cmake_args+=("-DAMSS_ENABLE_OPENMP=$AMSS_ENABLE_OPENMP")
[[ -n "${AMSS_MPI_CUDA_AWARE:-}" ]] && cmake_args+=("-DAMSS_MPI_CUDA_AWARE=$AMSS_MPI_CUDA_AWARE")

echo "==> Configure: $CMAKE -S \"$ROOT_DIR\" -B \"$BUILD_DIR\" ${cmake_args[*]:-} $*"
"$CMAKE" -S "$ROOT_DIR" -B "$BUILD_DIR" "${cmake_args[@]}" "$@"

echo "==> Build: $CMAKE --build \"$BUILD_DIR\" -j $JOBS"
"$CMAKE" --build "$BUILD_DIR" -j "$JOBS"

echo "==> Built executables:"
ls -lh "$BUILD_DIR/TwoPunctureABE" "$BUILD_DIR/ABE"
if [[ -f "$BUILD_DIR/ABEGPU" ]]; then
  ls -lh "$BUILD_DIR/ABEGPU"
fi
