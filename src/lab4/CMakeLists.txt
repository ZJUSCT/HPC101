cmake_minimum_required(VERSION 3.20)
project(amss_ncku_lab LANGUAGES CXX Fortran)

# ---------------------------------------------------------------------------
# AMSS-NCKU numerical relativity lab — CPU + GPU build.
# Replaces the original makefile / makefile.inc.
# Toolchain on node m602 is provided via: spack load gcc@13.3.0 openmpi@5.0.9
# GPU toolchain: /usr/local/cuda-13.3/bin/nvcc, A100 sm_80
# ---------------------------------------------------------------------------

find_package(MPI REQUIRED COMPONENTS CXX)

# Compile flags (from the original makefile.inc):
#   C++     : <opt> -fno-strict-aliasing -Wno-deprecated -Dfortran3 -Dnewc
#   Fortran : <opt> -fno-strict-aliasing, with C preprocessing
#
# -fno-strict-aliasing is applied unconditionally because the legacy
# AMSS-NCKU code has multiple type-punning / strict-aliasing violations
# that gcc 14 exploits at -O2/-O3, producing an MPI deadlock in
# Parallel::transfer's ghost-zone exchange at evolution step 0. One
# specific instance (uninitialized bssn_class::LastConsOut) is fixed in
# the source, but additional aliasing UB remains: dropping this flag at
# -O3 still deadlocks. It is kept out of the tunable AMSS_OPT so changing
# the optimization level never accidentally loses the safety flag.
#
# The optimization level is configurable for testing:
#   cmake -B build -DAMSS_OPT="-O0"
set(AMSS_OPT "-O3" CACHE STRING "C/C++/Fortran optimization level")
separate_arguments(AMSS_OPT_LIST UNIX_COMMAND "${AMSS_OPT}")
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:${AMSS_OPT_LIST};-fno-strict-aliasing;-Wno-deprecated;-Dfortran3;-Dnewc>"
  "$<$<COMPILE_LANGUAGE:Fortran>:${AMSS_OPT_LIST};-fno-strict-aliasing;-cpp>"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Shared Fortran source list used by both ABE and ABEGPU targets.
set(AMSS_F90SOURCES
  src/bssn_rhs.f90
  src/diff_new.f90 src/enforce_algebra.f90
  src/fadmquantites_bssn.f90 src/fmisc.f90
  src/getnp4.f90
  src/initial_puncture.f90 src/kodiss.f90
  src/lopsidediff.f90
  src/prolongrestrict_cell.f90 src/rungekutta4_rout.f90
  src/sommerfeld_rout.f90
)

# --- ABE: CPU main evolution executable ---------------------------------------
# Source list derived from the original makefile object lists
# (C++FILES + F90FILES + F77FILES + AHFDOBJS). Pruned in Phase 5.
set(ABE_SOURCES
  # C++FILES
  src/ABE.C src/Ansorg.C src/Block.C src/MPatch.C src/Parallel.C
  src/bssn_class.C src/cgh.C src/checkpoint.C
  src/misc.C src/monitor.C src/perf.C
  src/surface_integral.C src/transpbh.C src/var.C
  ${AMSS_F90SOURCES}
)

add_executable(ABE ${ABE_SOURCES})
target_link_libraries(ABE PRIVATE MPI::MPI_CXX gfortran)
target_link_directories(ABE PRIVATE /home/ckyasb/miniconda3/lib)

# --- TwoPunctureABE: initial-data solver ------------------------------------
add_executable(TwoPunctureABE src/TwoPunctureABE.C src/TwoPunctures.C)
target_link_libraries(TwoPunctureABE PRIVATE MPI::MPI_CXX gfortran)
target_link_directories(TwoPunctureABE PRIVATE /home/ckyasb/miniconda3/lib)

# --- ABEGPU: GPU main evolution executable (3d298f3 device-resident baseline)
# BSSN RHS + ghost exchange + prolongation + analysis all on device; host-device
# transfers only at move_to_gpu/move_to_cpu boundaries. Built from the same
# C++/Fortran sources as ABE, EXCEPT bssn_class.C is replaced by bssn_gpu_class.C
# (which redefines bssn_class::Evolve/Step/RecursiveStep/...) and the GPU
# kernel/agent files are added.
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 80)             # NVIDIA A100, sm_80
set(CMAKE_CUDA_FLAGS "-rdc=true -lineinfo")

set(ABEGPU_CXX_SOURCES
  src/ABE.C src/Ansorg.C src/Block.C src/MPatch.C src/Parallel.C
  src/bssn_gpu_class.C src/bssn_step_gpu.C
  src/cgh.C src/checkpoint.C src/misc.C
  src/monitor.C src/perf.C src/surface_integral.C
  src/transpbh.C src/var.C
  src/helper.cpp src/Parallel_GPU.cpp
)
set(ABEGPU_CUDA_SOURCES
  src/fmisc_gpu.cu src/lopsidediff_gpu.cu src/kodiss_gpu.cu src/diff_new_gpu.cu
  src/bssn_rhs_gpu.cu src/prolongrestrict_cell_gpu.cu src/gpu_manager.cu
  src/enforce_algebra_gpu.cu src/sommerfeld_rout_gpu.cu src/rungekutta4_rout_gpu.cu
  src/fadmquantites_bssn_gpu.cu src/misc_gpu.cu src/MPatch_gpu.cu
  src/surface_integral_gpu.cu src/getnp4_gpu.cu
)

add_executable(ABEGPU ${ABEGPU_CXX_SOURCES} ${ABEGPU_CUDA_SOURCES}
               ${AMSS_F90SOURCES})
set_source_files_properties(${ABEGPU_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
set_target_properties(ABEGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(ABEGPU PRIVATE USE_GPU fortran3 newc)
target_include_directories(ABEGPU PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  /home/jjsnam/spack/opt/spack/linux-icelake/openmpi-5.0.9-nd3t2hacw7x7xkmrtqkzr4gxxr54isly/include
  /usr/local/cuda-13.3/include
)
target_link_libraries(ABEGPU PRIVATE MPI::MPI_CXX gfortran)
target_link_directories(ABEGPU PRIVATE /home/ckyasb/miniconda3/lib)
