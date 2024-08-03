cmake_minimum_required(VERSION 3.12)
project(bicgstab C CXX Fortran)

find_package(OpenMP REQUIRED)
# Uncomment below is MPI env is available
# find_package(MPI REQUIRED)

# Set Compiler
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_Fortran_COMPILER "gfortran")

# Set flags
set(CMAKE_C_FLAGS "-Og -g")
set(CMAKE_CXX_FLAGS "-Og -g")
set(CMAKE_Fortran_FLAGS "-Og -g")

# Include headers
include_directories(include)

# Add common source files
file(GLOB COMMON_SOURCES "src/*.cpp")

# Create an object library for common sources
add_library(judger OBJECT ${COMMON_SOURCES})

# If found C implementation, add it to the build
if(EXISTS "${CMAKE_SOURCE_DIR}/src/bicgstab/solver.c")
  add_executable(bicgstab src/bicgstab/solver.c $<TARGET_OBJECTS:judger>)
endif()

# If found Fortran implementation, add it to the build
if(EXISTS "${CMAKE_SOURCE_DIR}/src/bicgstab/solver.f90")
  add_executable(bicgstab-fortran src/bicgstab/solver.f90 $<TARGET_OBJECTS:judger>)
endif()
