cmake_minimum_required(VERSION 3.12)
project(gemm CXX CUDA)
include(FetchContent)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(NVHPC_CUDA_VERSION 12.4)
set(CMAKE_CUDA_ARCHITECTURES "75;80")

list(APPEND CMAKE_PREFIX_PATH ${NVHPC_ROOT})
find_package(NVHPC REQUIRED COMPONENTS CUDA MATH)

set(CUTLASS_SYSTEM_SOURCE_DIR /river/hpc101/2024/lab3/cutlass)
set(CUTLASS_SOURCE_DIR ${CUTLASS_SYSTEM_SOURCE_DIR})

set(NVBENCH_SYSTEM_SOURCE_DIR /river/hpc101/2024/lab3/nvbench)
set(NVBENCH_SOURCE_DIR ${CMAKE_SOURCE_DIR}/nvbench)

if(NOT EXISTS ${NVBENCH_SOURCE_DIR})
	file(COPY ${NVBENCH_SYSTEM_SOURCE_DIR} DESTINATION ${CMAKE_SOURCE_DIR})
endif()

set(CUTLASS_INCLUDE_DIR ${CUTLASS_SOURCE_DIR}/include ${CUTLASS_SOURCE_DIR}/tools/util/include)


message(STATUS "CUTLASS_SOURCE_DIR: ${CUTLASS_SOURCE_DIR}")
message(STATUS "NVBENCH_SOURCE_DIR: ${NVBENCH_SOURCE_DIR}")

include_directories(include)
include_directories(${CUTLASS_INCLUDE_DIR})
include_directories(${NVBENCH_SOURCE_DIR})
add_subdirectory(${NVBENCH_SOURCE_DIR})


add_executable(check_sgemm src/sgemm_kernel.cu src/main.cu src/cublas.cu)
target_compile_definitions(check_sgemm PRIVATE FP32=1)
target_link_libraries(check_sgemm PRIVATE cublasLt)

add_executable(check_hgemm src/hgemm_kernel.cu src/main.cu src/cublas.cu)
target_compile_definitions(check_hgemm PRIVATE FP16=1)
target_link_libraries(check_hgemm PRIVATE cublasLt)

add_executable(bench_sgemm src/sgemm_kernel.cu src/bench.cu src/cublas.cu)
target_compile_definitions(bench_sgemm PRIVATE FP32=1)
target_link_libraries(bench_sgemm PRIVATE cublasLt nvbench::main)

add_executable(bench_hgemm src/hgemm_kernel.cu src/bench.cu src/cublas.cu)
target_compile_definitions(bench_hgemm PRIVATE FP16=1)
target_link_libraries(bench_hgemm PRIVATE cublasLt nvbench::main)