cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE PATH "Path to nvcc compiler")
project(CudaConv CXX CUDA)

find_package(CUDAToolkit 12.9 REQUIRED)
find_package(OpenMP REQUIRED)


set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES "70")


set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass 
    CACHE PATH "Path to cutlass directory")
message(STATUS "CUTLASS dir is ${CUTLASS_DIR}")

set(CUTLASS_INCLUDE_DIR ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUTLASS_INCLUDE_DIR})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr -Wno-deprecated-gpu-targets")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set inputs

set(BATCH_SIZE 32 CACHE STRING "Batch size")
set(INPUT_SIZE 100 CACHE STRING "Size of the input")
set(KERNEL_SIZE 3 CACHE STRING "Size of the kernel")
set(IN_CHANNEL 128 CACHE STRING "Number of input channels")
set(OUT_CHANNEL 128 CACHE STRING "Number of output channels")

add_executable(conv_int8 src/main.cu src/conv.cu)
target_include_directories(conv_int8 PRIVATE ${CUTLASS_INCLUDE_DIR})
target_compile_options(conv_int8 PRIVATE -O3 -Xcompiler -fopenmp)
set_target_properties(conv_int8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(conv_int8 OpenMP::OpenMP_CXX)
target_compile_options(conv_int8 PRIVATE -DBATCH_SIZE=${BATCH_SIZE} 
                                    -DINPUT_SIZE=${INPUT_SIZE} 
                                    -DKERNEL_SIZE=${KERNEL_SIZE} 
                                    -DIN_CHANNEL=${IN_CHANNEL} 
                                    -DOUT_CHANNEL=${OUT_CHANNEL}
                                    -DUSE_INT8)


add_executable(conv_half src/main.cu src/conv.cu)
target_include_directories(conv_half PRIVATE ${CUTLASS_INCLUDE_DIR})
target_compile_options(conv_half PRIVATE -O3 -Xcompiler -fopenmp)
set_target_properties(conv_half PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(conv_half OpenMP::OpenMP_CXX)
target_compile_options(conv_half PRIVATE -DBATCH_SIZE=${BATCH_SIZE} 
                                    -DINPUT_SIZE=${INPUT_SIZE} 
                                    -DKERNEL_SIZE=${KERNEL_SIZE} 
                                    -DIN_CHANNEL=${IN_CHANNEL} 
                                    -DOUT_CHANNEL=${OUT_CHANNEL}
                                    -DUSE_HALF)

add_library(conv_int8_ptx OBJECT src/main.cu src/conv.cu)
target_include_directories(conv_int8_ptx PRIVATE ${CUTLASS_INCLUDE_DIR})
target_compile_options(conv_int8_ptx PRIVATE -O3 -Xcompiler -fopenmp)
set_target_properties(conv_int8_ptx PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(conv_int8_ptx OpenMP::OpenMP_CXX)
target_compile_options(conv_int8_ptx PRIVATE -DBATCH_SIZE=${BATCH_SIZE} 
                                    -DINPUT_SIZE=${INPUT_SIZE} 
                                    -DKERNEL_SIZE=${KERNEL_SIZE} 
                                    -DIN_CHANNEL=${IN_CHANNEL} 
                                    -DOUT_CHANNEL=${OUT_CHANNEL}
                                    -DUSE_INT8)
set_target_properties(conv_int8_ptx PROPERTIES CUDA_PTX_COMPILATION ON
                                    OUTPUT_NAME conv_int8.ptx)

add_library(conv_half_ptx OBJECT src/main.cu src/conv.cu)
target_include_directories(conv_half_ptx PRIVATE ${CUTLASS_INCLUDE_DIR})
target_compile_options(conv_half_ptx PRIVATE -O3 -Xcompiler -fopenmp)
set_target_properties(conv_half_ptx PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(conv_half_ptx OpenMP::OpenMP_CXX)
target_compile_options(conv_half_ptx PRIVATE -DBATCH_SIZE=${BATCH_SIZE} 
                                    -DINPUT_SIZE=${INPUT_SIZE} 
                                    -DKERNEL_SIZE=${KERNEL_SIZE} 
                                    -DIN_CHANNEL=${IN_CHANNEL} 
                                    -DOUT_CHANNEL=${OUT_CHANNEL}
                                    -DUSE_HALF)
set_target_properties(conv_half_ptx PROPERTIES CUDA_PTX_COMPILATION ON
                                    OUTPUT_NAME conv_half.ptx)