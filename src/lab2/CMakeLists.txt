cmake_minimum_required(VERSION 3.14)

project(lab2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

include_directories(${PROJECT_SOURCE_DIR}/include)

<<<<<<< HEAD
# Framework (driver + scalar reference baseline): fixed portable flags.
# This is the yardstick every submission is measured against — arch-specific
# flags must never apply here, or the baseline itself would speed up.
add_library(framework OBJECT main.cpp src/moe_ref.cpp src/data.cpp)
target_compile_options(framework PRIVATE -O3 -g)

# Student code lives in student/ — everything else is replaced with the
# original version when grading. Arch-specific flags apply here and only
# here, per target ISA; on any other machine (e.g. an ARM laptop) none
# apply and the project still builds.
add_library(student OBJECT student/moe_opt.cpp)
target_compile_options(student PRIVATE -O3 -g)
=======
# Student code lives in student/ — everything else is replaced with the
# original version when grading.
add_executable(${PROJECT_NAME} main.cpp src/moe_ref.cpp src/data.cpp
               student/moe_opt.cpp)

add_executable(${PROJECT_NAME}_bonus bonus.cpp src/moe_ref.cpp src/data.cpp
               src/lora_ref.cpp src/gradcheck.cpp student/lora_opt.cpp)
>>>>>>> 0e1a6fe (feat(lab2): adopt DeepSeek-V3 MoE architecture and student-file framework)

# OpenMP is used only by the optimized student implementation.  Keep the
# scalar framework unchanged so the baseline remains a single-threaded
# reference, but link the final executable with the OpenMP runtime.
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(student PRIVATE OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP CXX support was not found; OpenMP pragmas will be ignored")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  check_cxx_compiler_flag("-march=sapphirerapids" COMPILER_SUPPORTS_SPR)
  if(COMPILER_SUPPORTS_SPR)
    target_compile_options(student PRIVATE -march=sapphirerapids)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
  check_cxx_compiler_flag("-march=rv64gcv" COMPILER_SUPPORTS_RVV)
  if(COMPILER_SUPPORTS_RVV)
    target_compile_options(student PRIVATE -march=rv64gcv)
  endif()
endif()

add_executable(${PROJECT_NAME} $<TARGET_OBJECTS:framework>
                               $<TARGET_OBJECTS:student>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
