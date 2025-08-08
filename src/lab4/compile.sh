#!/bin/bash

# 先加载 spack 再加载编译环境
# e.g. 
# source ${HOME}/spack/share/spack/setup-env.sh
# spack load intel-oneapi-compilers

# 运行你的命令
cmake -B build
cmake --build build
