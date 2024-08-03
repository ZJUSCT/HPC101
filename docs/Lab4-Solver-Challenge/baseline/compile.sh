#!/bin/bash

# 加载所需的模块
# e.g. 
# module load openmpi

# 运行你的命令
cmake -B build
cmake --build build