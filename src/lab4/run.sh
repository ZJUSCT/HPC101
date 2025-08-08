#!/bin/bash
#SBATCH --job-name=solver
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --partition=solver1

# 对于 sbatch 脚本参数，请阅读 https://slurm.schedmd.com/sbatch.html
# ntasks-per-node 对应每个节点上的进程数
# cpus-per-task 对应每个进程的线程数

# 先加载 spack 再加载编译环境
# e.g. 
# source ${HOME}/spack/share/spack/setup-env.sh
# spack load intel-oneapi-mpi

# Run BICGSTAB
./build/bicgstab $1