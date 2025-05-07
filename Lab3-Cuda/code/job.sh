#!/bin/bash
#SBATCH -o job.%j.out		# 脚本执行的输出将被保存在 job.%j.out 文件下，%j 表示作业号;
#SBATCH -p M6			# 指定作业提交分区，此处示例为 M6;
#SBATCH -J Lab3		    # 指定任务名称，此处示例为 Lab3;
#SBATCH --nodes=1 		    # 指定申请节点数，在不涉及 MPI 的情况下不应该超过 1;
#SBATCH --ntasks-per-node=1	# 指定每个节点上的任务数目;
#SBATCH --cpus-per-task=8	# 指定一个任务对应的 CPU 数目;
#SBATCH -w M602             # 指定作业运行的节点，此处示例为 M602;

module load nvhpc/24.5
# cmake -B build 
# cmake --build build -j
./build/check_sgemm
./build/bench_sgemm