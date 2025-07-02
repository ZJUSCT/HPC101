# 提交作业

## Slurm 简介

!!! quote

    - [Quick Start User Guide](https://slurm.schedmd.com/quickstart.html)

      Slurm 提供三大功能：

      - 控制用户对节点的访问
      - 提供并行任务执行框架
      - 提供任务调度和队列

      Slurm 中的几个实体：

      - 节点（node）组成分区（partition），可以视为任务队列
      - 任务（job）由一个或多个步骤（step）组成


## 分区简介

通过 `sinfo` 命令可以查看分区信息：

```shell
sinfo
M6*          up    1:00:00      4    mix M[600-603]
M6*          up    1:00:00      1   idle M604
M7           up    1:00:00      2   idle M[700-701]
V100         up    1:00:00      3    mix v[00-01,13]
V100         up    1:00:00     29   idle v[02-03,10-12,20-23,30-33,40-43,50-53,60-63,70-73]
kunpeng      up      30:00      1   idle k11
riscv        up      30:00      1  idle* rv01
riscv        up      30:00      1   idle rv00
```

- M6, M7 分区是超算队自有节点。

- V100 分区是信息中心提供的 GPU 节点，共 8 台物理机，64 个 V100 GPU。我们把它切成了 32 个节点，每个节点有 2 个 GPU。

- kunpeng 分区由信息中心提供的鲲鹏 920 节点构成。

- riscv 分区由[**进迭时空**](https://www.spacemit.com/)提供的给超算队的 RISC-V 开发板构成。

**具体参见 [集群概况](./overview.md)**

!!! tip

    对于 M6 集群，m60[0-1] CPU 一样, m60[2-4] CPU 一样，跑任务的时候尽量避免混用。提交任务时可以使用 `-w` 参数指定使用哪些节点。



## 提交任务

Slurm 有三种提交任务的方式：

- `srun`
- `sbatch`
- `salloc`

### 批处理任务

使用 `sbatch` 提交任务时，脚本文件模板如下：

```shell title="job.sh"
#!/bin/bash
#SBATCH --job-name=hpl
#SBATCH --partition=M6
#SBATCH --nodes=4                # Number of nodes
#SBATCH --ntasks=8               # Number of MPI ranks
#SBATCH --ntasks-per-node=2      # Number of MPI ranks per node
#SBATCH --cpus-per-task=8        # Number of OpenMP threads for each MPI rank
#SBATCH --time=00:00:30
#SBATCH --mail-type=END,FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mail@zju.edu # Email address
#SBATCH --output=%x_%j.log     # Standard output and error log
## Command(s) to run (example):
module load openmpi hpl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun xhpl
```

在运行 Intel MPI 时，需要设置环境变量 `I_MPI_PMI_LIBRARY`。例如，使用 `srun` 运行 `xhpl` 时:

```shell hl_lines="3 10 12"
spack load intel-oneapi-mpi

export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0
export I_MPI_DEBUG=+5

# 使用 srun 命令运行 xhpl 程序
# -N 2: 使用 2 个节点
# -p M7: 在 M7 分区运行
# -w M700,M701: 指定使用 M700 和 M701 两个节点
# --mpi=pmi2: 使用 PMI2 接口进行 MPI 通信
# --ntasks-per-node=1: 每个节点运行 1 个 MPI 进程
# --cpus-per-task=96: 每个任务使用 96 个 CPU 核心
srun -N 2 -p M7 -w M700,M701 --mpi=pmi2 --ntasks-per-node=1 --cpus-per-task=96 xhpl
```

!!! tip

    我们推荐使用 Intel MPI。


!!! warning

    请注意，slurm 会强制绑定任务（包括ssh会话里的进程）运行在CPU的某些核心上，在提交作业时如果不指定 `--cpus-per-task`，则默认使用 1 个线程。
    这时所有进程都会运行在同一个核心上。



输出文件名有一些特殊的格式：

| 格式 | 描述 |
| --- | --- |
| `%j` | 作业 ID |
| `%n` | 节点名<br>使用它可以将不同节点的输出分开 |
| `%x` | 任务名 |
| `%u` | 用户名 |

举例：

```text
job%2j-%2t.out
    job128-00.out, job128-01.out, ...
```

### 交互式任务

使用 `salloc` 可以申请节点进行交互式任务：

1. **申请节点**：

    ```bash
    salloc -p V100
    ```

2. **连接到分配的节点**：

    ```bash
    ssh nodename
    ```

    你也可以使用 `-w` 参数指定申请的节点名：

    ```bash
    salloc -p V100 -w v63
    ```

!!! tip

    使用 `salloc` 申请到节点后，记得在完成工作后使用 `exit` 命令或 `scancel` 释放资源。


## 查看任务

可以使用 `squeue` 命令查看当前任务：

```shell
squeue
```



要使用 `sacct` 命令查看历史任务，你可以按照以下步骤操作：

1. **基本使用**：直接使用 `sacct` 命令来查看当前或者过去任务的概况。你可以在终端中输入以下命令：

   ```bash
   sacct
   ```

   这将显示最近的任务信息，包括任务 ID、任务名称、状态等。

2. **指定时间范围**：如果你只想查看特定时间范围内的任务，可以使用 `--start` 和 `--end` 参数。例如，要查看从 2024 年 8 月 1 日到 8 月 31 日之间的任务，你可以使用：

   ```bash
   sacct --start=2024-08-01 --end=2024-08-31
   ```

3. **指定字段**：默认情况下，`sacct` 可能会显示许多你不关心的字段。你可以通过 `--format` 参数指定显示哪些字段。例如，如果你只想查看任务 ID、任务名称、状态和运行时间，可以使用：

   ```bash
   sacct --format=JobID,JobName,State,Elapsed
   ```

4. **查看某个特定任务**：如果你知道某个任务的 ID，并且想查看该任务的详细信息，可以直接使用任务 ID：

   ```bash
   sacct -j <JobID>
   ```

   例如：

   ```bash
   sacct -j 12345
   ```

5. **过滤状态**：你还可以使用 `-s` 参数来过滤任务状态。例如，查看所有失败的任务：

   ```bash
   sacct -s F
   ```

6. **更多帮助**：`sacct` 命令有许多选项，你可以使用 `man sacct` 或 `sacct --help` 查看所有可用选项及其详细说明。

通过这些步骤，你可以灵活地使用 `sacct` 命令查看并管理你的历史任务记录。
