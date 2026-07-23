# 提交计算任务

DevPod 适合编辑、编译和小规模调试，但它的资源是受限的。需要完整算力（整机核数、独占绑核、GPU）时，需要把程序作为**计算任务**提交到分区队列中运行。

在 DevPod 内使用 `hpc` 命令提交和管理任务。它的用法是 Slurm 风格的，如果你用过 Slurm 会觉得很熟悉：

!!! tip "Tips for agent"
    `hpc skills` 显示 `hpc` 的用法。

| Slurm | hpc | 说明 |
| --- | --- | --- |
| `sinfo` | `hpc partitions`（别名 `hpc sinfo`） | 查看分区容量、限制、可用镜像 |
| `srun cmd` | `hpc submit cmd` | 运行一条命令并实时输出 |
| `sbatch job.sh` | `hpc submit job.sh` | 提交脚本，`#HPC` 指令设置参数 |
| `srun --pty bash` | `hpc submit --interactive bash` + `hpc attach` | 交互式任务 |
| `squeue` | `hpc squeue` | 查看排队/运行中的任务 |
| `scancel` | `hpc cancel` | 取消任务 |

在同一个物理区域内，任务和 DevPod **共享同一个家目录**，故在 DevPod 里准备好代码，直接 `hpc submit` 即可在计算节点上运行。

## 分区

任务提交到**分区**（partition）。用下面的命令查看你能用的分区、它们的资源范围和排队情况：

```shell
hpc partitions   # 分区容量、CPU/GPU/内存/时长限制、允许的镜像
hpc limits       # 每个分区对「你」的限制（可申请的资源范围、同时任务数上限）
```

各实验开放的分区以实验文档和平台页面为准。不指定 `-p` 时任务会进入默认分区。

## 提交任务

### 直接运行命令

```shell
hpc submit -p intels4 -c 4 -m 8Gi ./build/myprog
```

任务会排队、启动，然后把输出实时流回你的终端，结束后返回（退出码就是你命令的退出码）。加 `-d`（detach）则只打印任务号立即返回。默认只显示任务自己的输出；加 `--events` 可以看到排队位置、调度事件等横幅。

!!! warning "hpc 的参数必须写在命令前面"

    从第一个非选项参数（你的命令）开始，后面的内容会**原样传给你的程序**（引号也会保留），不再被 `hpc` 解析：

    ```shell
    hpc submit uname -a                        # -a 传给 uname
    hpc submit python3 -m http.server -p 80    # -m、-p 80 都传给 python3
    hpc submit -c 4 -p intels4 uname -a        # -c/-p 是 hpc 的，-a 是 uname 的
    hpc submit bash -c 'echo $HOSTNAME && date'  # 引号里的脚本原样交给 bash
    ```

    多个参数会像 `srun` 一样**不经 shell 直接执行**。要用管道、`&&` 等 shell 语法，
    写成一个带引号的整行，或者写进脚本提交：

    ```shell
    hpc submit 'make -j && ./run.sh'
    ```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-p, --partition` | 目标分区（省略则用默认分区） |
| `-c, --cpu N` | CPU 核数 |
| `-g, --gpu N` | GPU 数（仅支持 GPU 的分区） |
| `-m, --mem SIZE` | 内存，如 `8Gi` |
| `-t, --time DUR` | 最长运行时长，如 `30m`、`1h30m` |
| `-i, --image` | 容器镜像（须在分区允许列表内，默认取第一个） |
| `-e, --env K=V,…` | 额外环境变量（覆盖继承的同名变量） |
| `--export ALL\|NONE` | 环境继承（见下，默认 ALL） |
| `-o, --output PATH` | 输出文件（见下；`-` = 明确不落文件） |
| `-n, --name` | 任务显示名 |
| `-d, --detach` | 提交后立即返回任务号，不流式输出 |
| `--interactive` | 交互式任务（见下） |
| `--events` | 显示排队/调度横幅（默认只显示任务自己的输出） |

### 环境变量（sbatch 语义）

默认情况下（`--export=ALL`），提交时 shell 的**整个环境跟着任务走**——在 DevPod 里
`export OMP_NUM_THREADS=8` 之后提交，任务里直接生效，和 `sbatch` 一致（会话类变量如
`SSH_*`、`HOSTNAME`、`HOME` 除外）。`PATH` 会与任务镜像自带的 PATH **合并**：你的工具
优先命中，镜像里的（如 CUDA 的 `nvcc`）也不会丢。`-e K=V` 覆盖继承的同名变量；
`--export=NONE` 则只传 `-e` 给出的变量。

### 提交脚本（sbatch 风格）

把参数写进脚本的 `#HPC` 指令（也兼容 `#SBATCH` 写法），直接提交脚本文件。脚本必须以
`#!` 开头，并且**按它的 shebang 解释器执行**——`#!/bin/bash` 就跑在 bash 里，
`#!/usr/bin/env python3` 也可以：

```shell title="job.sh"
#!/bin/bash
#HPC --partition=intels4
#HPC --cpu=8
#HPC --mem=16Gi
#HPC --time=30m
#HPC --output=result_%j.log

make -j
./build/myprog
```

```shell
hpc submit job.sh
```

命令行上显式给出的参数优先于脚本内的指令。

### 交互式任务

需要一个「在计算节点上的 shell」时（类似 `srun --pty bash` / `salloc`）：

```shell
hpc submit -p intels4 -c 8 --interactive bash
```

任务启动后自动进入终端；断开后可以随时用任务号重新接入：

```shell
hpc attach <任务号>
```

!!! tip

    交互式任务同样占用你的任务配额并受时长限制，用完请 `exit` 或 `hpc cancel`，不要让它空挂。

### 工作目录（sbatch 语义）

任务在**你提交时所在的目录**中运行（相对家目录）。在 `~/lab2` 下执行 `hpc submit`，任务的工作目录就是 `~/lab2`。

注意：只有家目录是共享存储。如果你在家目录之外提交，任务看不到那里的文件，会回退到 `~` 运行（提交时会有警告）。

## 输出与日志

输出去向和 Slurm 一致，按提交模式分：

- **前台提交**（默认，`srun` 语义）：输出只流回你的终端，**不写文件**。要留档加 `-o`。
- **后台提交**（`-d`，`sbatch` 语义）：stdout/stderr 写入**工作目录下的持久文件**，默认 `j<任务号>.out`，可用 `-o` 自定义。

文件名支持占位符：

| 占位符 | 含义 |
| --- | --- |
| `%j` | 任务号 |
| `%u` | 用户名 |
| `%x` | 任务名 |
| `%Y %m %d %H %M %S` | 提交时间 |

```shell
hpc submit -d -o result_%j.log ...
```

`hpc logs <任务号>`（`-f` 跟随）看的是容器的**实时日志**——它在任务结束、容器被回收后就没有了。要留档请用 `-d` 或 `-o` 落文件。

任务的最终状态反映你的命令的退出码。

## 查看与管理

```shell
hpc ls               # 我的任务列表（-s STATE 过滤状态）
hpc squeue           # 全体用户的活跃任务
hpc info <任务号>     # 任务详情（状态、节点、资源）
hpc logs -f <任务号>  # 实时日志
hpc cancel <任务号>   # 取消（别名 hpc rm）
```

这些信息在平台的「任务」「队列」页面上也都能看到，网页上同样可以提交任务和查看实时输出。

## 完整示例

```shell
# 编译（在 DevPod 里）
cd ~/lab2 && make -j

# 前台提交，输出直接流回终端（srun 风格,不落文件）
hpc submit -p intels4 -c 8 -m 16Gi -t 30m ./build/benchmark

# 后台提交，输出落到 ~/lab2/j<id>.out（sbatch 风格）
hpc submit -d -c 8 ./build/benchmark
hpc ls
cat ~/lab2/j123.out

# 前台但也想留档
hpc submit -o run_%j.log ./build/benchmark
```
