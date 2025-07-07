# 软件环境

!!! todo

    - [ ] Spack 使用说明
    - [ ] HPC 中常用软件平台说明
    - [ ] 接入可观测性平台，动态监测软件环境

!!! bug

    已知且未修复的问题：

    - VTune modulefile 的 PATH 路径错误。
        - 原因：oneAPI 自带的 modulefile 对 lmod 有兼容性问题。
        - 解决办法：自行修改 PATH。或使用 `/opt/intel/oneapi/setvars.sh` 加载 oneAPI 环境。

## 环境、包管理器

集群使用 Lmod 管理软件环境，并配置了常用软件的 modulefiles。用户登录集群后，应当可以通过 `module` 命令查看可用的软件环境。

```text
$ module avail
---- /opt/apps/spack/share/spack/lmod/linux-debian12-x86_64/Core -----
   openmpi/5.0.3-pe46zvn

------------------ /opt/nvidia/hpc_sdk/modulefiles -------------------
   nvhpc-byo-compiler/24.5    nvhpc-hpcx/24.5     nvhpc-openmpi3/24.5
   nvhpc-hpcx-cuda12/24.5     nvhpc-nompi/24.5    nvhpc/24.5

----------------------- /opt/intel/modulefiles -----------------------
   advisor/2024.2                    dpl/2022.6
   ccl/2021.13.0                     ifort/2024.2.0
   compiler-intel-llvm/2024.2.0      ifort32/2024.2.0
   compiler-intel-llvm32/2024.2.0    intel_ipp_ia32/2021.12
   compiler-rt/2024.2.0              intel_ipp_intel64/2021.12
   compiler-rt32/2024.2.0            intel_ippcp_ia32/2021.12
   compiler/2024.2.0                 intel_ippcp_intel64/2021.12
   compiler32/2024.2.0               mkl/2024.2
   dal/2024.0.0                      mkl32/2024.2
   debugger/2024.2.0                 mpi/2021.13
   dev-utilities/2024.2.0            tbb/2021.13
   dnnl/3.5.0                        tbb32/2021.13
   dpct/2024.2.0                     vtune/2024.2
```

当你使用 `module load` 加载一个 module 时，Lmod 会帮助你修改 `PATH`、`LD_LIBRARY_PATH`、`CPATH` 等环境变量，方便你构建和运行软件。

集群 modulefiles 的来源有以下几种：

- 独立安装的软件，如 Intel oneAPI、NVIDIA HPC SDK 等。主要是为了方便用户从源码构建软件。
- 从 Spack 管理的软件包中生成，方便用户直接加载使用。

请尽量不要混用以上两种体系的 modulefiles。

### 从源码构建

如果觉得使用 Spack 等包管理器不直观、学习难度高，可以选择从源码构建软件。集群上独立安装的 modulefiles 正是为了方便用户从源码构建软件而配置的。

例如，如果你需要使用 NVIDIA 编译器，可以加载 `nvhpc/24.5` module：

```bash
$ module load nvhpc nvhpc-hpcx
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

但少数项目（比如 cuda-sample）会在 Makefile 中使用其他环境变量如 `CUDA_PATH` 指定 CUDA 安装路径，而 module 不会修改这些环境变量。此时你可以查看 module 所添加的 `PATH`，手动设置该环境变量：

```bash
$ echo $PATH
/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/mpi/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/bin
$ export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda
$ make
```

### Spack

Spack 生成的 modulefile 具有层次结构：只有加载的相应的基础包，才会显示基于它的其他包。这样能够避免不同实现软件包之间的冲突。以 `hpl` 为例，没有加载任何 MPI 实现时不会显示：

```shell
$ module avail
---- /opt/apps/spack/share/spack/lmod/linux-debian12-x86_64/Core ----
   openmpi/5.0.3-pe46zvn
$ module load openmpi
$ module avail
---- /opt/apps/spack/share/spack/lmod/linux-debian12-x86_64/Core ----
   openmpi/5.0.3-pe46zvn
   hpl/2.3.0-7q6z7jv
```

!!! note "Tips"
    Spack 安装或解析依赖过程较慢属于正常现象。若出现长时间无响应的情况，可添加 `--debug` 参数以查看详细调试信息。

集群上目前在 `/pxe/opt/spack` 目录下安装了 `Spack`（下称集群 Spack），提供了常用的编译环境。 你可以通过如下命令，将集群 Spack 设置为本地 Spack 的上游 Spack 实例，从而复用集群 Spack 已安装的软件包，节省安装时间。

```shell
spack config add upstreams:zjusct-spack:install_tree:/pxe/opt/spack/opt/spack
```

你可以通过 `spack env list` 来查看已经构建好的环境：
```shell
==> 4 environments
    hpc101-cuda  hpc101-gnu  hpc101-intel  hpc101-llvm
```

并通过 `spack env activate <env_name>` 来加载相应的环境。

加载环境后，你可以通过 `spack find` 来查看已经安装的软件包。

```shell
==> In environment hpc101-intel
==> 5 root specs
[+] intel-oneapi-compilers  [+] intel-oneapi-mpi  [+] intel-oneapi-vtune
[+] intel-oneapi-mkl        [+] intel-oneapi-tbb

-- linux-debian12-icelake / gcc@12.2.0 --------------------------
gcc-runtime@12.2.0               intel-oneapi-mpi@2021.14.0
glibc@2.36                       intel-oneapi-tbb@2022.0.0
intel-oneapi-compilers@2025.0.0  intel-oneapi-vtune@2025.0.0
intel-oneapi-mkl@2024.2.2
==> 7 installed packages
```

更多 Spack 操作，请参考 [该文档](https://docs.zjusct.io/operation/software/spack/).

### Conda

集群预装了 Conda，用户不需要自行安装。你可以直接使用 `conda activate` 激活默认 Conda 环境：

```text
user@machine:~$ conda activate
(base) user@machine:~$
```

在 `base` 环境下，我们预装了 numpy、pytorch 等常用的包。你没有权限直接修改 `base` 环境。如果这些包不能满足你的需求，可以联系我们添加。

当然，如果 `base` 环境不符合你的需求，也可以直接创建自己的环境。新创建的环境将保存在你的家目录下：

```bash hl_lines="10"
(base) user@machine:~$ conda create -n myenv python=3.8
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /home/user/.conda/envs/myenv

  added / updated specs:
    - python=3.8

Proceed ([y]/n)?
```

Conda 支持多级配置文件。你可以在 `~/.condarc` 中配置个人的 Conda 环境，包括添加 channels、设置 proxy 等。你可以使用 `conda info` 查看当前 Conda 实例的配置：

```bash
(base) user@machine~$ conda info
     active environment : base
    active env location : /opt/conda
       user config file : /home/user/.condarc
 populated config files : /opt/conda/.condarc
```

## 可用性矩阵

| 软件/套件名               | 安装方式            | 安装位置        | 可用性                                           |                 |            |                           |
| ------------------------- | ------------------- | --------------- | ------------------------------------------------ | --------------- | ---------- | ------------------------- |
|                           |                     |                 | Debian 12 (bookworm)                             | Debian  testing | Debian sid | Ubuntu  22.04 LTS (jammy) |
| MLNX_OFED                 | 官方下载            | 默认            | ✅MLNX_OFED_LINUX-24.04-0.6.6.0-debian12.1-x86_64 |                 |            |                           |
| CUDA Driver               | 官方下载            | 默认            | ✅NVIDIA-Linux-x86_64-550.90.07                   |                 |            |                           |
| Lmod                      | 源码                | /opt/apps/lmod  | ✅git  latest repo                                |                 |            |                           |
| Spack                     | 源码                | /opt/apps/spack | ✅git  latest repo                                |                 |            |                           |
| Conda                     | 官方下载            | /opt/conda      | ✅  24.5.0     Python 3.12.4.final.0              |                 |            |                           |
| NVIDIA HPC SDK            | 官方下载            |                 | ✅nvhpc_2024_245_Linux_x86_64_cuda_12.4           |                 |            |                           |
| Intel oneAPI Base Toolkit | 官方下载            |                 | ✅l_BaseKit_p_2024.2.0.634_offline                |                 |            |                           |
| Intel® HPC Toolkit        | 官方下载            |                 | ✅l_HPCKit_p_2024.2.0.635_offline                 |                 |            |                           |
| GCC, G++                  | 系统包管理器        |                 | ✅                                                |                 |            |                           |
| GDB                       | 系统包管理器        |                 | ✅                                                |                 |            |                           |
| gFortran                  | 系统包管理器        |                 | ✅                                                |                 |            |                           |
| GMP                       | 系统包管理器        |                 | ✅  libgmp-dev                                    |                 |            |                           |
| Clang                     | 系统包管理器        |                 | ✅                                                |                 |            |                           |
| LLDB                      | 系统包管理器        |                 | ✅                                                |                 |            |                           |
| MPICH                     |                     |                 |                                                  |                 |            |                           |
| OpenMPI                   | Spack               |                 | ✅                                                |                 |            |                           |
| htop/btop/nvtop           | 系统包管理器        |                 | ✅                                                |                 |            |                           |
| Tmux/screen               | 系统包管理器        |                 | ✅                                                |                 |            |                           |
| Docker                    |                     |                 | ❌使用  apptainer                                 |                 |            |                           |
| Podman                    | 系统包管理器        |                 | ✅                                                |                 |            |                           |
| Singularity               |                     |                 | ⏸️暂无安装计划                                    |                 |            |                           |
| Apptainer                 | GitHub  发布 Deb 包 |                 | ✅apptainer_1.3.2_amd64.deb                       |                 |            |                           |
