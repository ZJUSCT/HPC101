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

### Spack

!!! note "Tips"
    Spack 安装或解析依赖过程较慢属于正常现象。若出现长时间无响应的情况，可添加 `--debug` 参数以查看详细调试信息。

集群上目前在 `/pxe/opt/spack` 目录下安装了 `Spack`（下称集群 Spack），提供了常用的编译环境。 你可以通过如下命令，将集群 Spack 设置为本地 Spack 的上游 Spack 实例，从而复用集群 Spack 已安装的软件包，节省安装时间。

```shell
# Load Cluster Level spack
source /pxe/opt/spack/share/spack/setup-env.sh
```

你可以通过 `spack find` 来查看已经安装的软件包。
```shell
$ spack find
   -- linux-debian12-haswell / gcc@12.2.0 --------------------------
   aocc@5.0.0       curl@7.88.1         gettext@0.21                     intel-oneapi-mpi@2021.14.1   libfabric@1.22.0   libxml2@2.13.5      mpich@4.2.3                     openmpi@5.0.6           pmix@5.0.5            rdma-core@52.0          xz@5.4.6
   autoconf@2.71    diffutils@3.8       glibc@2.36                       intel-oneapi-vtune@2025.0.1  libffi@3.4.6       llvm@19.1.7         mvapich2@2.3.7-1                openssh@9.2p1           py-docutils@0.20.1    readline@8.2            yaksa@0.3
```

并通过 `spack load <package_name>` 来加载相应软件包。
```shell
$ spack load cuda@12.8.0
$ nvcc --version
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2025 NVIDIA Corporation
   Built on Wed_Jan_15_19:20:09_PST_2025
   Cuda compilation tools, release 12.8, V12.8.61
   Build cuda_12.8.r12.8/compiler.35404655_0 
```

---

需要使用多个软件包时，逐个执行 `spack load` 比较繁琐，我们可以通过环境来管理一组软件包。集群 Spack 预装了几个环境，你可以通过 `spack env list` 来查看已经构建好的环境：
```shell
$ spack env list
   ==> 4 environments
      hpc101-cuda  hpc101-gnu  hpc101-intel  hpc101-llvm
```

并通过 `spack env activate <env_name>` 来加载相应环境中的软件包。

加载环境后，你可以通过 `spack find` 来查看该环境内的软件包。

```shell
$ spack find
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

如有需要，你也可以构建自己的环境。

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
