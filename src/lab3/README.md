# Lab3: CUDA Conv2D Implementation

## 简介

本目录下包含了一个简单的 CUDA 卷积算子实现。为了便于你编写，所有的输入均为常量，定义在 `include/common.cuh` 中。你需要优化 `src/conv.cu` 中的两种不同精度下的卷积算子。

`reference_cpu.cuh` 和 `reference_gpu.cuh` 中分别包含了使用 `OpenMP` 和 `CUTLASS` 实现的 CPU 和 GPU 卷积算子作为参考实现。你可以使用这些实现来验证你的 CUDA 实现的正确性。

不考虑特殊指令集 （如 Tensor Core 和 DP4A） 的情况下，两种不同精度下的卷积算子的优化思路是相同的。

为了避免舍入带来的精度问题，输入数据的范围做了限制，可以保证中间过程和结果不超过 `float16` 和 `int8` 能**精确**表示的范围，也就是说如果你的实现正确，你的结果应该和参考实现完全一致。

## 编译和运行

项目依赖 `CUTLASS`，集群上有一个已经下载好的版本在 `/river/hpc101/2025/lab3/cutlass`。

```bash
cmake -B build -DCUTLASS_DIR=/river/hpc101/2025/lab3/cutlass
cmake --build build 
```

如果你需要修改输入大小，可以直接在 `include/common.cuh` 中修改。

## 运行

```bash
./build/conv_int8
./build/conv_half
```

因为 `CUTLASS` 对 `int8` 的实现无法通过编译，所以 `int8` 使用 CPU 实现来参考，而 `half` 使用 `CUTLASS` 的 GPU 实现。

另外，编译命令里还有两个 `conv_int8_ptx` 和 `conv_half_ptx` 的目标，这两个目标会生成对应的 `.ptx` 文件，你可以通过这些 `.ptx` 文件来验证是否使用了正确的指令。