# 大作业题目一：Winograd 卷积优化（CPU 版本）

## 简介
本目录下提供了 Winograd 算法的 CPU 实现，主要包括以下文件：

- `naive_conv.c` 文件中包含了使用朴素卷积算法的基准实现，请不要修改；
- `winograd_conv.c` 文件中的提供了未经优化的 F(2x2, 3x3) 的 Winograd 卷积算法实现，您需要在此基础上保留 `winograd_conv` 的函数签名进行优化；
- `main.c` 文件中包含了测试代码和性能评估，请不要修改。

## 编译和运行
Makefile 文件中定义了编译规则，您可以尝试使用不同编译器和编译参数来优化性能。默认使用 `gcc -O3 -fopenmp` 编译。

```bash
# 使用 Makefile 编译
make

# 或者直接使用 gcc 编译
gcc -O3 -fopenmp main.c naive_conv.c winograd_conv.c -o winograd
```

`inputs` 文件夹中提供了两组测试样例，`test.txt` 文件的输入较小，运行速度较快，不作为性能评估结果，您可以用于正确性测试；`config.txt` 是最终用于性能测试的输入。

要运行程序，可以使用以下命令：

```bash
./winograd <config_file>
```

您将会在命令行中看到输出结果，以下是一个示例输出：

```
$ ./winograd inputs/test.txt 
=== Running Naive Convolution ===
Layer  0: 1.34 ms (0.0 GFLOPS)
Layer  1: 1.15 ms (0.1 GFLOPS)
Layer  2: 1.13 ms (2.2 GFLOPS)
Layer  3: 4.00 ms (42.5 GFLOPS)
Layer  4: 58.45 ms (93.0 GFLOPS)
Baseline Total: 66.08 ms (84.9 GFLOPS)

=== Running Winograd Convolution ===
Layer  0: 0.81 ms (0.0 GFLOPS)
Layer  1: 0.87 ms (0.1 GFLOPS)
Layer  2: 1.08 ms (2.3 GFLOPS)
Layer  3: 11.32 ms (15.0 GFLOPS)
Layer  4: 67.45 ms (80.6 GFLOPS)
Custom Total: 81.52 ms (68.8 GFLOPS)

=== Correctness Check ===
Layer  0: CORRECT (Speedup: 1.67x)
Layer  1: CORRECT (Speedup: 1.32x)
Layer  2: CORRECT (Speedup: 1.05x)
Layer  3: CORRECT (Speedup: 0.35x)
Layer  4: CORRECT (Speedup: 0.87x)

=== Final Results ===
Results are correct!
Baseline: 66.08 ms (84.9 GFLOPS)
Custom:   81.52 ms (68.8 GFLOPS)
Overall speedup: 0.81x
```