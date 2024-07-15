# 实验二 Bonus：手写 SIMD 向量化

本部分选做，感兴趣的同学可以尝试着完成。

Bonus 部分完成即有加分（完成 Bonus 部分实验要求，且加速比大于 1），我们将根据完成质量提供 10-15 分的加分（与 Lab2 权重相同）。

<!-- ## 1 实验环境

[aistation 平台](https://aistation.zju.edu.cn:32206) -->

## 1 实验基础知识

现代处理器一般都支持向量化指令，x86 架构下 Intel 和 AMD 两家的处理器都提供了诸如 SSE，AVX 等 SIMD 指令集，一条指令可以同时操作多个数据进行运算，大大提高了现代处理器的数据吞吐量。

现代编译器在高优化等级下，具有自动向量化的功能，对于结构清晰，循环边界清晰的程序，编译器的自动向量化已经可以达到很优秀的程度了。然而，编译器的优化始终是保守的，很多情况下编译器无法完成使用 SIMD 指令进行向量化的工作，为了追求性能，高性能计算领域经常需要手写 SIMD 代码进行代码优化。

显然直接手写汇编指令过于困难，在 C 语言环境下，Intel 提供了一整套关于 SIMD 指令的函数封装接口和指令相关行为的参照手册，可以在实验文档的参考资料中找到。

使用这些函数 API 需要 include 对应的头文件，不同 SIMD 指令集需要的头文件不同，具体需要参考 Intel 相关文档。

```c
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
```

另外深入到这个级别的优化已经开始需要考虑具体处理器的体系结构细节了，如某个架构下某条指令的实现延时和吞吐量是多少，处理器提供了多少向量寄存器，访存的对齐等等。这种时候编译器具体产生的汇编代码能比 C 语言代码提供更多的信息，你能了解到自己使用了多少寄存器，编译器是否生成了预期外的代码等等。

参考资料中提供的 godbolt 是一款基于 web 的研究不同编译器编译产生汇编代码的工具，大家在进行本实验的时候可以学习使用。

## 2 实验步骤

首先大家可以参考之前的[bonus](https://zjusct.pages.zjusct.io/summer-course-2023/HPC101-Labs-2023/Lab2.5-Vectors-Bonus/), 也就是我们上课的时候让同学尝试过的, 如果有同学依然对手写SIMD不熟悉的可以先参考实现那个简单的.

但是想必大家都~~意犹未尽~~, 所以今年我们对该bonus进行了升级(这是你没体验过的船新版本)

!!! tip 算术入门(雾)

    先假设你有一只兔子。
    🐇
    假设现在有人又给了你另一个兔子。
    🐇🐇
    现在，数一数你所拥有的兔子数量，你会得到结果是两只。也就是说一只兔子加一只兔子等于两只兔子，也就是一加一等于二。
    1+1=2
    这就是算术的运算方法了。

    那么，现在你已经对算术的基本原理有了一定的了解，就让我们来看看下面这个简单的例子，来把我们刚刚学到的东西运用到实践中吧。

    ---

    ### 试试看! 
    #### 例题 1.7

    \[
    \begin{gathered} 
    \log \Pi(N)=\Big(N+\dfrac{1}{2}\Big)\log N -N+A-\int_{N}^{\infty}\dfrac{\overline{B}_1(x){\rm d} x}{x}, A=1+\int_{1}^{\infty}\dfrac{\overline{B}_1(x){\rm d} x}{x} \\
    \log \Pi(s)=\Big(s+\dfrac{1}{2}\Big)\log s-s+A-\int_{0}^{\infty}\dfrac{\overline{B}_1(t){\rm d} t}{t+s} 
    \end{gathered}
    \]



baseline如下, 简单来说就是进行`MAXN`次$D_{4*4}=A_{4*12}*B_{12*4}$的矩阵乘法.
其中`MAXN`个矩阵的内容不同, 计算的时候要注意位移.

!!! danger "因为该题只是为了使用手写SIMD, 由此不接受其他优化方法, 包括omp等"

```c
for (int n = 0; n < MAXN; n++)
    {
        /* 可以修改的代码区域 */
        // -----------------------------------
        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 12; j++)
                {
                    *(d + n * 16 + i * 4 + k) += *(a + n * 48 + i * 12 + j) * \
                                                    *(b + n * 48 + j * 4 + k);
                }
            }
        }
        // -----------------------------------
    }
```

!!! tip 编译

    在编译时添加以下选项可以允许编译器生成使用 AVX2 和 FMA 指令集的代码，如果你使用了其它不在编译器默认范围内的指令集，类似的编译选项是必要的。

    ``` bash
    -mavx2 -mfma
    ```

参照 Intel 文档优化完成后就可以开始测试优化前和优化后的性能差距，最后对比前后编译器产生的汇编代码的不同即可完成 Bonus 部分的实验。

## 3 实验初始代码

详见 [multi.cpp](https://git.zju.edu.cn/zjusct/summer_hpc101_2024/hpc-101-labs-2024/-/blob/main/docs/Lab2.5-Vectors-Bonus/multi.cpp)

## 4 实验任务与要求

1. 完成以上代码的手写 SIMD 向量化
2. 测试实现的正确性和加速比
3. 提交代码和简要的思路，对比基础版本和 SIMD 版本编译器生成的汇编代码, 报告中只需附上关键部分

## 参考资料

- Intel® Intrinsics Guide： [https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- Compiler Explorer：[https://godbolt.org/](https://godbolt.org/)
