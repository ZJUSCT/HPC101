# 实验三：GDN Prefill 前向优化

!!! info "实验信息"

    负责助教：林熙、胡哲文、丁宁、童泽远

## 实验目标

本实验围绕 **GDN（Gated DeltaNet）的 prefill 前向计算**展开。你需要使用 TileLang 完成给定算子的
GPU 实现，并在保证结果正确的前提下进行优化。

完成实验后，你应当能够：

- 理解 Gated DeltaNet 的计算流程；
- 理解 chunk-wise parallel 的作用及其 GPU 实现；
- 使用分块、数据复用、融合和流水化等方法优化 GPU kernel；
- 使用性能分析工具定位瓶颈，并用可复现的数据验证优化效果。

## 背景知识

### Prefill 与 Decode

自回归模型推理通常分为两个阶段：

- **Prefill**：一次处理提示词中的全部 token，生成各层后续 decode 所需的状态；
- **Decode**：每次处理一个或少量新 token，并复用 prefill 阶段产生的状态。

本实验只关注 prefill forward。与单 token decode 相比，prefill 提供了更多序列维度上的并行机会，
但 GDN 的状态递推仍然引入了因果依赖，因此不能简单地把所有 token 当作相互独立的任务。

??? note "扩展阅读"

    这里只是对 prefill 与 decode 的概念进行简要说明，更多关于自回归模型推理的内容可以参考互联网上的一些讲解。
    
    *下面是私货夹带时间，里面也有一些相关的基础知识*

    [An Introduction to Efficient LLM Inference for ZJUSCT](https://blogs.erix025.me/EfficientAI/sct-llm-talk/sct-llm-talk/)

### Linear Attention 与 Gated DeltaNet

#### Linear Attention

标准的因果自注意力可以写为：

$$
O = \operatorname{softmax}\left(\frac{QK^{\mathsf T} + M}{\sqrt{d_k}}\right)V.
$$

其中 $M$ 是下三角的因果掩码。对于长度为 $L$ 的序列，显式计算所有 token 两两之间的注意力分数需要
$O(L^2d)$ 的计算量。这一二次复杂度来源于构造 $QK^{\mathsf T}$ 所产生的
$L\times L$ 中间结果。Linear Attention 的核心思想是避免显式计算这一二次规模的中间矩阵，从而将复杂度降低到线性。

最初的经典方法使用特征映射 $\phi(\cdot)$ 对 softmax kernel 进行近似：

$$
\exp(QK^{\mathsf T})
\approx
\phi(Q)\phi(K)^{\mathsf T},
\qquad
O_t=
\frac{
\sum_{i=1}^{t}\phi(q_t)\phi(k_i)^{\mathsf T}v_i
}{
\sum_{i=1}^{t}\phi(q_t)\phi(k_i)^{\mathsf T}
}.
$$

由于 $\phi(q_t)$ 与求和无关，可以利用矩阵乘法的结合律将其提出求和，即先累积历史的 key-value 外积，再与当前 query 做乘法：

$$
O_t=
\frac{
\phi(q_t)
\left(
\sum_{i=1}^{t}\phi(k_i)^{\mathsf T}v_i
\right)
}{
\phi(q_t)
\left(
\sum_{i=1}^{t}\phi(k_i)^{\mathsf T}
\right)
}.
$$

对于因果注意力，上式中的历史累积项可以递推计算。为简洁起见，下文省略特征映射 $\phi(\cdot)$ 和归一化项，仅保留状态更新形式，则有

$$
\begin{aligned}
S_t &= S_{t-1} + k_t^{\mathsf T}v_t, \\
o_t &= q_tS_t,
\end{aligned}
$$

这里将单个 token 的向量记为行向量，即 $q_t,k_t\in\mathbb{R}^{1\times d_k}$、
$v_t\in\mathbb{R}^{1\times d_v}$，因此
$S_t\in\mathbb{R}^{d_k\times d_v}$。由于状态 $S_t$ 的大小与序列长度无关，Linear Attention 可以像 RNN 一样仅维护固定大小的状态。

在递推形式下，Linear Attention 按照 token 顺序更新 $S_t$，每步仅维护固定大小的状态，总计算量关于序列长度为线性，特别适合 autoregressive decoding，但序列维度上的依赖链限制了 GPU 并行度。

Linear Attention 的固定大小状态同时也是它的限制。所有历史关联都被叠加到同一个矩阵中，
当不同 key 相似或状态容量逐渐饱和时，旧关联容易互相干扰；纯加法更新也没有主动遗忘或覆盖旧信息的机制。
后续的 GLA、DeltaNet 和 Gated DeltaNet 都是在改进这个状态更新规则。

#### GLA, DeltaNet 与 Gated DeltaNet

[GLA（Gated Linear Attention）](https://arxiv.org/abs/2312.06635)在状态递推中加入
由当前输入决定的遗忘门。用简化的标量门表示，其更新为：

$$
S_t = \alpha_t S_{t-1} + k_t^{\mathsf T}v_t,\qquad \alpha_t\in(0,1).
$$

实际的 GLA 可以使用更细粒度的门。与固定 decay 相比，数据相关的 $\alpha_t$ 能让模型根据输入决定
保留多少历史状态：$\alpha_t$ 接近 1 时记忆基本保留，接近 0 时快速遗忘。它缓解了状态无限累积的问题，
但写入仍然是简单的外积相加，无法针对与当前 key 冲突的旧关联做精确修改。

[DeltaNet](https://arxiv.org/abs/2406.06484)把加法写入替换为 delta rule。它先用当前 key
从旧状态中读取预测值，再根据预测与目标 value 之间的误差更新状态：

$$
\begin{aligned}
\widehat{v}_t &= k_tS_{t-1}, \\
S_t &= S_{t-1} + \beta_tk_t^{\mathsf T}(v_t-\widehat{v}_t),
\qquad \beta_t\in(0,1).
\end{aligned}
$$

在 key 已归一化的情况下，可以把 $\beta_t$ 理解为写入强度：$\beta_t=0$ 时不修改状态，
$\beta_t=1$ 时当前 key 对应的旧预测被完整替换为 $v_t$。因此 DeltaNet 能够定向修正已有的
key-value 关联，而不只是继续向状态中叠加信息。不过，它没有独立的全局遗忘门，难以快速清除大量已经无关的记忆。

[Gated DeltaNet](https://arxiv.org/abs/2412.06464)把门控遗忘与 delta update 组合起来：

$$
\begin{aligned}
\overline{S}_t &= \alpha_tS_{t-1}, \\
S_t &= \overline{S}_t
      + \beta_tk_t^{\mathsf T}(v_t-k_t\overline{S}_t), \\
o_t &= q_tS_t.
\end{aligned}
$$

等价地，状态更新也可以写为
$S_t=\alpha_t(I-\beta_tk_t^{\mathsf T}k_t)S_{t-1}+\beta_tk_t^{\mathsf T}v_t$。
$\alpha_t$ 负责控制旧状态整体保留多少，$\beta_t$ 负责控制当前 key 对应的关联修改多少。
两种机制分工不同且可以互补：门控适合快速释放状态容量，delta rule 适合精确覆盖特定记忆。

| 模型 | 状态更新的核心机制 | 能力与局限 |
| --- | --- | --- |
| Linear Attention | 直接累加 $k_t^{\mathsf T}v_t$ | 简单高效，但旧关联只能累积 |
| GLA | 写入前对旧状态施加数据相关 decay | 能主动遗忘，但写入仍是加法 |
| DeltaNet | 根据预测误差定向修改关联 | 能精确覆盖，但缺少独立的快速遗忘机制 |
| Gated DeltaNet | decay + delta rule | 同时支持快速遗忘与定向更新 |

### Chunk-wise Parallel

递推形式虽然数学形式简洁，但是由于存在逐 token 的依赖链，GPU 并行度受限。而完全并行的形式 $O=(QK^{\mathsf T}\odot M)V$ 需要 $O(L^2d)$ 的计算量, 失去了线性复杂度的优势。因此 [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) 提出了一种折中方案 Chunk-wise Parallel，它把序列切成长度为 $C$ 的 chunk，每次 S 的更新只在 chunk 之间进行，由此让递推的计算由向量计算转换为以 chunk 为单位的矩阵计算。

$$
\begin{aligned}
S_{[i+1]} &= S_{[i]} + \sum_{j=iC}^{(i+1)C - 1} k_{j}^{\mathsf T} v_{j}\\
O_{[i]} &= Q_{[i]} S_{[i]} + ((Q_{[i]} K_{[i]}^{\mathsf T}) \odot M) V_{[i]}
\end{aligned}
$$

一般而言，$C$ 决定串行度、并行度和片上资源占用之间的权衡。$C=1$ 退化为逐 token recurrence，
$C=L$ 接近完全并行形式。本实验的**算法 chunk size 固定为 $C=64$**，门控前缀和与 $A$ 的预处理
都按照这一大小进行。

关于 GDN chunk-wise parallel 形式推导的更多细节可以参考 GDN 论文的 Section 3.3。这里也提供一个看上去冗长且复杂但是不需要什么二级结论和经验的推导过程，供有兴趣的同学参考：[从零开始的 GDN 推导](https://blogs.erix025.me/EfficientAI/GDN_from_scratch/)。

### TileLang

为了减少 GPU kernel 开发的复杂度，本实验使用 [TileLang](https://tilelang.com/) 作为开发语言。TileLang 是一种面向高性能计算的领域专用语言，能够让开发者专注于算法本身，而不必过多关注底层硬件细节。关于 TileLang 的语法和使用方法，请参考官方文档和[官方仓库中的示例](https://github.com/tile-ai/tilelang/tree/main/examples/gemm)。

虽然 TileLang 提供了很多便利，但仍然需要开发者理解 GPU 的计算模型和性能优化方法。你需要根据 GDN 的计算特点，合理使用 TileLang 提供的功能，设计高效的 kernel。关于 GPU 的计算模型和性能优化方法这里不再赘述，请参考 7 月 13 日下午课程 “GPU 编程基础” 的课程内容。

### GPU Profiling

在我们的理论课中已经介绍了 Profiling 的重要性，它能够帮助你定位性能瓶颈，指导优化方向。你可以使用 NVIDIA Nsight Compute 和 Nsight Systems 等工具对你的 kernel 进行性能分析。

- Nsight Compute：用于分析单个 kernel 的性能指标，如吞吐量、延迟、访存效率等。你可以通过它来了解你的 kernel 在计算和访存方面的表现，从而找到优化的切入点。
- Nsight Systems：用于分析整个应用的性能，包括 kernel launch 的开销、CPU 和 GPU 的协同工作等。如果你的解法由多个 kernel 组成，你可以使用它来分析各个 kernel 的时间占比，找出性能瓶颈。

## 实验任务

### 计算过程描述

对于 GDN 来说，chunk-wise parallel 包括以下几个阶段：

1. 在每个 chunk 内计算门控量的前缀和 $g^{\mathrm{cumsum}}$ 和矩阵 $A$；
2. 计算中间量 $U$ 和 $W$；
3. 递推更新状态 $S$；
4. 计算输出 $O$。

对第 $c$ 个 chunk，令其有效长度为 $\ell_c=\min(C,L-cC)$。下式中的 $Q,K,V$、
$\Gamma$ 和 $B$ 均表示该 chunk 内的有效部分，并省略 chunk 下标：

$$
\begin{aligned}
A &= (I+\mathrm{StrictLower}(B\Gamma KK^\top \Gamma^{-1}))^{-1}\\
U &= ABV, \qquad W = AB\Gamma K\\
S_{[c+1]} &= \gamma_{\ell_c}S_{[c]}
  +\gamma_{\ell_c}K^\top\Gamma^{-1}(U-WS_{[c]})\\
O_{[c]} &= \frac{1}{\sqrt{d_k}}\left[
  \Gamma QS_{[c]}
  +\Gamma\mathrm{Lower}(QK^\top)\Gamma^{-1}(U-WS_{[c]})
\right]
\end{aligned}
$$

对于整个长度为 $L$ 的序列，总共有 $\lceil L/C\rceil$ 个 chunk。最后一个 chunk 可能不足 64 个
token，此时 $\gamma_{\ell_c}$ 表示最后一个有效 token 对应的门控累积值。框架中 $A$ 的最后一维
仍按 64 存储，但尾块只有前 $\ell_c$ 列参与计算；实现必须屏蔽其余无效位置。

为了简化，$g^{\mathrm{cumsum}}$ 和 $A$ 的计算已经给出且位于核心计时区间之外。你需要实现并优化
$U$、$W$、$S$ 和 $O$ 的全部计算。

### 输入输出定义

记 batch size、序列长度、query/key head 数和 value head 数分别为
$B,T,H_q,H_v$，key 和 value 的 head dimension 分别为 $d_k,d_v$。函数
`gdn_prefill_forward` 的接口如下：

| 张量 | 形状 | 数据类型 | 含义 |
| --- | --- | --- | --- |
| `q`, `k` | `[B, T, Hq, dk]` | BF16 | 已完成 L2 normalization 的 query 和 key |
| `v` | `[B, T, Hv, dv]` | BF16 | value |
| `g_cumsum` | `[B, T, Hv]` | FP32 | 每个 64-token chunk 内独立计算的 log-space 门控前缀和 |
| `beta` | `[B, T, Hv]` | FP32 | delta rule 的写入强度 |
| `A` | `[B, T, Hv, 64]` | BF16 | 分块 KKT 下三角矩阵的逆 |
| `initial_state` | `[B, Hv, dk, dv]` | FP32 | 可选的初始状态；未提供时使用零矩阵 |
| 返回的 `output` | `[B, T, Hv, dv]` | BF16 | prefill 输出 |
| 返回的 `final_state` | `[B, Hv, dk, dv]` | FP32 | 最后一个 chunk 更新后的状态 |

当前评测固定 $C=64$、$d_k=d_v=128$。上述维度顺序和数据类型均属于函数接口的一部分；
评测会同时检查 `output` 和 `final_state`。

完整 forward 的原始门控输入记为 `raw_g`，满足

$$
\mathrm{raw\_g}_{c,r}=\log\alpha_{c,r},\qquad
g^{\mathrm{cumsum}}_{c,r}
=\sum_{j=1}^{r}\mathrm{raw\_g}_{c,j}
=\log\gamma_{c,r},\qquad
\gamma_{c,r}=\exp\left(g^{\mathrm{cumsum}}_{c,r}\right).
$$

这里的前缀和会在每个 chunk 开头重新从零开始计算。函数接收的是
`g_cumsum`，而不是 $\alpha$、`raw_g` 或线性空间的 $\gamma$。在公式中使用 $\gamma$ 时需要对
`g_cumsum` 取 `exp`。$\Gamma=\mathrm{Diag}(\gamma)$、$B=\mathrm{Diag}(\beta)$ 只是为了方便
表达矩阵运算，实际实现不需要显式构造这两个对角阵。

同时为了保证算子的通用性，我们还会考虑以下几种输入变种：

- **给定 $S_0$**：部分 case 会传入非零的 `initial_state`，其余 case 从零状态开始；两种情况都必须正确处理。
- **GVA（Group-Value Attention）**：可能有 $H_v>H_q$，且保证 $H_v$ 能被 $H_q$ 整除。令组大小
  $G=H_v/H_q$，第 $h_v$ 个 value head 使用的 query/key head 为
  $h_{qk}=\lfloor h_v/G\rfloor$。也就是说，每个 query/key head 连续对应 $G$ 个 value head；
  `g_cumsum`、`beta`、`A`、输出和状态均按 $H_v$ 编号。

### 任务要求

你的任务是使用 TileLang 完成 GDN 的 prefill forward kernel 实现，基于给定的
`g_cumsum` 和 $A$ 完成 $S$ 和 $O$ 的计算，并在保证结果正确的前提下进行优化。

- 实验框架中已给出 PyTorch 的参考实现，你需要在 TileLang 中实现相同的计算过程，并确保输出结果与参考实现一致。
- 实验框架中已经给出 `g_cumsum` 和 $A$ 的计算，你不需要进行修改。这两部分的计算 kernel 也是使用 TileLang 实现的，可以作为语法参考。
- 你需要在实验报告中说明你的优化思路和依据，并尝试分析优化后的性能提升及其原因。
- 实验框架中会给出目前主流开源实现的性能数据，你可以参考这些数据来评估你的优化效果，并尝试挑战更高的性能。

!!! tip "向上游贡献"

    如果你的实现在某些 case 上取得了稳定的性能提升，欢迎整理成可复现的 benchmark 和实现，尝试向相关开源项目提交 issue 或 pull request。

## 优化方向

### Profiling

Profiling 是性能优化的基础。通过 profiling 可以了解 kernel 的性能瓶颈，从而有针对性地进行优化。建议在每次优化后都进行 profiling，分析优化前后的性能变化，并在报告中说明你的分析和结论。

### 等价数学变换

相信大家也看到了 GDN 的公式还是比较复杂的，在 GDN 论文中的公式推导更多考虑与之前 linear attention 的继承性和数学上的可解释性，而在 GPU 实现上，可能有一些等价的数学变换来改变计算顺序、减少中间量、降低复杂度或增加并行度。

- 你可以尝试自己推导一些等价的数学变换，或者参考 FlashQLA、FLA、FlashInfer 等开源实现中的优化方法，看看是否有可以借鉴的地方。
- 不管是自己的推导还是参考别人的方法，都需要在实验报告中说明你的思路和依据，并且在代码中注明来源。
- 如果你对数学变换不熟悉，也可以先保持原有的公式不变，从其他优化方向入手。

### Kernel Fusion

在 GDN 的论文推导中，公式被拆分成了多个计算过程。但如果在 GPU 上直接将每一步都作为单独的 kernel 来执行，就会导致中间结果需要在 global memory 中读写，增加了访存开销和 kernel launch 的开销。因此 kernel fusion 是一个常见的优化手段，它通过将多个计算过程融合到一个 kernel 中，减少中间结果的读写和 kernel launch 的次数，从而提高性能。

- 你可以尝试将多个计算过程融合到一个 kernel 中来减少访存开销和 kernel launch 的开销。
- 如果你进行了 kernel fusion，需要在实验报告中说明你的思路和依据，尝试分析融合后的性能提升及其原因，并且注明每个 kernel 对应的计算过程。

### Shared Memory

Shared Memory 是 GPU 上的一种高速缓存，它可以被同一个 SM 内的所有线程访问。合理使用 Shared Memory 可以减少 global memory 的访问次数，从而提高性能。因此在 GDN 的计算中，将输入数据和中间结果放入 Shared Memory 中进行计算，可以减少 global memory 的访问次数，有效提高性能。

- 建议尽可能的使用 Shared Memory 来存储输入数据和中间结果，减少 global memory 的访问次数。
- 如果你使用了 Shared Memory，请确保你理解 Shared Memory 的大小限制并且在报告中说明你对 Shared Memory 空间的规划。

### ping-pong buffer / multi-buffering

在 GDN 的计算中，S 矩阵的递推更新是一个串行依赖的过程。如果每次更新 S 矩阵都需要等待相关数据的读入和写出，就会导致 GPU 的计算单元闲置，降低了运算效率。因此可以考虑使用 ping-pong buffer 或 multi-buffering 的方式，在第 t 步计算的同时准备第 t+1 步甚至以后的数据，从而实现计算和访存的重叠，提高 GPU 的利用率。

- 你可以尝试使用 ping-pong buffer 或 multi-buffering 的方式，在第 t 步计算的同时准备第 t+1 步甚至以后的数据，从而实现计算和访存的重叠，提高 GPU 的利用率。
- 这里需要注意的是通常会使用两个/多个 buffer 来存储，因此需要在报告中说明你对 buffer 的规划和使用方式，并且分析其对性能的影响。
- 这里需要注意对数据传输阶段和计算阶段的依赖同步问题，请在报告中说明你是如何处理同步的。
- 这里的实现可能与下一节的 warp specialization 紧密相关，因此可以结合考虑。

### 手动 warp specialization

在 GPU 上，warp 是最小的执行单元。因此可以通过 warp specialization 的方式，将不同的计算过程分配给不同的 warp，从而实现更精细的控制和 warp 间计算的重叠，提高 GPU 的利用率。

特别的，在 GDN 的计算中不仅会涉及到基于 Tensor Core 的矩阵乘法，还会涉及到一些在 CUDA Core 上执行的 element-wise 计算。可以考虑利用 warp specialization 将 CUDA Core 上的计算掩盖在 Tensor Core 上的计算中，从而实现更优的性能。

- 你可以尝试使用 warp specialization 的方式，将不同的计算过程分配给不同的 warp，从而实现更精细的控制和 warp 间计算的重叠，提高 GPU 的利用率。
- 这里需要注意的是 warp specialization 的实现可能会涉及到 warp 内线程的分工和 warp 间的同步，请在报告中明确说明你的每一个 warp 的分工和每次同步发生时对应计算流程的关系。
- 这里的实现可能与上一节的 ping-pong buffer / multi-buffering 紧密相关，因此可以结合考虑。

### 其他优化手段

上述优化手段仅供参考，并不意味着你必须使用这些方法。你可以根据自己的理解和实验结果，尝试其他优化手段。如果你有其他优化思路，也可以在实验报告中说明你的思路和依据，并尝试分析其对性能的影响。

下面列举一些可能的优化手段，供参考：

- 更多访存优化手段
- 更合理的 thread block 划分
- ...

## 评测方式

我们会提供 PyTorch reference，以及统一的正确性测试和性能测试。评测程序会将 TileLang 实现的结果与 reference 进行比较；只有通过正确性检查的 case 才会参与性能评分。

性能评测包含两种计时区间：

- **forward 端到端时间**：从 `raw_g` 开始，包含 `g_cumsum`、$A$、$U$、$W$、$S$ 和 $O$
  的全部计算，以及中间结果读写和 kernel launch 的开销。**这部分时间仅供参考与其他实现对比，最终评分不以此为准**。
- **核心计算时间**：只统计学生函数内 $U$、$W$、$S$ 和 $O$ 的计算，不包括
  `g_cumsum` 和 $A$ 的预处理。学生函数内的张量分配、数据转换和 kernel launch 均包含在内。
  **这部分时间将作为最终评分的主要依据**。

正确性评测和性能评测都会在多组不同的输入 shape 下进行。最终的性能分数是各组 shape 的加权平均。

FlashQLA 将作为本实验的主要性能基线，同时也会给出 FLA 和 FlashInfer 在相同 case 下的结果作为
参考。性能分根据多个 case 上的综合表现计算，而不是由某一个 case 的最快结果决定。未通过正确性
检查、运行出错或超时的 case 不计性能分。

## 实验框架

实验框架代码位于 [HPC101 课程仓库](https://github.com/ZJUSCT/HPC101) 的 `src/lab3` 路径下，详细的文件结构请见 `README.md`。

- 你需要完成的代码文件位于 `student/tilelang_fwd.py`，你能且仅能修改这个文件。
- 评测代码位于 `evaluation/run.py`，你可以直接运行该文件以执行测试。
- 为了简化运行流程以便于 profile，可以使用 `python evaluation/run.py --profile` 来只运行一次你的实现。你也可以自己参考 run.py 写一个简单的调用来 profile。

## 如何获取计算资源

1. 登陆[实验平台](https://platform.s.zjusct.io)
2. 创建预设为 `x86-5418Y` 的 DevPod。
3. 在 DevPod 内进行代码开发。
4. 由于 GPU 资源有限，DevPod 内不提供 GPU，你可以使用 `hpc submit` 提交任务来调试。
5. Lab 3 的镜像为 `harbor.s.zjusct.io/public/hpc101-lab3:v26.0`

你可以参考下面的命令

```bash
# 在 DevPod 中运行
git clone https://github.com/zjusct/hpc101
cd hpc101/src/lab3

# 运行
hpc submit -p lab3 "python evaluation/run.py"

# 使用 ncu/nsys profile 你的程序
hpc submit -p lab3 "ncu -o <output_name> python evaluation/run.py --profile"
hpc submit -p lab3 "nsys profile -o <output_name> python evaluation/run.py --profile"
```

更详细的实验平台使用教程请参考文档 [集群使用](https://hpc101.zjusct.io/guide/)

## 实现要求

- 使用 TileLang 完成题目指定的 forward 计算，禁止调用其他实现完成被测计算；
- 保持给定的函数签名、输入输出和计算语义；
- 可以自行决定 kernel 的拆分与融合方式，也可以为不同 shape 选择不同配置；
- 不得修改 PyTorch reference、测试程序和计时逻辑；
- 不得硬编码测试数据或输出结果，也不得利用评测程序的漏洞绕过计算；
- 实验框架所需的依赖库和工具链已经在课程环境中安装好，禁止使用未说明的软件包或自建工具链。
- 你可以阅读 FlashQLA、FLA 和 FlashInfer 等开源实现，了解已有的算法和优化方法。实验报告中应注明参考过的实现或资料，并说明自己的修改
与优化。

## 提交要求

你需要提交以下内容：

1. **实现代码**：TileLang forward 实现及其运行所需的辅助代码；
2. **实验报告**：PDF 格式，说明你的优化思路、依据和结果。具体要求见下一节；

评测时只会收取指定目录中的文件。新增的辅助文件也需要放在该目录中，并保证程序可以在课程提供的
环境中从干净目录直接运行。

请勿提交测试数据、编译产物、模型权重或 profiler 生成的大文件。

## 实验报告

实验报告不需要重复大段背景知识，重点说明你做了哪些优化、为什么这样做，以及结果如何。报告至少应包含：

- 算法与数据依赖分析；
- baseline 的性能结果与瓶颈分析；
- 每项主要优化的设计、实现和收益；
- 最终性能数据、测试环境和运行命令；
- 与 FlashQLA 等各实现在各个 case 上的对比；
- 尝试过但未采用的方案及其原因。

性能数据需要注明对应的输入 shape、统计方式和单位。建议使用表格或图表展示各阶段优化前后的变化；
较长的代码、命令输出和 profiler 结果可以作为附件，不需要全部放在正文中。

注意：实验代码可用 Agent 辅助完成，因此请认真撰写实验报告，在实验报告中重点体现你的思路、分析和思考，而不是仅仅用 Agent 生成报告。这将成为你在实验中的主要评分依据。

## OJ 自动评测

!!! warning "施工中"

    目前 OJ 自动评测系统仍在建设中😭，我们会尽快发布。

## 参考资料

- [TileLang: Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels](https://tilelang.com/)
- [Gated DeltaNet: A Gated Linear Attention Model with Delta Update](https://arxiv.org/abs/2412.06464)
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)
- [FlashQLA: High-Performance Linear Attention Kernel Library built on TileLang](https://github.com/QwenLM/FlashQLA)
- [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)
- [Flash Linear Attention: Efficient implementations for emerging model architectures](https://github.com/fla-org/flash-linear-attention)
