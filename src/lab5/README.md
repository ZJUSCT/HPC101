# 实验五：Gemma4 端到端推理优化

本目录包含 Lab 5 的推理框架、公开数据集和实验脚本。实验主要分为两部分：

1. 使用 GPTQ 将 Gemma4-12B 从 BF16 量化为 W4A16；
2. 在 10 GiB GPU 环境中优化端到端生成吞吐量。

完整的实验背景、算法原理和提交要求请阅读实验手册。本文只说明如何使用脚本和运行实验。

## 1. 目录说明

```text
.
├── config.yaml                        # 配置文件
├── datasets
│   ├── calibration-256.jsonl          # 参考校准数据集
│   ├── performance_public.jsonl       # 公开性能评测数据集
│   ├── performance_small.jsonl        # 小规模性能测试数据集
│   └── quality_public.jsonl           # 公开精度评测数据集
├── pyproject.toml
├── README.md                          # 实验框架和工具脚本说明
├── results
│   └── bf16-public-quality.json       # 公开精度评测数据集的 BF16 精度评测结果
├── scripts
│   ├── evaluate_quality.py            # 量化精度评测脚本
│   ├── quantize.py                    # 量化框架入口
│   └── run_generation_queue.py        # 推理框架入口
└── src
    └── hpc101_infer                   # 实验框架
```

下文命令均假设当前目录为仓库中的 `src/lab5/`。

## 2. 准备环境

在 DevPod 中，使用以下命令申请 Lab5 的实验容器并进入交互式环境：

```bash
hpc submit -p lab5 -g 1 -t 20m --interactive bash
```

容器中已经预装了包管理器 `uv` 和实验所需的 Python 依赖，进入容器后，你需要将你实现的 `hpc101_infer` 安装到虚拟环境中：

```bash
uv pip install -e . --no-deps
```

安装完毕后，请确认三个入口脚本可以正常启动：

```bash
python3 scripts/quantize.py --help
python3 scripts/evaluate_quality.py --help
python3 scripts/run_generation_queue.py --help
```

Gemma4-12B 的原始权重（BF16）已经挂载到容器中，下文使用环境变量简化命令：

```bash
export MODEL_DIR=/checkpoints/gemma-4-12b
export QUANT_DIR=/path/to/quantized-gemma-4   # 你可以自行指定输出目录
```

你可以通过 `nvidia-smi` 查看当前可用的 GPU 设备和显存使用情况，正常情况下你应当能看到一个 10 GiB 显存的 MIG 实例（1/7 张 H800）。如果需要指定具体的 GPU，可通过 `CUDA_VISIBLE_DEVICES` 指定分配到的 GPU，MIG 实例的 UUID 可以通过 `nvidia-smi -L` 查询：

```bash
export CUDA_VISIBLE_DEVICES=<MIG-UUID>
```

## 3. 运行量化实验

量化脚本可以从共享 YAML 文件的 `quantization` 段读取 `QuantizationConfig`：

```bash
python3 scripts/quantize.py \
  --config config.yaml \
  --model "$MODEL_DIR" \
  --output "$QUANT_DIR"
```

配置文件的 `quantization` 段提供量化算法参数，外层 `calibration` 段可以指定校准数据路径、样本上限、micro batch size 和最大 token 数。模型路径、输出目录和设备仍必须通过命令行指定。显式命令行参数优先于配置文件。

脚本保留 `QuantizationConfig.from_yaml()` 返回的完整对象，并只覆盖命令行中显式指定的量化字段。学生可以在 `QuantizationConfig` 中新增字段、在 YAML 的 `quantization` 段设置并在量化实现中使用，无需修改 CLI。

### 3.1 使用 RTN 检查量化流程

RTN 不需要校准数据，适合先检查 checkpoint 的读取、打包和保存流程：

```bash
python3 scripts/quantize.py \
  --config config.yaml \
  --model "$MODEL_DIR" \
  --output /path/to/gemma-4-12b-rtn-w4a16 \
  --algorithm rtn \
  --group-size 128 \
  --scale-dtype float16 \
  --device cpu
```

### 3.2 使用 GPTQ 生成提交模型

完成 `src/hpc101_infer/quantization/methods/gptq.py` 后，使用公开校准集运行 GPTQ：

```bash
python3 scripts/quantize.py \
  --config config.yaml \
  --model "$MODEL_DIR" \
  --output "$QUANT_DIR" \
  --device cuda \
  --verbose
```

常用量化参数：

| 参数                       | 作用                                   | 默认值    |
| -------------------------- | -------------------------------------- | --------- |
| `--config`                 | 共享 YAML 配置文件                     | 无        |
| `--algorithm`              | 量化方法，可选 `rtn`、`gptq`           | `rtn`     |
| `--group-size`             | 每组包含的输入列数，可选 `64` 或 `128` | `128`     |
| `--symmetric`              | 使用对称量化                           | 开启      |
| `--asymmetric`             | `--no-symmetric` 的兼容别名            | 关闭      |
| `--scale-dtype`            | scale 的数据类型                       | `float16` |
| `--calibration-limit`      | 最多读取的校准样本数                   | 不限制    |
| `--max-calibration-tokens` | 每个 Linear 最多收集的有效 token 数    | `4096`    |
| `--gptq-block-size`        | GPTQ 分块处理的列数                    | `128`     |
| `--gptq-damp-percent`      | Hessian 阻尼比例                       | `0.01`    |
| `--max-shard-size-mib`     | 输出权重分片的大小上限                 | `1024`    |

量化完成后，`QUANT_DIR` 中会保存 packed INT4 权重、量化配置和 manifest。不要覆盖原始 BF16 checkpoint。

如果量化时显存不足，优先减小 `--max-calibration-tokens`，并保持 `--calibration-micro-batch-size 1`。修改参数后应重新进行精度评测。

## 4. 运行精度评测

公开精度评测使用 teacher forcing 计算平均 NLL。框架已提供同一数据集上的 BF16 结果：

```text
results/bf16-public-quality.json
```

直接评测量化模型并与 BF16 基准比较：

```bash
python3 scripts/evaluate_quality.py \
  --model "$QUANT_DIR" \
  --dataset datasets/quality_public.jsonl \
  --output results/gptq-public-quality.json \
  --reference results/bf16-public-quality.json \
  --max-delta-nll 0.16 \
  --linear-backend int4_reference \
  --device cuda \
  --dtype bfloat16 \
  --max-sequence-length 2048 \
  --chunk-size 128
```

重点检查输出 JSON 中的字段：

- `mean_nll`：量化模型在公开数据集上的平均 NLL；
- `delta_nll`：量化模型与 BF16 基准的 NLL 差值；
- `perplexity_ratio`：量化模型与 BF16 基准的困惑度比值；
- `passed`：`delta_nll` 是否不超过 `--max-delta-nll`，当实测 `delta_nll > max_delta_nll` 时，`passed` 为 `false`，脚本会以非零状态退出。

你可以将测试结果中的 `delta_nll` 与实验手册中任务一的得分曲线进行对照，确认量化精度是否满足要求。

调试时可以使用 `--limit N` 只评测前 `N` 条记录；正式记录结果时不要限制样本数量。`--chunk-size` 只控制每次产生 logits 的 token 数，可在显存不足时适当减小。

精度数据集为 JSONL，每行可以提供文本或冻结后的 token 序列：

```json
{"text": "Zhejiang University is located in Hangzhou."}
{"input_ids": [2, 123, 456, 789, 1]}
```

## 5. 运行端到端性能实验

生成脚本支持从共享 YAML 文件的 `engine` 段读取 `EngineConfig`。模型、输入输出文件和生成参数必须通过命令行指定；配置文件中的其他顶层配置段不会被生成脚本使用：

```bash
python3 scripts/run_generation_queue.py \
  --config config.yaml \
  --model "$QUANT_DIR" \
  --input datasets/performance_public.jsonl \
  --output results/gptq-public-generation.jsonl \
  --summary-output results/gptq-public-summary.json
```

命令行中显式提供的引擎参数优先于配置文件，例如 `--batch-size 4` 会覆盖 `engine.scheduler_batch_size`。

脚本会保留 `EngineConfig.from_yaml()` 返回的完整配置对象，并只覆盖命令行中显式指定的字段。因此，学生在 `EngineConfig` 中增加新的 dataclass 字段、在 YAML 的 `engine` 段设置该字段并在引擎实现中使用后，不需要修改运行脚本。

也可以完全使用命令行运行生成脚本：

```bash
python3 scripts/run_generation_queue.py \
  --model "$QUANT_DIR" \
  --input datasets/performance_public.jsonl \
  --output results/gptq-public-generation.jsonl \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-sequence-length 2048 \
  --seed 0
```

性能数据集采用 JSONL 格式，每行是一条请求：

```json
{"prompt": "Hello", "max_new_tokens": 32}
{"input_ids": [2, 3, 4], "max_new_tokens": 16, "stop_token_ids": [1]}
```

每条请求必须提供 `prompt` 或 `input_ids`。如果请求没有 `max_new_tokens`，脚本使用命令行参数 `--max-new-tokens`，默认值为 `32`。

生成脚本参数：

| 参数                                                 | 作用                                                                                | 默认值或配置字段                                  |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| `--config PATH`                                      | 共享 YAML 配置文件；脚本只读取其中的 `engine` 段，显式命令行参数优先                | 无                                                |
| `--model PATH`                                       | 模型或量化 checkpoint 目录；不能从配置文件读取                                      | 必填                                              |
| `--input PATH`                                       | 输入请求 JSONL；使用 `-` 时从标准输入读取                                           | 必填                                              |
| `--output PATH`                                      | 逐请求生成结果 JSONL；使用 `-` 时写入标准输出                                       | `-`                                               |
| `--summary-output PATH`                              | 将标准错误中的整体性能摘要额外写入 JSON 文件                                        | 无                                                |
| `--device DEVICE`                                    | 推理设备，例如 `cuda` 或 `cpu`                                                      | `engine.device`，缺省为 `cuda`                    |
| `--dtype DTYPE`                                      | 推理 dtype，可选 `bfloat16`、`float16`、`float32`                                   | `engine.dtype`，缺省为 `bfloat16`                 |
| `--batch-size N`                                     | 调度器每批处理的最大请求数                                                          | `engine.scheduler_batch_size`，缺省为 `1`         |
| `--max-batch-size N`                                 | KV cache 可容纳的最大 batch；显式设置 `--batch-size` 且未设置本参数时，两者取相同值 | `engine.max_batch_size`，缺省为 `1`               |
| `--max-sequence-length N`                            | 单条请求的 prompt 与生成 token 的最大总长度                                         | `engine.max_sequence_length`，缺省为 `4096`       |
| `--attention-backend BACKEND`                        | 注意力实现；当前仅支持 `eager`                                                      | `engine.attention_backend`，缺省为 `eager`        |
| `--linear-backend BACKEND`                           | Linear 实现，可选 `bf16`、`int4_reference`                                          | `engine.linear_backend`，缺省为 `bf16`            |
| `--scheduler-backend BACKEND`                        | 调度器实现；当前仅支持 `static_batch`                                               | `engine.scheduler_backend`，缺省为 `static_batch` |
| `--max-new-tokens N`                                 | 请求记录未提供 `max_new_tokens` 时使用的默认生成长度；不能从配置文件读取            | `32`                                              |
| `--seed N`                                           | 采样随机种子                                                                        | `engine.seed`，缺省为 `0`                         |
| `--synchronize-metrics` / `--no-synchronize-metrics` | 开启或关闭各指标计时区间前后的设备同步                                              | `engine.synchronize_metrics`，缺省为开启          |
| `--progress` / `--no-progress`                       | 开启或关闭请求完成进度条；不能从配置文件读取                                        | 开启                                              |

输出请求结果写入 `--output` 指定的 JSONL 文件，每行对应一条请求。整体性能摘要始终写入标准错误；指定 `--summary-output` 后，同一份摘要还会写入独立的 JSON 文件。摘要包括：

- `requests`：完成的请求数；
- `generated_tokens`：全部请求生成的 token 总数；
- `elapsed_s`：处理全部请求的总时间；
- `requests_per_s`：每秒完成的请求数；
- `generated_tokens_per_s`：每秒生成的 token 数；
- `batch_size`：本次运行使用的 batch size。

实验二的性能指标为 `elapsed_s`，你可以对照实验手册中的得分曲线确认优化效果。

测试不同 batch size 时，建议分别保存请求结果和性能摘要：

```bash
python3 scripts/run_generation_queue.py \
  --model "$QUANT_DIR" \
  --input datasets/performance_public.jsonl \
  --output results/generation-bs4.jsonl \
  --summary-output results/generation-bs4-summary.json \
  --batch-size 4 \
  --max-sequence-length 2048
```

如果在 `performance_public.jsonl` 数据集上测试时显存不足，可以先使用 `performance_small.jsonl` 进行测试。

## 6. 推荐实验流程

1. 先用 RTN 跑通量化、加载和精度评测闭环；
2. 实现 GPTQ，并生成 W4A16 checkpoint；
3. 在 `quality_public.jsonl` 上确认量化精度；
4. 用 `performance_public.jsonl` 记录未优化版本的性能基线；
5. 逐项实现显存、算子或调度优化；
6. 每次修改后重新检查精度，并使用相同参数重复性能测试；
7. 保存最终精度 JSON、生成结果 JSONL 和性能日志，用于实验报告与提交。

为了让性能结果可比较，每组实验应固定模型 checkpoint、数据集、GPU、`batch-size`、`max-sequence-length`、随机种子和计时选项。有条件的情况下，建议先预热一次，再重复运行多次并报告稳定结果。

## 7. 常见问题

### CUDA out of memory

- 减小性能实验的 `--batch-size` 或 `--max-sequence-length`；
- 减小量化时的 `--max-calibration-tokens`；
- 确认只使用分配到的 MIG 实例，且没有残留进程占用显存。

### 精度评测提示 reference 不匹配

`evaluate_quality.py` 会检查数据集哈希、长度 buckets 和评测配置。请使用未修改的 `datasets/quality_public.jsonl`，并保持与 BF16 reference 相同的 buckets。

### 无法加载量化 checkpoint

确认量化流程完整结束，输出目录中包含 `quantization_config.json`、manifest、模型配置、tokenizer 和全部权重分片。
