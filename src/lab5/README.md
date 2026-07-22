# 实验五：Gemma4 端到端推理优化

本目录包含 Lab 5 的推理框架、公开数据集和实验脚本。实验主要分为两部分：

1. 使用 GPTQ 将 Gemma4-12B 从 BF16 量化为 W4A16；
2. 在 10 GiB GPU 环境中优化端到端生成吞吐量。

完整的实验背景、算法原理和提交要求请阅读实验手册。本文只说明如何使用脚本和运行实验。

## 1. 目录说明

```text
.
├── datasets/
│   ├── calibration-256.jsonl          # 参考校准数据集
│   ├── performance_public.jsonl       # 公开性能评测数据集
│   ├── performance_small.jsonl        # 小规模性能测试数据集
│   └── quality_public.jsonl           # 公开精度评测数据集
├── results/
│   └── bf16-public-quality.json    # BF16 公开精度基准
├── scripts/
│   ├── quantize.py                 # W4A16 量化入口
│   ├── evaluate_quality.py         # NLL 精度评测入口
│   └── run_generation_queue.py     # 端到端生成与性能测试入口
└── src/hpc101_infer/               # 推理与量化框架
```

下文命令均假设当前目录为仓库中的 `src/lab5/`。

## 2. 准备环境

项目使用 `uv` 管理 Python 环境和依赖：

```bash
uv sync
```

确认三个入口脚本可以正常启动：

```bash
uv run python scripts/quantize.py --help
uv run python scripts/evaluate_quality.py --help
uv run python scripts/run_generation_queue.py --help
```

准备原始 Gemma4-12B Hugging Face checkpoint，并记住其目录。下文使用环境变量简化命令：

```bash
export MODEL_DIR=/path/to/gemma-4-12b
export QUANT_DIR=/path/to/gemma-4-12b-gptq-w4a16
```

`MODEL_DIR` 中应包含模型配置、tokenizer 和 safetensors 权重。脚本按本地 checkpoint 加载模型，不会自动下载缺失文件。

在 MIG 环境中，可通过 `CUDA_VISIBLE_DEVICES` 指定分配到的 GPU：

```bash
export CUDA_VISIBLE_DEVICES=<MIG-UUID>
```

## 3. 运行量化实验

### 3.1 使用 RTN 检查量化流程

RTN 不需要校准数据，适合先检查 checkpoint 的读取、打包和保存流程：

```bash
uv run python scripts/quantize.py \
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
uv run python scripts/quantize.py \
  --model "$MODEL_DIR" \
  --output "$QUANT_DIR" \
  --algorithm gptq \
  --calibration-file datasets/calibration-256.jsonl \
  --calibration-limit 256 \
  --calibration-micro-batch-size 1 \
  --max-calibration-tokens 16384 \
  --group-size 128 \
  --gptq-block-size 128 \
  --gptq-damp-percent 0.01 \
  --scale-dtype float16 \
  --device cuda \
  --verbose
```

常用量化参数：

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `--algorithm` | 量化方法，可选 `rtn`、`gptq` | `rtn` |
| `--group-size` | 每组包含的输入列数，可选 `64` 或 `128` | `128` |
| `--asymmetric` | 使用非对称量化；不指定时使用对称量化 | 关闭 |
| `--scale-dtype` | scale 的数据类型 | `float16` |
| `--calibration-limit` | 最多读取的校准样本数 | 不限制 |
| `--max-calibration-tokens` | 每个 Linear 最多收集的有效 token 数 | `4096` |
| `--gptq-block-size` | GPTQ 分块处理的列数 | `128` |
| `--gptq-damp-percent` | Hessian 阻尼比例 | `0.01` |
| `--max-shard-size-mib` | 输出权重分片的大小上限 | `1024` |

量化完成后，`QUANT_DIR` 中会保存 packed INT4 权重、量化配置和 manifest。不要覆盖原始 BF16 checkpoint。

如果量化时显存不足，优先减小 `--max-calibration-tokens`，并保持 `--calibration-micro-batch-size 1`。修改参数后应重新进行精度评测。

## 4. 运行精度评测

公开精度评测使用 teacher forcing 计算平均 NLL。框架已提供同一数据集上的 BF16 结果：

```text
results/bf16-public-quality.json
```

直接评测量化模型并与 BF16 基准比较：

```bash
uv run python scripts/evaluate_quality.py \
  --model "$QUANT_DIR" \
  --dataset datasets/quality_public.jsonl \
  --output results/gptq-public-quality.json \
  --reference results/bf16-public-quality.json \
  --max-delta-nll 0.2 \
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
- `passed`：`delta_nll` 是否不超过 `--max-delta-nll`。

实验要求公开测试集上的 `delta_nll < 0.2`。当 `passed` 为 `false` 时，脚本会以非零状态退出。

调试时可以使用 `--limit N` 只评测前 `N` 条记录；正式记录结果时不要限制样本数量。`--chunk-size` 只控制每次产生 logits 的 token 数，可在显存不足时适当减小。

精度数据集为 JSONL，每行可以提供文本或冻结后的 token 序列：

```json
{"text": "Zhejiang University is located in Hangzhou."}
{"input_ids": [2, 123, 456, 789, 1]}
```

## 5. 运行端到端性能实验

使用量化 checkpoint 和公开性能请求运行生成脚本：

```bash
uv run python scripts/run_generation_queue.py \
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

常用生成参数：

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `--input` | 输入 JSONL；使用 `-` 时从标准输入读取 | 必填 |
| `--output` | 输出 JSONL；使用 `-` 时写入标准输出 | `-` |
| `--batch-size` | 调度 batch size，同时决定 KV cache 的 batch 容量 | `1` |
| `--max-sequence-length` | prompt 与生成 token 的最大总长度 | `4096` |
| `--max-new-tokens` | 请求未指定时的默认生成长度 | `32` |
| `--seed` | 采样随机种子 | `0` |
| `--no-progress` | 关闭进度条 | 关闭 |
| `--no-synchronize-metrics` | 关闭各指标区间前后的设备同步 | 关闭 |

输出请求结果写入 `--output` 指定的 JSONL 文件。整体性能摘要写入标准错误，包括：

- `elapsed_s`：处理全部请求的总时间；
- `requests_per_s`：每秒完成的请求数；
- `generated_tokens_per_s`：每秒生成的 token 数；
- `batch_size`：本次运行使用的 batch size。

测试不同 batch size 时，建议保留完整日志：

```bash
uv run python scripts/run_generation_queue.py \
  --model "$QUANT_DIR" \
  --input datasets/performance_public.jsonl \
  --output results/generation-bs4.jsonl \
  --batch-size 4 \
  --max-sequence-length 2048 \
  2> results/generation-bs4.log
```

`--no-synchronize-metrics` 会改变细分指标的计时语义。进行正式对比时，应保持所有实验的该选项一致，且不得通过修改计时区间跳过必要工作。

如果在 `performance_public.jsonl` 数据集上测试时显存不足，可以先使用 `performance_small.jsonl` 进行测试。

## 6. 推荐实验流程

1. 先用 RTN 跑通量化、加载和精度评测闭环；
2. 实现 GPTQ，并生成 W4A16 checkpoint；
3. 在 `quality_public.jsonl` 上确认 `delta_nll < 0.2`；
4. 用 `performance_public.jsonl` 记录未优化版本的性能基线；
5. 逐项实现显存、算子或调度优化；
6. 每次修改后重新检查精度，并使用相同参数重复性能测试；
7. 保存最终精度 JSON、生成结果 JSONL 和性能日志，用于实验报告与提交。

为了让性能结果可比较，每组实验应固定模型 checkpoint、数据集、GPU、`batch-size`、`max-sequence-length`、随机种子和计时选项。建议先预热一次，再重复运行多次并报告稳定结果。

## 7. 常见问题

### CUDA out of memory

- 减小性能实验的 `--batch-size` 或 `--max-sequence-length`；
- 减小量化时的 `--max-calibration-tokens`；
- 确认只使用分配到的 MIG 实例，且没有残留进程占用显存。

### 精度评测提示 reference 不匹配

`evaluate_quality.py` 会检查数据集哈希、长度 buckets 和评测配置。请使用未修改的 `datasets/quality_public.jsonl`，并保持与 BF16 reference 相同的 buckets。

### 无法加载量化 checkpoint

确认量化流程完整结束，输出目录中包含 `quantization_config.json`、manifest、模型配置、tokenizer 和全部权重分片。
