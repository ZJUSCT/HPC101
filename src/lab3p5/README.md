# FusedAddRmsNorm（Lab 3.5）

请选择 Ascend C、Triton-Ascend 或 TileLang-Ascend 中的一种实现 `fused_add_rmsnorm`。正式评测会使用隐藏 case，请勿针对公开 shape 硬编码。

## 提交 NPU 任务

在 `lab3p5/` 根目录使用 `hpc submit` 将任务提交到 `lab3p5` NPU 分区。任务默认使用当前目录作为工作目录。

### Ascend C

```bash
hpc submit -p lab3p5 bash checker/run.sh       # 正确性测试
hpc submit -p lab3p5 bash checker/profile.sh   # 性能测试
```

主要修改 `src/ascendc/op_kernel/` 和 `src/ascendc/op_host/` 下的算子代码。

### Triton-Ascend

```bash
hpc submit -p lab3p5 -e LANG=triton bash checker/run.sh
hpc submit -p lab3p5 -e LANG=triton bash checker/profile.sh
```

实现文件：`src/triton/fused_add_rmsnorm.py`。

### TileLang-Ascend

```bash
hpc submit -p lab3p5 -e LANG=tilelang bash checker/run.sh
hpc submit -p lab3p5 -e LANG=tilelang bash checker/profile.sh
```

实现文件：`src/tilelang/fused_add_rmsnorm.py`。

如需后台提交，可添加 `-d`，再使用 `hpc logs -f <job_id>` 查看日志。

## Checker 说明

- `checker/run.sh` 只执行正确性测试；不传参数时运行全部公开 case。
- `hpc submit -p lab3p5 bash checker/run.sh 2` 可单独运行 case 2，即 `256×1024`。
- `checker/profile.sh` 只测试并输出 student 算子的性能。
- 性能测试固定使用 case 2（`256×1024`），不接受 case 参数。
- 性能采集使用 `msprof op --warm-up=10`，最终输出一次 `Task Duration(us)`。

## 提交代码

验证完成后，通过 HPC101 平台上传所选语言对应的整个目录：

| 实现 | 提交目录 |
| --- | --- |
| Ascend C | `src/ascendc/` |
| Triton-Ascend | `src/triton/` |
| TileLang-Ascend | `src/tilelang/` |

只提交其中一个实现目录。不要提交 `checker/`、`env.sh`、`README.md`、构建产物或 profiling 输出目录。
