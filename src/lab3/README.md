# Lab 3 代码说明

## 代码结构

```text
src/lab3
├── run.py                            # 测试入口与结果输出
├── evaluation                        # 测试、正确性检查与性能测量
│   ├── cases.csv                     # 公开测试 case 配置
│   └── support.py                    # 输入生成、正确性比较和计时辅助函数
├── preprocessing                     # 计时区外的 GDN 预处理
│   ├── tilelang_cumsum.py            # 使用 TileLang 计算 gate 的分块前缀和
│   └── tilelang_kkt_solve.py         # 使用 TileLang 计算分块 KKT 三角求解结果 A
├── references                        # 正确性基准与公开实现参考
│   ├── torch_gdr.py                  # PyTorch 高精度参考实现及可运行 starter
│   └── official                      # 官方高性能实现的统一接口适配
│       ├── fla.py                    # FLA 实现适配
│       ├── flash_qla.py              # FlashQLA 实现适配
│       └── flashinfer.py              # FlashInfer 实现适配
└── student                           # ★ 你的实现（只有此目录会被收取）
    └── tilelang_fwd.py               # 需要使用 TileLang 优化的 gdn_prefill_forward
```

## 修改范围限制

只能修改 `student/` 目录中的文件，也可以在该目录内新增辅助文件。评测时只会收取
`student/`，其余文件均会替换为原版，因此在其他目录中的修改不会生效。

你的实现必须使用 TileLang，不得调用 PyTorch reference、FlashQLA、FLA 或
FlashInfer 代替被测计算，也不得在计时区外预计算结果、硬编码测试输入或输出。
