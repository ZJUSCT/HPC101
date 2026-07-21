import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


# 张量 shape
CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128

# 随机输入和正确性比较使用的固定参数
SEED = 42
RTOL = 5e-3
ATOL = 5e-3
DEFAULT_WARMUP = 10
DEFAULT_REPETITIONS = 100


@dataclass
class Case:
    name: str
    batch_size: int
    seqlen: int
    num_heads_qk: int
    num_heads_v: int
    use_initial_state: bool
    purpose: str = "benchmark"
    gate_mode: str = "random_decay"


@dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor | None


def load_cases(path: Path) -> list[Case]:
    with path.open(newline="", encoding="utf-8") as file:
        return [
            Case(
                name=row["name"],
                batch_size=int(row["batch_size"]),
                seqlen=int(row["seqlen"]),
                num_heads_qk=int(row["num_heads_qk"]),
                num_heads_v=int(row["num_heads_v"]),
                use_initial_state=row["use_initial_state"] == "true",
                purpose=row["purpose"],
                gate_mode=row["gate_mode"],
            )
            for row in csv.DictReader(file)
        ]


def make_inputs(case: Case, device: str | torch.device = "cuda") -> Inputs:
    # 固定随机种子后，你每次修改 kernel 都会面对相同输入，便于比较结果
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)

    qk_shape = (
        case.batch_size,
        case.seqlen,
        case.num_heads_qk,
        HEAD_DIM_K,
    )
    v_shape = (
        case.batch_size,
        case.seqlen,
        case.num_heads_v,
        HEAD_DIM_V,
    )
    gate_shape = (case.batch_size, case.seqlen, case.num_heads_v)

    # Q/K 按 head 做 L2 normalize，再转为题目规定的 BF16。
    q = F.normalize(
        torch.randn(qk_shape, device=device, generator=generator), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(qk_shape, device=device, generator=generator), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(v_shape, device=device, generator=generator).to(torch.bfloat16)
    # g 位于 log-space, 完整 forward 会先在每个 chunk 内做前缀和。
    g = F.logsigmoid(torch.randn(gate_shape, device=device, generator=generator)).div_(
        16
    )
    # mixed case 中一半 token 不衰减：线性 gate 为 1，对应 log-space gate 为 0。
    if case.gate_mode == "mixed":
        g[:, ::2, :] = 0
    beta = torch.sigmoid(torch.randn(gate_shape, device=device, generator=generator))

    initial_state = None
    if case.use_initial_state:
        initial_state = torch.randn(
            (
                case.batch_size,
                case.num_heads_v,
                HEAD_DIM_K,
                HEAD_DIM_V,
            ),
            device=device,
            generator=generator,
        )

    return Inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
    )


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
        msg=lambda message: f"{name}: {message}",
    )
