# 这是你开始实现和优化 Lab 3 的入口。
# 如果暂时不清楚 W/U/state/output 的关系，先对照 references/torch_gdr.py 中 torch_gdn_prefill_forward 的 chunk 循环阅读并理解
import torch

from references.torch_gdr import torch_gdn_prefill_forward


# q/k: [B, T, Hq, 128] BF16
# v: [B, T, Hv, 128] BF16
# g_cumsum/beta: [B, T, Hv] FP32
# A: [B, T, Hv, 64] BF16
# initial_state/final_state: [B, Hv, 128, 128] FP32
def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 这个 PyTorch 调用展示了接口应返回什么，但速度很慢。优化时替换下面这次调用即可；你在这个函数内完成的分配和计算都会计入核心时间
    return torch_gdn_prefill_forward(
        q,
        k,
        v,
        g_cumsum,
        beta,
        A,
        initial_state,
        student_interface=True,
    )
