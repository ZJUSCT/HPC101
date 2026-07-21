# FLA 与课程一样直接接收 log-space 的 raw_g。
import torch


SCALE = 128**-0.5


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as official_forward

    return official_forward(
        q,
        k,
        v,
        raw_g,
        beta,
        scale=SCALE,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
