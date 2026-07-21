# FlashQLA 直接接收 log-space 的 raw_g。
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
    # 只有打开 reference benchmark 才会导入它，所以普通测试不会等待它的 JIT。
    from flash_qla import chunk_gated_delta_rule as official_forward

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
        auto_cp=True,
    )
