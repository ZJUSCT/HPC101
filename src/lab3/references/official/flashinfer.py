# FlashInfer 的接口与课程略有不同, gate 和 state 有两次转换
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
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as official_forward

    batch_size, sequence_length = q.shape[:2]
    total_tokens = batch_size * sequence_length

    q_flat = q.reshape(total_tokens, *q.shape[2:]).contiguous()
    k_flat = k.reshape(total_tokens, *k.shape[2:]).contiguous()
    v_flat = v.reshape(total_tokens, *v.shape[2:]).contiguous()
    # 课程输入是 log(alpha)，FlashInfer 输入是线性空间的 alpha。
    alpha_flat = raw_g.exp().reshape(total_tokens, *raw_g.shape[2:]).contiguous()
    beta_flat = beta.reshape(total_tokens, *beta.shape[2:]).contiguous()
    cu_seqlens = (
        torch.arange(batch_size + 1, device=q.device, dtype=torch.int64)
        * sequence_length
    )

    # 课程 state 为 [K, V]，FlashInfer state 为 [V, K]。
    flashinfer_state = (
        initial_state.transpose(-2, -1).contiguous()
        if initial_state is not None
        else None
    )

    output, final_state = official_forward(
        q_flat,
        k_flat,
        v_flat,
        alpha_flat,
        beta_flat,
        scale=SCALE,
        initial_state=flashinfer_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        use_cp="auto",
    )

    # 恢复接口的 batch 维和 state 布局
    output = output.reshape(batch_size, sequence_length, *output.shape[1:])
    final_state = final_state.transpose(-2, -1).contiguous()
    return output, final_state
