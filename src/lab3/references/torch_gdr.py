# 默认用 FP64 给出标准答案
# 每个 chunk 的核心计算为：
#   W = A @ (beta * K * exp(g))
#   U = A @ (beta * V)
#   V_new = U - W @ S
# 随后由旧 state 和 chunk 内 Q/K/V_new 共同得到 output，再更新跨 chunk state。
import torch


CHUNK_SIZE = 64


def _expand_qk_heads(x: torch.Tensor, num_heads_v: int) -> torch.Tensor:
    num_heads_qk = x.shape[2]
    if num_heads_qk == num_heads_v:
        return x
    # GVA 中一个 Q/K head 连续对应多个 V head。
    return x.repeat_interleave(num_heads_v // num_heads_qk, dim=2)


def torch_chunk_local_cumsum(g: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(g)
    # 每个 chunk 都从零开始累计，不能把前一个 chunk 的和带进来。
    for start in range(0, g.shape[1], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, g.shape[1])
        output[:, start:end] = g[:, start:end].cumsum(dim=1)
    return output


def torch_kkt_solve(
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    batch_size, num_tokens = k.shape[:2]
    num_heads_v = g_cumsum.shape[-1]
    k = _expand_qk_heads(k, num_heads_v).to(dtype)
    g_cumsum = g_cumsum.to(dtype)
    beta = beta.to(dtype)
    output = torch.zeros(
        (batch_size, num_tokens, num_heads_v, CHUNK_SIZE),
        dtype=dtype,
        device=k.device,
    )

    for start in range(0, num_tokens, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, num_tokens)
        length = end - start
        kc = k[:, start:end].transpose(1, 2)
        gc = g_cumsum[:, start:end].transpose(1, 2)
        bc = beta[:, start:end].transpose(1, 2)

        # M 是单位下三角矩阵，A = M^{-1}。
        gram = torch.einsum("bhid,bhjd->bhij", kc, kc)
        decay = torch.exp(gc[..., :, None] - gc[..., None, :])
        lower = torch.tril(gram * bc[..., :, None] * decay, diagonal=-1)
        identity = torch.eye(length, dtype=dtype, device=k.device)
        matrix = lower + identity
        inverse = torch.linalg.solve_triangular(
            matrix,
            identity.expand(batch_size, num_heads_v, length, length),
            upper=False,
            unitriangular=True,
        )
        output[:, start:end, :, :length] = inverse.transpose(1, 2)
    return output


def torch_gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float64,
    student_interface: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    dtype = torch.float32 if student_interface else dtype
    batch_size, num_tokens, _, head_dim_k = q.shape
    _, _, num_heads_v, head_dim_v = v.shape
    q = _expand_qk_heads(q, num_heads_v).to(dtype)
    k = _expand_qk_heads(k, num_heads_v).to(dtype)
    v = v.to(dtype)
    g_cumsum = g_cumsum.to(dtype)
    beta = beta.to(dtype)
    A = A.to(dtype)
    scale = head_dim_k**-0.5

    # 没有传入 initial_state 时，从全零状态开始。
    if initial_state is None:
        state = torch.zeros(
            (batch_size, num_heads_v, head_dim_k, head_dim_v),
            dtype=dtype,
            device=q.device,
        )
    else:
        state = initial_state.to(dtype, copy=True)

    output = torch.empty(
        (batch_size, num_tokens, num_heads_v, head_dim_v),
        dtype=dtype,
        device=q.device,
    )
    # 按时间顺序传递 state；每个 chunk 内部使用矩阵运算并行计算。
    for start in range(0, num_tokens, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, num_tokens)
        length = end - start
        qc = q[:, start:end].transpose(1, 2)
        kc = k[:, start:end].transpose(1, 2)
        vc = v[:, start:end].transpose(1, 2)
        gc = g_cumsum[:, start:end].transpose(1, 2)
        bc = beta[:, start:end].transpose(1, 2)
        Ac = A[:, start:end, :, :length].transpose(1, 2)

        # 先求 W、U，再扣除旧 state 对当前 value update 的预测。
        w = torch.einsum(
            "bhij,bhjd->bhid",
            Ac,
            kc * bc[..., None] * torch.exp(gc)[..., None],
        )
        u = torch.einsum("bhij,bhjv->bhiv", Ac, vc * bc[..., None])
        v_new = u - torch.einsum("bhid,bhdv->bhiv", w, state)

        # output = 旧 state 贡献 + 当前 chunk 内的因果贡献。
        decay = torch.tril(torch.exp(gc[..., :, None] - gc[..., None, :]))
        scores = torch.einsum("bhid,bhjd->bhij", qc, kc)
        output_from_state = (
            scale
            * torch.exp(gc)[..., None]
            * torch.einsum("bhid,bhdv->bhiv", qc, state)
        )
        output_in_chunk = scale * torch.einsum("bhij,bhjv->bhiv", scores * decay, v_new)
        output[:, start:end] = (output_from_state + output_in_chunk).transpose(1, 2)

        # 将旧 state 衰减到 chunk 末尾，再累加当前 chunk 的更新。
        g_last = gc[..., -1]
        state = torch.exp(g_last)[..., None, None] * state + torch.einsum(
            "bhid,bhiv->bhdv",
            kc * torch.exp(g_last[..., None] - gc)[..., None],
            v_new,
        )
    if student_interface:
        return output.to(torch.bfloat16), state.to(torch.float32)
    return output, state


def ref_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 完整 reference 还包括 chunk-local cumsum 和 gated KKT/A solve。
    g_cumsum = torch_chunk_local_cumsum(g)
    A = torch_kkt_solve(k, g_cumsum, beta)
    return torch_gdn_prefill_forward(
        q,
        k,
        v,
        g_cumsum,
        beta,
        A,
        initial_state,
    )
