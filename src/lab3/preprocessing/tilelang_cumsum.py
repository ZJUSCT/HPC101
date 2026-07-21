# raw_g 进入forward 前，会先在每个 64-token chunk 内独立计算前缀和。
import torch
import tilelang
import tilelang.language as T


CHUNK_SIZE = 64


@tilelang.jit
def tilelang_chunk_local_cumsum(H, accum_dtype, g_dtype):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    g_shape = (batch_size, num_tokens, H)

    @T.prim_func
    def kernel(
        g_raw: T.Tensor(g_shape, dtype=g_dtype),
        g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        total_chunks: T.int32,
    ):
        with T.Kernel(total_chunks, threads=128) as (bc,):
            bb = bc % batch_size
            chunk_idx = bc // batch_size
            left = chunk_idx * CHUNK_SIZE
            right = left + CHUNK_SIZE

            # T.cumsum 沿 fragment 的最后一维工作，先将 [token, head] 转置成 [head, token]。
            g_fragment = T.alloc_fragment((H, CHUNK_SIZE), dtype=accum_dtype)
            g_transposed = T.alloc_fragment((CHUNK_SIZE, H), dtype=g_dtype)
            g_shared = T.alloc_shared((CHUNK_SIZE, H + 1), dtype=g_dtype)

            # 尾块不足 64 个 token 时补零，写回时只保留有效位置。
            if right <= num_tokens:
                T.copy(g_raw[bb, left:right, 0:H], g_transposed)
            else:
                for token, head in T.Parallel(CHUNK_SIZE, H):
                    if left + token < num_tokens:
                        g_transposed[token, head] = g_raw[bb, left + token, head]
                    else:
                        g_transposed[token, head] = 0
            T.copy(g_transposed, g_shared[:, :H])

            for head, token in T.Parallel(H, CHUNK_SIZE):
                g_fragment[head, token] = g_shared[token, head]
            T.cumsum(g_fragment, dim=1)
            for head, token in T.Parallel(H, CHUNK_SIZE):
                g_shared[token, head] = g_fragment[head, token]
            T.copy(g_shared[:, :H], g_transposed)

            if right <= num_tokens:
                T.copy(g_transposed, g_cumsum[bb, left:right, 0:H])
            else:
                for token, head in T.Parallel(CHUNK_SIZE, H):
                    if left + token < num_tokens:
                        g_cumsum[bb, left + token, head] = g_transposed[token, head]

    return kernel


# 输入输出均为 [B, T, Hv] FP32。
def chunk_local_cumsum(g: torch.Tensor) -> torch.Tensor:
    batch_size, num_tokens, num_heads = g.shape
    total_chunks = batch_size * tilelang.cdiv(num_tokens, CHUNK_SIZE)
    output = torch.empty_like(g)
    kernel = tilelang_chunk_local_cumsum(
        num_heads,
        accum_dtype="float32",
        g_dtype=g.dtype,
    )
    kernel(g, output, total_chunks)
    return output
