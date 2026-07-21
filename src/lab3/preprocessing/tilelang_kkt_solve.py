# 构造 gated KKT 矩阵，并求 forward 会收到的 A
import torch
import tilelang
import tilelang.language as T


CHUNK_SIZE = 64
HEAD_DIM_K = 128
LOG2E = 1.4426950408889634


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_kkt_solve(
    H,
    Hg,
    qk_dtype,
    gate_dtype,
    accum_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    k_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)

    @T.macro
    def solve_one_chunk(bb, chunk_idx, bh, bhg, k, g, beta, a):
        left = chunk_idx * CHUNK_SIZE
        right = left + CHUNK_SIZE

        # 一个 chunk 可以看成一个 64×64 小矩阵问题
        # K/g/beta 先放进 shared memory，矩阵乘加则在 FP32 fragment 中完成
        k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
        g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
        beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
        a64_fragment = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)

        a16i_row = T.alloc_fragment((4, 16), dtype=accum_dtype)
        a16i_sum = T.alloc_fragment((4, 16), dtype=accum_dtype)
        a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
        a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)

        a32i_fragment = T.alloc_fragment((2, 32, 32), dtype=accum_dtype)
        a32i0_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)
        a64_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)

        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )

        # 尾块将无效 token 补零，使后续矩阵运算仍保持固定 64×64 形状
        if right <= num_tokens:
            T.async_copy(k[bb, left:right, bhg, 0:HEAD_DIM_K], k_shared)
        else:
            for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                if left + token < num_tokens:
                    k_shared[token, dim] = k[bb, left + token, bhg, dim]
                else:
                    k_shared[token, dim] = 0

        for token in T.Parallel(CHUNK_SIZE):
            if left + token < num_tokens:
                g_shared[token] = g[bb, left + token, bh]
                beta_shared[token] = beta[bb, left + token, bh]
            else:
                g_shared[token] = 0
                beta_shared[token] = 0

        if right <= num_tokens:
            T.ptx_wait_group(0)
        # 先用一次 GEMM 得到 chunk 内所有 token 两两之间的 <k_i, k_j>
        T.gemm(
            k_shared,
            k_shared,
            a64_fragment,
            transpose_B=True,
            clear_accum=True,
        )

        # M[i,j] = beta_i <k_i,k_j> exp(g_i-g_j)，只保留下三角；对角线为 1
        for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
            if row > col:
                a64_fragment[row, col] *= beta_shared[row] * T.exp2(
                    (g_shared[row] - g_shared[col]) * LOG2E
                )
            elif row == col:
                a64_fragment[row, col] = 1
            else:
                a64_fragment[row, col] = 0

        # 利用块下三角结构，依次完成 16×16、32×32、64×64 的求逆
        for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
            if row >= 32 and col < 32:
                a32o_shared[row - 32, col] = -a64_fragment[row, col]
            elif (row // 16) == (col // 16) + 1:
                a16o_shared[row // 32, row % 16, col % 16] = -a64_fragment[row, col]
            elif (row // 16) == (col // 16):
                a16i_shared[row // 16, row % 16, col % 16] = a64_fragment[row, col]

        T.clear(a16i_row)
        for pivot in T.unroll(1, 16):
            for block, col in T.Parallel(4, 16):
                if col < pivot:
                    a16i_row[block, col] = a16i_shared[block, pivot, col]
            T.clear(a16i_sum)
            for previous in T.unroll(pivot):
                for block, col in T.Parallel(4, 16):
                    a16i_sum[block, col] -= (
                        a16i_shared[block, previous, col] * a16i_row[block, previous]
                    )
            for block, col in T.Parallel(4, 16):
                if col < pivot:
                    a16i_shared[block, pivot, col] = a16i_sum[block, col]

        T.clear(a16o_fragment)
        for reduction in T.unroll(16):
            for block, row, col in T.Parallel(2, 16, 16):
                a16o_fragment[block, row, col] += (
                    a16i_shared[block * 2 + 1, row, reduction]
                    * a16o_shared[block, reduction, col]
                )
        for block, row, col in T.Parallel(2, 16, 16):
            a16o_shared[block, col, row] = a16o_fragment[block, row, col]
        T.clear(a16o_fragment)
        for reduction in T.unroll(16):
            for block, row, col in T.Parallel(2, 16, 16):
                a16o_fragment[block, row, col] += (
                    a16o_shared[block, reduction, row]
                    * a16i_shared[block * 2, reduction, col]
                )
        T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

        for block, row, col in T.Parallel(2, 32, 32):
            if row < 16 and col >= 16:
                a32i_fragment[block, row, col] = 0
        for block, row, col in T.Parallel(2, 32, 32):
            if row >= 16 and col < 16:
                a32i_fragment[block, row, col] = a16o_shared[block, row - 16, col]
        for block, row, col in T.Parallel(2, 32, 32):
            if row // 16 == col // 16:
                a32i_fragment[block, row, col] = a16i_shared[
                    block * 2 + row // 16, row % 16, col % 16
                ]
        for block, row, col in T.Parallel(2, 32, 32):
            if block == 0:
                a32i0_shared[row, col] = a32i_fragment[block, row, col]
            else:
                a32i1_shared[row, col] = a32i_fragment[block, row, col]
        T.gemm(a32i1_shared, a32o_shared, a32o_fragment, clear_accum=True)
        T.copy(a32o_fragment, a32o_shared)
        T.gemm(a32o_shared, a32i0_shared, a32o_fragment, clear_accum=True)

        for block, row, col in T.Parallel(2, 32, 32):
            a64_shared[block * 32 + row, block * 32 + col] = a32i_fragment[
                block, row, col
            ]
        for row, col in T.Parallel(32, 32):
            a64_shared[32 + row, col] = a32o_fragment[row, col]
        for row, col in T.Parallel(32, 32):
            a64_shared[row, 32 + col] = 0

        if right <= num_tokens:
            T.copy(a64_shared, a[bb, left:right, bh, 0:CHUNK_SIZE])
        else:
            for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                if left + row < num_tokens:
                    a[bb, left + row, bh, col] = a64_shared[row, col]

    @T.prim_func
    def kernel(
        k: T.Tensor(k_shape, dtype=qk_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        beta: T.Tensor(gate_shape, dtype=gate_dtype),
        a: T.Tensor(a_shape, dtype=qk_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(num_chunks * H, threads=128) as (block,):
            chunk_idx, bh = block // H, block % H
            bb = chunk_idx % batch_size
            local_chunk = chunk_idx // batch_size
            # GVA 中多个 V head 共享同一个 Q/K head。
            bhg = bh // (H // Hg)
            solve_one_chunk(bb, local_chunk, bh, bhg, k, g, beta, a)

    return kernel


# 输入 K 为 [B, T, Hq, 128]，输出 A 为 [B, T, Hv, 64]。
def kkt_solve(
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_tokens, num_heads_qk, _ = k.shape
    num_heads_v = g_cumsum.shape[-1]
    num_chunks = batch_size * tilelang.cdiv(num_tokens, CHUNK_SIZE)
    A = torch.empty(
        (batch_size, num_tokens, num_heads_v, CHUNK_SIZE),
        dtype=k.dtype,
        device=k.device,
    )
    kernel = tilelang_kkt_solve(
        num_heads_v,
        num_heads_qk,
        qk_dtype=k.dtype,
        gate_dtype=g_cumsum.dtype,
        accum_dtype="float32",
    )
    kernel(k, g_cumsum, beta, A, num_chunks)
    return A
