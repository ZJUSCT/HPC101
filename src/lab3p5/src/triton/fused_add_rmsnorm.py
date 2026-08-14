"""FusedAddRmsNorm — Triton-Ascend kernel skeleton.

Students: implement `_fused_add_rmsnorm_kernel` and the launch in
`fused_add_rmsnorm`. The grid is pinned to the physical vector-core count by
convention; you may adjust it.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr, residual_ptr, weight_ptr, y_ptr, resout_ptr,
    B, H,
    eps,
    BLOCK_H: tl.constexpr,
):
    # TODO: implement kernel
    pid = tl.program_id(0)
    pass


def fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    """x,residual: (B,H) fp16 NPU; weight: (H,) fp16. Returns (y, residual_out)."""
    # TODO: launch kernel
    raise NotImplementedError("fused_add_rmsnorm not implemented")


# Expose the same module name the test harness imports.
def custom_op(x, residual, weight, eps):
    return fused_add_rmsnorm(x, residual, weight, eps)
