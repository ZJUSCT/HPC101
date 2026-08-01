"""Triton-Ascend entry. When USE_TRITON=1, test_op.py imports this so that
`custom_ops_lib.fused_add_rmsnorm(...)` resolves to the Triton kernel below —
no wheel build needed.
"""
import sys

from .fused_add_rmsnorm import fused_add_rmsnorm


class _Lib:
    def fused_add_rmsnorm(self, x, residual, weight, eps):
        return fused_add_rmsnorm(x, residual, weight, eps)


sys.modules["custom_ops_lib"] = _Lib()
