"""TileLang-Ascend FusedAddRmsNorm kernel — skeleton.

Students: implement `_build_kernel` (the `@T.prim_func` body) and the
`fused_add_rmsnorm` launcher. The tilelang-ascend backend must be installed
(separate `tilelang-ascend` package) — otherwise `BackendUnavailable` is
raised and the harness skips this backend.
"""
from __future__ import annotations


class BackendUnavailable(RuntimeError):
    """Raised when the TileLang Ascend backend is not installed."""


_backend_ok: bool | None = None
_kernel_cache: dict = {}

try:
    import tilelang
    import tilelang.language as T
except Exception:  # pragma: no cover
    tilelang = None
    T = None


def _ascend_target_available() -> bool:
    """True iff the ascend target detector is registered (tilelang-ascend installed)."""
    if tilelang is None:
        return False
    try:
        from tilelang.backend.target import list_target_detectors
        return any("ascend" in str(d).lower() for d in list_target_detectors())
    except Exception:
        return False


def _check_backend() -> bool:
    global _backend_ok
    if _backend_ok is not None:
        return _backend_ok
    _backend_ok = _ascend_target_available()
    return _backend_ok


def fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    """x,residual: (B,H) fp16; weight: (H,) fp16. Returns (y, residual_out)."""
    # TODO: implement
    raise NotImplementedError("fused_add_rmsnorm not implemented")


KERNEL_NAME_HINT = "addrmsnorm"
