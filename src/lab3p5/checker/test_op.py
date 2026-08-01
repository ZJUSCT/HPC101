#!/usr/bin/env python3
# coding=utf-8
"""FusedAddRmsNorm — correctness and single-launch profiling entry.

Cases are generated on the fly from `case_specs.py` (random inputs by seed) —
NO .bin data ships in the kit. The judge swaps `case_specs.py` for a hidden one
with different shapes; this file itself never changes.

Flow (mirrors the S9 samples Concat/Transpose):
  1. import custom_ops_lib (the student's built extension / Triton / TileLang)
  2. for each case: generate inputs from seed, run the op on NPU, compare
     against the fp32-computed golden

Profiling mode launches the selected operator exactly once. ``msprof op`` owns
the warm-up/replay loop and writes the final Task Duration to OpBasicInfo.csv.

Usage (from kit root, after `source ./env.sh`):
  python3 checker/test_op.py            # run all cases
  python3 checker/test_op.py <case_num> # run a single case
  python3 checker/test_op.py --profile <case_num>  # one launch, no verification
"""
import argparse
import os
import sys
import torch
import torch_npu

# Pick the backend: Ascend C (default, via the pybind wheel), Triton
# (LANG=triton), or TileLang (LANG=tilelang). The latter two are pure Python and
# need no wheel build.
_lang = os.environ.get("LANG", "ascendc").lower()
if _lang in ("triton", "tilelang"):
    import importlib
    importlib.import_module(f"src.{_lang}")  # registers `custom_ops_lib` in sys.modules
    import custom_ops_lib
else:
    import custom_ops_lib

torch.npu.config.allow_internal_format = False
torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))


# ---------------------------------------------------------------------------
# Case specs (public; judge replaces this module with a hidden one)
# ---------------------------------------------------------------------------
import case_specs  # noqa: E402


DATA_RANGES = {"S": (-1.0, 1.0), "M": (1.0, 10.0), "L": (-1000.0, 1000.0)}


def gen_inputs(B, H, data_range, seed):
    """Deterministic random inputs from seed (reproducible across runs/judge)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    lo, hi = DATA_RANGES[data_range]
    x = (torch.rand(B, H, generator=g) * (hi - lo) + lo).to(torch.float16)
    r = (torch.rand(B, H, generator=g) * (hi - lo) + lo).to(torch.float16)
    w = (torch.rand(H, generator=g) * 2.0).clamp(min=0.01).to(torch.float16)
    return x, r, w


def golden(x, residual, weight, eps):
    """fp32 compute then cast fp16 (matches the lab3.5_frame golden)."""
    R = residual.float() + x.float()
    ms = torch.mean(R * R, dim=-1, keepdim=True)
    rms = torch.sqrt(ms + eps)
    Y = (R / rms) * weight.float()
    return Y.to(torch.float16), R.to(torch.float16)


# ---------------------------------------------------------------------------
# Verification (ported verbatim from the upstream checker verify_result.py)
# ---------------------------------------------------------------------------
def verify_result(real, golden_t):
    """Element passes if abs_err<=tol OR rel_err<=tol, where rel_err uses
    |golden| (clamped to eps) as the denominator — NOT max(|real|,|golden|).
    The whole tensor passes only if error_ratio <= 0.0 (zero mismatches),
    matching the upstream checker exactly.
    """
    if golden_t.dtype == torch.float32:
        tol = 1e-4
    else:
        tol = 1e-3
    # Match the upstream checker: rel_err = abs_err / |golden| (clamped to eps).
    out = real.detach().cpu().to(torch.float64).reshape(-1)
    g = golden_t.detach().cpu().to(torch.float64).reshape(-1)
    eps = 1e-12
    denom = torch.where(g.abs() < eps, torch.tensor(eps), g.abs())
    abs_err = (out - g).abs()
    rel_err = abs_err / denom
    pass_check = (abs_err <= tol) | (rel_err <= tol)
    error_ratio = float((~pass_check).sum().item()) / g.numel()
    # upstream: return error_ratio <= 0.0
    ok = error_ratio <= 0.0
    if not ok:
        # show up to 100 failing indices, like the upstream checker
        bad = torch.where(~pass_check)[0]
        for i, idx in enumerate(bad[:100]):
            gv = float(g[idx]); ov = float(out[idx])
            dv = abs(gv) if abs(gv) > eps else eps
            print(f"  idx={int(idx):06d} expected={gv:.9f} actual={ov:.9f} "
                  f"rdiff={abs(ov-gv)/dv:.6f}")
        print(f"  error_ratio={error_ratio:.6f} (tolerance: 0.0000)")
    else:
        print("test pass")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_case(idx, spec):
    B, H, eps, data_range, seed = spec
    name = f"case_{idx}"
    x, r, w = gen_inputs(B, H, data_range, seed)
    gy, gres = golden(x, r, w, eps)         # CPU golden

    x_d, r_d, w_d = x.npu(), r.npu(), w.npu()
    y, resout = custom_ops_lib.fused_add_rmsnorm(x_d, r_d, w_d, eps)
    if y is None:
        print(f"[{name}] execution returned None (timeout?)")
        return False
    y = y.cpu()
    resout = resout.cpu()

    ok_y = verify_result(y, gy)
    ok_r = verify_result(resout, gres)
    print(f"[{name}] {B}x{H} y={'pass' if ok_y else 'FAIL'} "
          f"residual_out={'pass' if ok_r else 'FAIL'}")
    return ok_y and ok_r


def profile_case(idx, spec):
    """Launch only the selected op; msprof op performs warm-up and replay."""
    B, H, eps, data_range, seed = spec
    x, r, w = gen_inputs(B, H, data_range, seed)
    x_d, r_d, w_d = x.npu(), r.npu(), w.npu()
    custom_ops_lib.fused_add_rmsnorm(x_d, r_d, w_d, eps)
    torch.npu.synchronize()
    print(f"[profile] case_{idx} {B}x{H} launched once")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_num", nargs="?", type=int)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="launch one case once for msprof op; skip golden and verification",
    )
    args = parser.parse_args()

    case_num = args.case_num
    cases = case_specs.CASES

    if case_num is not None:
        # 1-based case number, mirroring S9 test_op.py <num>
        i = case_num - 1
        if i < 0 or i >= len(cases):
            print(f"[ERROR] case_{case_num} not found (have {len(cases)} cases)")
            sys.exit(2)
        if args.profile:
            profile_case(i, cases[i])
            return
        ok = run_case(i, cases[i])
        sys.exit(0 if ok else 1)

    if args.profile:
        print("[ERROR] --profile requires a case number")
        sys.exit(2)

    all_ok = True
    for i, spec in enumerate(cases):
        all_ok &= run_case(i, spec)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
