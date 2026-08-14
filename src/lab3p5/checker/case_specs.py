"""Public case specifications for FusedAddRmsNorm.

Each case is just a spec — inputs are generated on the fly from `seed` (so the
kit ships NO .bin data), and the golden is computed in-process (fp32, then cast
fp16) by test_op.py. The judge swaps this file for a hidden case_specs.py with
different shapes; test_op.py itself never changes.

Tuple layout: (B, H, eps, data_range, seed)
  data_range in {"S": (-1,1), "M": (1,10), "L": (-1000,1000)}
"""

CASES = [
    (32,   4096, 1e-6, "S", 1001),   # small, aligned
    (256,  1024, 1e-6, "S", 1002),   # regular, aligned
    (1,    4096, 1e-6, "S", 1003),   # single row — scalar-bound
    (1997, 3037, 1e-6, "S", 1004),   # misaligned B + misaligned H (tail handling)
    (2048, 4096, 1e-6, "S", 1005),   # large, aligned
]
