import numpy as np

from bilinear_interp.baseline import bilinear_interp_baseline
from bilinear_interp.vectorized import bilinear_interp_vectorized

from utils.timer import time_function

N = 8
H1 = 540
W1 = 960
C = 4
scale = 256
H2 = 720
W2 = 1280


if __name__ == '__main__':
    # Generate random data
    print('Generating Data...')
    a = np.random.randint(scale, size=(N, H1, W1, C))
    b = np.random.rand(N, H2, W2, 2) * [H1 - 1, W1 - 1]
    assert np.max(b[:, :, :, 0]) < H1
    assert np.max(b[:, :, :, 1]) < W1

    # Call Bilinear Interpolation Implementations
    print('Executing Baseline Implementation...')
    baseline_result, baseline_time = time_function(bilinear_interp_baseline, a, b)
    print(f'Finished in {baseline_time}s')

    print('Executing Vectorized Implementation...')
    vectorized_result, vectorized_time = time_function(bilinear_interp_vectorized, a, b)
    print(f'Finished in {vectorized_time}s')

    # Check Results
    if not np.array_equal(baseline_result, vectorized_result):
        raise Exception('Results are different!')
    else:
        print("[PASSED] Results are identical.")
    print(f"Speed Up {baseline_time / vectorized_time}x")
