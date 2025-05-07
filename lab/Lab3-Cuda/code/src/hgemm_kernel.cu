#include "gemm.cuh"
#include "cute/tensor.hpp"

__global__ void hgemm_kernel(const cute::half_t* __restrict__ A,
							 const cute::half_t* __restrict__ B, cute::half_t* __restrict__ C,
							 const int m, const int n, const int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n) {
		cute::half_t sum(0.0f);
		for (int i = 0; i < k; i++) {
			sum += A[row * k + i] * B[i + col * k];
		}
		C[row + col * m] = sum;
	}
}


template <>
void run_custom<cute::half_t>(thrust::device_vector<cute::half_t>& d_A,
							  thrust::device_vector<cute::half_t>& d_B,
							  thrust::device_vector<cute::half_t>& d_C,
							  const int m, const int n,
							  const int k) {
	dim3 block(32, 32);
	dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
	hgemm_kernel << <grid, block >> > (d_A.data().get(),
								  d_B.data().get(),
								  d_C.data().get(), m, n, k);
}