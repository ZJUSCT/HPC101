#include "gemm.cuh"

__global__ void sgemm_kernel(const float* __restrict__ A,
							 const float* __restrict__ B, float* __restrict__ C,
							 const int m, const int n, const int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n) {
		float sum = 0.0f;
		for (int i = 0; i < k; i++) {
			sum += A[row * k + i] * B[i + col * k];
		}
		C[row + col * m] = sum;
	}
}

template <>
void run_custom<float>(thrust::device_vector<float>& d_A,
					   thrust::device_vector<float>& d_B,
					   thrust::device_vector<float>& d_C, const int m,
					   const int n, const int k) {
	
	dim3 block(32, 32);
	dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
	sgemm_kernel<<<grid, block>>>(d_A.data().get(),
								  d_B.data().get(),
								  d_C.data().get(), m, n, k);
}