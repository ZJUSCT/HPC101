#include "winograd.cuh" 

__global__ void naive_conv_kernel(float* __restrict__ image, 
                                  float* __restrict__ filter,
                                  float* __restrict__ out,
                                  int N, int C, int H, int W, int K) {
    const int outH = H - 2;
    const int outW = W - 2;
    const int R = 3, S = 3;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = N * K * outH * outW;
    if (idx >= out_size) return;

    int w_out = idx % outW;
    int h_out = (idx / outW) % outH;
    int k_out = (idx / (outW * outH)) % K;
    int n_out = idx / (K * outW * outH);

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = h_out + r;
                int w_in = w_out + s;
                
                int image_idx = (n_out * C + c) * H * W + h_in * W + w_in;
                int filter_idx = (k_out * C + c) * R * S + r * S + s;
                
                acc += image[image_idx] * filter[filter_idx];
            }
        }
    }
    out[idx] = acc;
}

void naive_conv(thrust::device_vector<float>& image, 
                thrust::device_vector<float>& filter, 
                thrust::device_vector<float>& out,
                thrust::device_vector<float>& U,
                thrust::device_vector<float>& V, 
                thrust::device_vector<float>& M,
                int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    int out_size = N * K * outH * outW;

    const int threads_per_block = 256;
    int grid_size = (out_size + threads_per_block - 1) / threads_per_block;

    naive_conv_kernel<<<grid_size, threads_per_block>>>(
        image.data().get(), filter.data().get(), out.data().get(),
        N, C, H, W, K
    );
    
    cudaDeviceSynchronize();
}
