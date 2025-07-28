#include "winograd.cuh"

// Transformation matrices for F(2x2, 3x3)
__constant__ float G[4][3] = {
    {1.0f, 0.0f, 0.0f}, 
    {0.5f, 0.5f, 0.5f}, 
    {0.5f, -0.5f, 0.5f}, 
    {0.0f, 0.0f, 1.0f}
};

__constant__ float B_T[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f}, 
    {0.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, -1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, 0.0f, -1.0f}
};

__constant__ float B[4][4] = {
    {1.0f,  0.0f,  0.0f,  0.0f}, 
    {0.0f,  1.0f, -1.0f,  1.0f}, 
    {-1.0f, 1.0f,  1.0f,  0.0f}, 
    {0.0f,  0.0f,  0.0f, -1.0f}
};

__constant__ float A_T[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, -1.0f, -1.0f}
};

// Fused kernel for Winograd convolution F(2x2, 3x3)
__global__
void winograd_conv_kernel(const float* __restrict__ image,
                          const float* __restrict__ filter,
                          float* __restrict__ output,
                          int N, int C, int H, int W, int K, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    if (idx >= num_tiles) return;

    // Decompose thread index to get (n, k, tile_y, tile_x)
    int p_local = idx % ((outH / 2) * (outW / 2));
    int k = (idx / ((outH / 2) * (outW / 2))) % K;
    int n = idx / (K * (outH / 2) * (outW / 2));
    int tile_y = p_local / (outW / 2);
    int tile_x = p_local % (outW / 2);

    float m[4][4] = {{0.0f}};

    // Loop over input channels
    for (int c = 0; c < C; ++c) {
        // --- Filter Transform ---
        const float* g = filter + (k * C + c) * 9;
        float u_kc[4][4];
        float temp_g[4][3];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                temp_g[i][j] = G[i][0] * g[0 * 3 + j] + G[i][1] * g[1 * 3 + j] + G[i][2] * g[2 * 3 + j];
            }
        }
        for (int i = 0; i < 4; ++i) {
            u_kc[i][0] = temp_g[i][0];
            u_kc[i][1] = 0.5f * (temp_g[i][0] + temp_g[i][1] + temp_g[i][2]);
            u_kc[i][2] = 0.5f * (temp_g[i][0] - temp_g[i][1] + temp_g[i][2]);
            u_kc[i][3] = temp_g[i][2];
        }

        // --- Image Transform ---
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        float d[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                d[i][j] = image[(n * C + c) * H * W + (h_start + i) * W + (w_start + j)];
            }
        }
        float v_ncp[4][4];
        float temp_d[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                temp_d[i][j] = B_T[i][0] * d[0][j] + B_T[i][1] * d[1][j] + B_T[i][2] * d[2][j] + B_T[i][3] * d[3][j];
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v_ncp[i][j] = temp_d[i][0] * B[0][j] + temp_d[i][1] * B[1][j] + temp_d[i][2] * B[2][j] + temp_d[i][3] * B[3][j];
            }
        }

        // --- Element-wise product and accumulate ---
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[i][j] += u_kc[i][j] * v_ncp[i][j];
            }
        }
    }

    // --- Output Transform ---
    float temp_m[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            temp_m[i][j] = A_T[i][0] * m[0][j] + A_T[i][1] * m[1][j] + A_T[i][2] * m[2][j] + A_T[i][3] * m[3][j];
        }
    }
    float Y[2][2];
    for (int i = 0; i < 2; ++i) {
        Y[i][0] = temp_m[i][0] + temp_m[i][1] + temp_m[i][2];
        Y[i][1] = temp_m[i][1] - temp_m[i][2] - temp_m[i][3];
    }

    // --- Write output ---
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int h = tile_y * 2 + i;
            int w = tile_x * 2 + j;
            if (h < outH && w < outW) {
                output[((n * K + k) * outH + h) * outW + w] = Y[i][j];
            }
        }
    }
}

void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    
    const int threads_per_block = 256;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    int grid_size = (num_tiles + threads_per_block - 1) / threads_per_block;

    winograd_conv_kernel<<<grid_size, threads_per_block>>>(
        image.data().get(), filter.data().get(), out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}