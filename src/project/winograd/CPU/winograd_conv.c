#include "winograd.h"

// Transformation matrices for Winograd F(2x2, 3x3)
const float G[4][3] = {
    {1.0, 0.0, 0.0}, 
    {0.5, 0.5, 0.5}, 
    {0.5, -0.5, 0.5}, 
    {0.0, 0.0, 1.0}
};
const float G_T[3][4] = {
    {1, 0.5, 0.5, 0.0}, 
    {0.0, 0.5, -0.5, 0.0}, 
    {0.0, 0.5, 0.5, 1.0}
};
const float B[4][4] = {
    {1, 0, 0, 0}, 
    {0, 1, -1, 1}, 
    {-1, 1, 1, 0}, 
    {0, 0, 0, -1}
};
const float B_T[4][4] = {
    {1, 0, -1, 0}, 
    {0, 1, 1, 0}, 
    {0, -1, 1, 0}, 
    {0, 1, 0, -1}
};
const float A[4][2] = {
    {1, 0}, 
    {1, 1}, 
    {1, -1}, 
    {0, -1}
};
const float A_T[2][4] = {
    {1, 1, 1, 0}, 
    {0, 1, -1, -1}
};

void sgemm(const float* A, const float* B, float* out, 
           const int M, const int K, const int N) {
    for (int i = 0; i < M * N; ++i) {
        out[i] = 0.0f;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                out[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void sgemm_parallel(const float* A, const float* B, float* out,
                    const int M, const int K, const int N) {
    for (int i = 0; i < M * N; ++i) {
        out[i] = 0.0f;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                out[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

/** 
 * @brief Winograd Implementation of F(2x2, 3x3)
 * @param image: [batch * C * inHeight * inWidth]
 * @param filter: [K * C * 3 * 3]
 * @param out: Output result. Shape is [batch * K * outHeight * outWidth]
 * @param U: [4 * 4 * K * C], intermediate transformed filters
 * @param V: [4 * 4 * C * P], intermediate transformed image
 * @param M: [4 * 4 * K * P], intermediate result
 * @param inHeight: Height of input image
 * @param inWidth: Width of input image
 * @param C: Number of channels in input image
 * @param K: Number of filters
 * @param N: Batch size
 */
void winograd_conv(const float* restrict image, const float* restrict filter, float* restrict out,
                   float* restrict U, float* restrict V, float* restrict M,
                   const int inHeight, const int inWidth, const int C, const int K, const int N) {
    const int outHeight = inHeight - 2; // output height
    const int outWidth = inWidth - 2; // output width
    const int sizeI = inHeight * inWidth; // size of input image
    const int sizeF = 3 * 3; // size of filter
    const int sizeO = outHeight * outWidth; // size of output
    const int P = outHeight / 2 * outWidth / 2 * N; // size of output in blocks

    float tmp_u[12]; // 4 * 3
    float u[16];     // 4 * 4;
    
    // Transform filters and scatter to U
    // U[:, :, k, c] = G * filters[k, c, :, :] * G^T
    #pragma omp parallel for private(tmp_u, u)
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            float* filters_ptr = filter + (k * C + c) * sizeF;
            sgemm(&G[0][0], filters_ptr, tmp_u, 4, 3, 3);
            sgemm(tmp_u, &G_T[0][0], u, 4, 3, 4);
            
            for (int xi = 0; xi < 4; ++xi) {
                for (int nu = 0; nu < 4; ++nu) {
                    U[((xi * 4 + nu) * K + k) * C + c] = u[xi * 4 + nu];
                }
            }
        }
    }

    // Transform image and scatter to V
    // V[:, :, c, p] = B^T * image[c, b, :, :] * B
    float tmp_v[16];
    float d[16]; // d: [4 * 4];
    float v[16]; // v: [4 * 4];
    #pragma omp parallel for collapse(2) private(tmp_v, d, v)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < outHeight / 2; ++y) {
                for (int x = 0; x < outWidth / 2; ++x) {
                    // Generate d_cb
                    for (int iy = 0; iy < 4; ++iy) {
                        for (int ix = 0; ix < 4; ++ix) {
                            d[iy * 4 + ix] = image[(n * C + c) * sizeI +
                                                (y * 2 + iy) * inWidth + (x * 2 + ix)];
                        }
                    }
                    sgemm(&B_T[0][0], d, tmp_v, 4, 4, 4);
                    sgemm(tmp_v, &B[0][0], v, 4, 4, 4);
                    
                    int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
                    for (int xi = 0; xi < 4; ++xi) {
                        for (int nu = 0; nu < 4; ++nu) {
                            V[((xi * 4 + nu) * C + c) * P + b] = v[xi * 4 + nu];
                        }
                    }
                }
            }    
        }
    }

    // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
    for (int xi = 0; xi < 4; ++xi) {
        for (int nu = 0; nu < 4; ++nu) {
            float* M_ptr = M + (xi * 4 + nu) * K * P;
            float* U_ptr = U + (xi * 4 + nu) * K * C;
            float* V_ptr = V + (xi * 4 + nu) * C * P;
            sgemm_parallel(U_ptr, V_ptr, M_ptr, K, C, P);
        }
    }

    // Gather output and apply inverse transformation
    // Y = A^T * m * A
    float mm[16];      // 4 * 4
    float tmp_m[8];    // 2 * 4
    float temp_out[4]; // 2 * 2
    #pragma omp parallel for collapse(2) private(mm, temp_out, tmp_m)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int y = 0; y < outHeight / 2; ++y) {
                for (int x = 0; x < outWidth / 2; ++x) {
                    int b = (n * outHeight / 2 + y) * outWidth / 2 + x;
                    for (int xi = 0; xi < 4; ++xi) {
                        for (int nu = 0; nu < 4; ++nu) {
                            mm[xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
                        }
                    }
                    sgemm(&A_T[0][0], mm, tmp_m, 2, 4, 4);
                    sgemm(tmp_m, &A[0][0], temp_out, 2, 4, 2);
                    
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            out[((n * K + k) * outHeight + y * 2 + i) * outWidth +
                                x * 2 + j] = temp_out[i * 2 + j];
                        }
                    }
                }
            }
        }
    }
}