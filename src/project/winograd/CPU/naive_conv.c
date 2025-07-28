#include "winograd.h"

/**
 * @brief Naive convolution implementation.
 * @param image Input image data
 * @param filter Filter weights
 * @param out Output data
 * @param U intermediate matrix U
 * @param V intermediate matrix V for Winograd transformation
 * @param M intermediate matrix M for Winograd transformation
 * @param inHeight Height of the input image
 * @param inWidth Width of the input image
 * @param C Number of input channels
 * @param K Number of filters
 * @param N Batch size
 */
void naive_conv(const float* restrict image, const float* restrict filter, float* restrict out,
                float* restrict U, float* restrict V, float* restrict M,
                const int inHeight, const int inWidth, 
                const int C, const int K, const int N) {
    const int r = 3; // filter size
    const int outHeight = inHeight - r + 1;
    const int outWidth = inWidth - r + 1;
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int oh = 0; oh < outHeight; ++oh) {
                for (int ow = 0; ow < outWidth; ++ow) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int fh = 0; fh < r; ++fh) {
                            for (int fw = 0; fw < r; ++fw) {
                                int ih = oh + fh;
                                int iw = ow + fw;
                                
                                float image_val = image[(n * C + c) * inHeight * inWidth + ih * inWidth + iw];
                                float filter_val = filter[(k * C + c) * r * r + fh * r + fw];
                                sum += image_val * filter_val;
                            }
                        }
                    }
                    out[(n * K + k) * outHeight * outWidth + oh * outWidth + ow] = sum;
                }
            }
        }
    }
}