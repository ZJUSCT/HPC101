#ifndef WINOGRAD_H
#define WINOGRAD_H

void naive_conv(const float* restrict image, const float* restrict filter, float* restrict out,
                float* restrict U, float* restrict V, float* restrict M,
                const int inHeight, const int inWidth, 
                const int C, const int K, const int N);

void winograd_conv(const float* restrict image, const float* restrict filter, float* restrict out,
                   float* restrict U, float* restrict V, float* restrict M,
                   const int inHeight, const int inWidth, const int C, const int K, const int N);

#endif
