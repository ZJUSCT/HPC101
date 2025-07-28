#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

void naive_conv(thrust::device_vector<float>& image,
                thrust::device_vector<float>& filter, 
                thrust::device_vector<float>& out,
                thrust::device_vector<float>& U,
                thrust::device_vector<float>& V, 
                thrust::device_vector<float>& M,
                int H, int W, int C, int K, int N);

void winograd_conv(thrust::device_vector<float>& image, 
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N);