#include <cuda.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include "common.cuh"
#include "generate.cuh"
#include "reference_cpu.cuh"
#include "reference_gpu.cuh"
#include "conv.cuh"

double getTFLOPS(double ms) {
    return FLOP / (ms * 1e-3) / 1e12;
}


void Check(const Element *a, const Element *w, Element *b) {
    auto ref_b = new Element[N * H * W * K];
    std::cout << "Conv2d Reference Kernel Start... \n";
    if constexpr (std::is_same_v<Element, int8_t>) {
        conv2d_cpu_ref<Element, ElementCompute>(a, w, ref_b);
    }
    if constexpr (std::is_same_v<Element, cutlass::half_t>) {
        conv2d_gpu_ref<Element, ElementCompute>(a, w, ref_b);
    } 
    std::cout << "Checking Results... \n";
    for(size_t i=0;i<size_t(N);i++)
    {
        for(size_t j=0;j<H;j++)
        {
            for(size_t k=0;k<W;k++)
            {
                for(size_t l=0;l<K;l++)
                {

                    int pos = i * H * W * K + j * W * K + k * K + l; 
                    if (b[pos] != ref_b[pos]) {
                        std::cout << "\x1b[31m"
                                    "Wrong Answer"
                                    "\x1b[0m"
                                    " at (n=" << i << ", output_h=" << j << ", output_w=" << k << ", k=" << l << ")" << std::endl;
                        std::cout<<"expected "<<ref_b[pos]<< " but find " << b[pos] << ", Keep it up!"<<std::endl;
                        delete[] ref_b;
                        return;
                    }
                }
            }
        }
    }
    std::cout << "\x1b[32m"
                 "Correct"
                 "\x1b[0m"
              << std::endl;

    delete[] ref_b;
}

void run_cuda(const Element *a, const Element *w, Element *b, cudaEvent_t *start_e, cudaEvent_t *stop_e) {
    Element *d_a, *d_w, *d_b;
    CUDA_CALL(cudaMalloc(&d_a, N * H * W * C * sizeof(Element)));
    CUDA_CALL(cudaMemcpy(d_a, a, N * H * W * C * sizeof(Element), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_w, K * R * S * C * sizeof(Element)));
    CUDA_CALL(cudaMemcpy(d_w, w, K * R * S * C * sizeof(Element), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_b, N * H * W * K * sizeof(Element)));
    // Start Timer.
    cudaEventRecord(*start_e);
    KernelConfig config = get_kernel_config<Element>();
    dim3 grid = config.grid;
    dim3 block = config.block;
    int shared_memory_size = config.shared_memory_size;
    auto kernel = conv2d_cuda_kernel<Element, ElementCompute>;
    if (shared_memory_size > 48 * 1024) {
        CUDA_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    }
    kernel<<<grid, block, shared_memory_size>>>(d_a, d_w, d_b);
    CUDA_CALL(cudaGetLastError());
    // Stop Timer
    cudaEventRecord(*stop_e);
    cudaEventSynchronize(*stop_e);
    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(b, d_b, N * H * W * K * sizeof(Element), cudaMemcpyDeviceToHost));
    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_b);
}

int main() {
    auto a = new Element[size_t(N) * H * W * C];
    auto w = new Element[size_t(K) * R * S * C];
    auto b = new Element[size_t(N) * H * W * K];

    std::cout << "Generating input and kernel tensor... \n";
    GenerateInput(a, w);
    memset(b, 0, size_t(N) * H * W * K * sizeof(Element));


    std::cout << "Tensor generated.\n";

    cudaEvent_t start_e, stop_e;
    cudaEventCreate(&start_e);
    cudaEventCreate(&stop_e);

    // Conv(a, w, b);
    std::cout << "Your Conv2d Cuda Kernel Start... \n";
    run_cuda(a, w, b, &start_e, &stop_e);

    std::cout << "Verifying... \n";
    Check(a, w, b);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_e, stop_e);
    std::cout << "Your Conv2d: \n" << milliseconds << " ms\n";
    std::cout << std::fixed << std::setprecision(4) << getTFLOPS(milliseconds) << " TFLOPS\n";
    cudaEventDestroy(start_e);
    cudaEventDestroy(stop_e);

    // Output(a, w, b);
    delete[] a;
    delete[] w;
    delete[] b;
    return 0;
}
