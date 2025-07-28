#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "winograd.cuh"

#define LOOP_NUM 3
#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

class Config {
public:
    std::vector<int> C, H, W, K, Batch;
    int layer_num;

    Config(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + filename);
        }
        
        file >> layer_num;
        C.resize(layer_num);
        H.resize(layer_num);
        W.resize(layer_num);
        K.resize(layer_num);
        Batch.resize(layer_num);
        
        for (int i = 0; i < layer_num; i++) {
            file >> C[i] >> H[i] >> W[i] >> K[i] >> Batch[i];
        }
    }
};

class LayerResult {
public:
    double baseline_time = 0.0;
    double winograd_time = 0.0;
    double baseline_gflops = 0.0;
    double winograd_gflops = 0.0;
    double speedup = 0.0;
    bool passed = false;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    auto cfg = std::make_unique<Config>(argv[1]);
    std::vector<LayerResult> results(cfg->layer_num);
    
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float milliseconds = 0;

    thrust::host_vector<thrust::host_vector<float>> h_images(cfg->layer_num);
    thrust::host_vector<thrust::host_vector<float>> h_filters(cfg->layer_num);

    std::random_device rd;
    thrust::default_random_engine rng(rd());
    thrust::uniform_real_distribution<float> dist(0.0f, 10.0f);

    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeI = H * W, sizeF = 9;
        h_images[l].resize(N * C * sizeI);
        h_filters[l].resize(K * C * sizeF);
        thrust::generate(h_images[l].begin(), h_images[l].end(), [&] { return dist(rng); });
        thrust::generate(h_filters[l].begin(), h_filters[l].end(), [&] { return dist(rng); });
    }

    thrust::host_vector<thrust::host_vector<float>> h_baseline_outputs(cfg->layer_num);
    thrust::host_vector<thrust::host_vector<float>> h_custom_outputs(cfg->layer_num);

    std::cout << "=== Running Naive Convolution ===" << std::endl;
    double baseline_total_time = 0;
    long total_flops = 0;

    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeI = H * W, sizeF = 9, sizeO = (H-2) * (W-2);
        long P = N * (H-2)/2 * (W-2)/2;

        thrust::host_vector<float>& h_image = h_images[l];
        thrust::host_vector<float>& h_filter = h_filters[l];

        thrust::device_vector<float> d_image = h_image;
        thrust::device_vector<float> d_filter = h_filter;
        thrust::device_vector<float> d_out(N * K * sizeO);
        thrust::device_vector<float> d_U(16 * K * C);
        thrust::device_vector<float> d_V(16 * C * P);
        thrust::device_vector<float> d_M(16 * K * P);

        // Warmup
        naive_conv(d_image, d_filter, d_out, d_U, d_V, d_M, H, W, C, K, N);

        cudaEventRecord(start_event);
        for (int i = 0; i < LOOP_NUM; i++) {
            naive_conv(d_image, d_filter, d_out, d_U, d_V, d_M, H, W, C, K, N);
        }
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        results[l].baseline_time = static_cast<double>(milliseconds / 1000.0f) / LOOP_NUM;

        h_baseline_outputs[l] = d_out;

        long flops = static_cast<long>(N) * K * C * (H-2) * (W-2) * 18;
        results[l].baseline_gflops = flops * 1e-9 / results[l].baseline_time;
        
        std::cout << "Layer " << std::setw(2) << l << ": " << std::fixed << std::setprecision(2)
                  << results[l].baseline_time * 1000 << " ms (" << results[l].baseline_gflops << " GFLOPS)" << std::endl;
        
        baseline_total_time += results[l].baseline_time;
        total_flops += flops;
    }
    std::cout << "Baseline Total: " << std::fixed << std::setprecision(2) << baseline_total_time * 1000 
              << " ms (" << total_flops * 1e-9 / baseline_total_time << " GFLOPS)" << std::endl;

    std::cout << "\n=== Running Winograd Convolution ===" << std::endl;
    double winograd_total_time = 0;
    
    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeI = H * W, sizeF = 9, sizeO = (H-2) * (W-2);
        long P = N * (H-2)/2 * (W-2)/2;
        
        thrust::host_vector<float>& h_image = h_images[l];
        thrust::host_vector<float>& h_filter = h_filters[l];

        thrust::device_vector<float> d_image = h_image;
        thrust::device_vector<float> d_filter = h_filter;
        thrust::device_vector<float> d_out(N * K * sizeO);
        thrust::device_vector<float> d_U(16 * K * C);
        thrust::device_vector<float> d_V(16 * C * P);
        thrust::device_vector<float> d_M(16 * K * P);

        // Warmup
        winograd_conv(d_image, d_filter, d_out, d_U, d_V, d_M, H, W, C, K, N);
        
        cudaEventRecord(start_event);
        for (int i = 0; i < LOOP_NUM; i++) {
            winograd_conv(d_image, d_filter, d_out, d_U, d_V, d_M, H, W, C, K, N);
        }
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);

        results[l].winograd_time = static_cast<double>(milliseconds / 1000.0f) / LOOP_NUM;
        
        h_custom_outputs[l] = d_out;
        
        long flops = static_cast<long>(N) * K * C * (H-2) * (W-2) * 18;
        results[l].winograd_gflops = flops * 1e-9 / results[l].winograd_time;
        results[l].speedup = results[l].baseline_time / results[l].winograd_time;
        
        std::cout << "Layer " << std::setw(2) << l << ": " << std::fixed << std::setprecision(2)
                  << results[l].winograd_time * 1000 << " ms (" << results[l].winograd_gflops << " GFLOPS)" << std::endl;
        
        winograd_total_time += results[l].winograd_time;
    }
    
    std::cout << "Custom Total: " << std::fixed << std::setprecision(2) << winograd_total_time * 1000 
              << " ms (" << total_flops * 1e-9 / winograd_total_time << " GFLOPS)" << std::endl;
    
    std::cout << "\n=== Correctness Check ===" << std::endl;
    bool all_correct = true;
    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeO = (H-2) * (W-2);
        
        results[l].passed = true;
        for (long i = 0; i < N * K * sizeO; i++) {
            if (std::abs((h_custom_outputs[l][i] - h_baseline_outputs[l][i]) / (h_baseline_outputs[l][i] + 1e-8f)) > 1e-4f) {
                results[l].passed = false;
                all_correct = false;
                break;
            }
        }
        
        std::cout << "Layer " << std::setw(2) << l << ": " 
                  << (results[l].passed ? GREEN "CORRECT" RESET : RED "INCORRECT" RESET)
                  << " (Speedup: " << std::fixed << std::setprecision(2) << results[l].speedup << "x)" << std::endl;
    }
    
    double overall_speedup = baseline_total_time / winograd_total_time;
    std::cout << "\n=== Final Results ===" << std::endl;
    if (all_correct) {
        std::cout << GREEN "Results are correct!" RESET << std::endl;
    } else {
        std::cout << RED "Results are different!" RESET << std::endl;
    }

    std::cout << "Naive:    " << std::fixed << std::setprecision(2) << baseline_total_time * 1000 
              << " ms (" << total_flops * 1e-9 / baseline_total_time << " GFLOPS)" << std::endl;
    std::cout << "Winograd: " << std::fixed << std::setprecision(2) << winograd_total_time * 1000 
              << " ms (" << total_flops * 1e-9 / winograd_total_time << " GFLOPS)" << std::endl;
    std::cout << "Overall speedup: " << std::fixed << std::setprecision(2) << overall_speedup << "x" << std::endl;
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    return all_correct ? 0 : -1;
}