#include "gpu_manager.h"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include <cassert>

// 真正的管理器数据结构，只有 nvcc 编译器知道它的存在
struct GPUManager::Impl {
    std::unordered_map<size_t, std::vector<double*>> memory_pool;
    std::mutex pool_mutex;

    static constexpr int NUM_STREAMS = 4; // 强烈建议开多个流以支持并发
    cudaStream_t stream_pool[NUM_STREAMS];
    std::atomic<unsigned int> stream_idx{0};
};

// 单例获取
GPUManager& GPUManager::getInstance() {
    static GPUManager instance;
    return instance;
}

// 构造与析构
GPUManager::GPUManager() : pimpl(new Impl()) {
    for (int i = 0; i < Impl::NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&pimpl->stream_pool[i]));
    }
}

GPUManager::~GPUManager() {
    clear_pool();
    for (int i = 0; i < Impl::NUM_STREAMS; ++i) {
        cudaStreamDestroy(pimpl->stream_pool[i]);
    }
    delete pimpl;
}

// 显存池操作路由到 pimpl
double* GPUManager::allocate_device_memory(size_t num_elements) {
    // std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
    // if (pimpl->memory_pool.find(num_elements) != pimpl->memory_pool.end() && 
    //     !pimpl->memory_pool[num_elements].empty()) {
    //     double* d_ptr = pimpl->memory_pool[num_elements].back();
    //     pimpl->memory_pool[num_elements].pop_back();
    //     // CUDA_CHECK(cudaMemset(d_ptr, 0, num_elements * sizeof(double)));
    //     return d_ptr;
    // }
    double* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, num_elements * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, num_elements * sizeof(double)));
    return d_ptr;
}

void GPUManager::free_device_memory(double* d_ptr, size_t num_elements) {
    if (d_ptr == nullptr) return;
    // std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
    // pimpl->memory_pool[num_elements].push_back(d_ptr);
    CUDA_CHECK(cudaFree(d_ptr));
}

void GPUManager::clear_pool() {
    // std::lock_guard<std::mutex> lock(pimpl->pool_mutex);
    // for (auto& pair : pimpl->memory_pool) {
    //     for (double* d_ptr : pair.second) {
    //         cudaFree(d_ptr);
    //     }
    // }
    pimpl->memory_pool.clear();
}

cudaStream_t GPUManager::get_stream() {
    unsigned int cur = pimpl->stream_idx.fetch_add(1);
    return pimpl->stream_pool[cur % Impl::NUM_STREAMS];
}

void GPUManager::synchronize_all() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUManager::sync_to_gpu(const double* h_ptr, double* d_ptr, size_t num_elements) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, num_elements * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUManager::sync_to_cpu(double* h_ptr, const double* d_ptr, size_t num_elements) {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, num_elements * sizeof(double), cudaMemcpyDeviceToHost));
}