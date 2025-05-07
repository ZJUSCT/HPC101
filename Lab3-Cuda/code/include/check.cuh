#pragma once
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

const double eps = 1e-2;


template <typename T>
bool check(const thrust::host_vector<T>& h_C,
           const thrust::host_vector<T>& h_C_ref, double norm = 1.0) {
    assert(h_C.size() == h_C_ref.size());
    for (size_t i = 0; i < h_C.size(); i++) {
        double diff = ((double)h_C[i] - (double)h_C_ref[i]) / norm;
        if (std::isnan(diff) || std::abs(diff) > eps) {
            std::cout << "Mismatch at index " << i << ": " << (double)h_C[i]
                << " != " << (double)h_C_ref[i] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T>
bool print_matrix(const thrust::host_vector<T>& h_C, const int m, const int n, const int l_row, const int l_col) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << (double)h_C[i * l_row + j * l_col] << " ";
        }
        std::cout << std::endl;
    }
    return true;
}
