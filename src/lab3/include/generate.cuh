#pragma once
#include <random>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#define InitRandom()                           \
    std::random_device r;                      \
    std::default_random_engine generator(r()); \
    std::uniform_int_distribution<> distribution(-2, 2);

auto get_seed() {
    std::random_device rd;
    return rd();
}

void GenerateInput(int8_t *a, int8_t *w) {
    InitRandom();
    for (int i = 0; i < N * H * W * C; ++i) {
        a[i] = static_cast<int8_t>(distribution(generator));
    }
    for (int i = 0; i < K * R * S * C; ++i) {
        w[i] = static_cast<int8_t>(distribution(generator));
    }
}

// GenerateInput for cutlass::half_t with integer
void GenerateInput(cutlass::half_t *a, cutlass::half_t *w) {
    InitRandom();
    for (int i = 0; i < N * H * W * C; ++i) {
        a[i] = static_cast<cutlass::half_t>(distribution(generator));
    }
    for (int i = 0; i < K * R * S * C; ++i) {
        w[i] = static_cast<cutlass::half_t>(distribution(generator));
    }
}