# ============================================================================
#  HPC101 Lab 4.5 — Makefile
#  在 H800 (sm_90a) 上编译 benchmark。按需调整 CUDA_HOME / ARCH。
# ============================================================================
CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_90a
NVCCFLAGS := -arch=$(ARCH) -O3 -std=c++17 -lineinfo -Iinclude
LDLIBS    := -lcublas -lcudart -lcuda

SRCS := benchmark.cu \
        baseline/baseline_fp64.cu \
        baseline/cublas_baseline.cu \
        baseline/cublas_emulated.cu \
        utils.cu \
        submit/my_int8_fp64.cu
OBJS := $(SRCS:.cu=.o)
DEPS := include/gemm_api.h include/utils.h

.PHONY: all clean run

all: benchmark

benchmark: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LDLIBS) -o $@

%.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 快速默认测试
run: benchmark
	./benchmark 1024,2048,4096 2,4,6,8 5

clean:
	rm -f $(OBJS) benchmark
