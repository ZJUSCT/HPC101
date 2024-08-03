#ifndef _JUDGER_H_
#define _JUDGER_H_

#include <chrono>
#include <string>

constexpr double TOL   = 1.0e-12;
constexpr int MAX_ITER = 1e6;

using Duration = std::chrono::duration<int64_t, std::nano>;

void read_data(const std::string filename, int* N, double** A, double** b, double** x);
void judge(int iter, Duration duration, int N, double* A, double* b, double* x);

#endif