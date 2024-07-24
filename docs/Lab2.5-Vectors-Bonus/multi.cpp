#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#define MAXN 10000000

double *a;
double *b;
double *c;
double *d;

int main()
{
    printf("Initializing\n");
    a = (double *)malloc(sizeof(double) * MAXN * 48); // 4*12
    b = (double *)malloc(sizeof(double) * MAXN * 48); // 12*4
    c = (double *)malloc(sizeof(double) * MAXN * 16); // 4*4
    d = (double *)malloc(sizeof(double) * MAXN * 16); // 4*4
    memset(c, 0, sizeof(double) * MAXN * 16);
    memset(d, 0, sizeof(double) * MAXN * 16);
    for (int i = 0; i < MAXN * 48; i++)
    {
        *(a + i) = 1.0 / (rand() + 1);
        *(b + i) = 1.0 / (rand() + 1);
    }

    printf("Raw computing\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < MAXN; n++)
    {
        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 12; j++)
                {
                    *(c + n * 16 + i * 4 + k) += *(a + n * 48 + i * 12 + j) * *(b + n * 48 + j * 4 + k);
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double time1 = std::chrono::duration<double>(end - start).count();

    printf("New computing\n");
    start = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < MAXN; n++)
    {
        /* 可以修改的代码区域 */
        // -----------------------------------
        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 12; j++)
                {
                    *(d + n * 16 + i * 4 + k) += *(a + n * 48 + i * 12 + j) * *(b + n * 48 + j * 4 + k);
                }
            }
        }
        // -----------------------------------
    }

    end = std::chrono::high_resolution_clock::now();

    double time2 = std::chrono::duration<double>(end - start).count();
    printf("raw time=%lfs\nnew time=%lfs\nspeed up:%lfx\nChecking\n", time1, time2, time1 / time2);

    for (int i = 0; i < MAXN * 16; i++)
    {
        if (fabs(c[i] - d[i]) / d[i] > 0.0001)
        {
            printf("Check Failed at %d\n", i);
            return 0;
        }
    }
    printf("Check Passed\n");
}
