#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include "winograd.h"

#define LOOP_NUM 3
#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

typedef struct {
    int *C, *H, *W, *K, *Batch;
    int layer_num;
} config_t;

typedef struct {
    double baseline_time;
    double winograd_time;
    double baseline_gflops;
    double winograd_gflops;
    double speedup;
    int passed;
} layer_result_t;

static double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1.e-6;
}

static config_t* load_config(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;
    
    config_t *cfg = malloc(sizeof(config_t));
    fscanf(f, "%d", &cfg->layer_num);
    
    cfg->C = malloc(sizeof(int) * cfg->layer_num);
    cfg->H = malloc(sizeof(int) * cfg->layer_num);
    cfg->W = malloc(sizeof(int) * cfg->layer_num);
    cfg->K = malloc(sizeof(int) * cfg->layer_num);
    cfg->Batch = malloc(sizeof(int) * cfg->layer_num);
    
    for (int i = 0; i < cfg->layer_num; i++) {
        fscanf(f, "%d%d%d%d%d", &cfg->C[i], &cfg->H[i], &cfg->W[i], &cfg->K[i], &cfg->Batch[i]);
    }
    fclose(f);
    return cfg;
}

// Fast random number generator
static unsigned int fast_rand_seed = 1;
static inline float fast_rand() {
    fast_rand_seed = fast_rand_seed * 1103515245 + 12345;
    return (float)(fast_rand_seed & 0x7FFFFFFF) / 0x7FFFFFFF * 9.0f + 1.0f;
}

static int validate_and_benchmark(config_t *cfg) {    
    layer_result_t* results = malloc(sizeof(layer_result_t) * cfg->layer_num);
    
    // Initialize random seed
    fast_rand_seed = (unsigned int)time(NULL);
    
    // Allocate storage for baseline outputs to compare later
    float** baseline_outputs = malloc(sizeof(float*) * cfg->layer_num);
    float** custom_outputs = malloc(sizeof(float*) * cfg->layer_num);
    
    // Pre-generate all random data to ensure same inputs for both implementations
    float** all_images = malloc(sizeof(float*) * cfg->layer_num);
    float** all_filters = malloc(sizeof(float*) * cfg->layer_num);
    
    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeI = H * W, sizeF = 9;
        
        all_images[l] = aligned_alloc(64, sizeof(float) * N * C * sizeI);
        all_filters[l] = aligned_alloc(64, sizeof(float) * K * C * sizeF);
        
        for (long i = 0; i < N * C * sizeI; i++) all_images[l][i] = fast_rand();
        for (long i = 0; i < K * C * sizeF; i++) all_filters[l][i] = fast_rand();
    }
    
    printf("=== Running Naive Convolution ===\n");
    double baseline_total_time = 0;
    long total_flops = 0;
    
    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeI = H * W, sizeF = 9, sizeO = (H-2) * (W-2);
        long P = N * (H-2)/2 * (W-2)/2;
        
        float* image = all_images[l];
        float* filter = all_filters[l];
        float* out = aligned_alloc(64, sizeof(float) * N * K * sizeO);
        float* U = aligned_alloc(64, sizeof(float) * 16 * K * C);
        float* V = aligned_alloc(64, sizeof(float) * 16 * C * P);
        float* M = aligned_alloc(64, sizeof(float) * 16 * K * P);

        // Warmup
        naive_conv(image, filter, out,
                   U, V, M,
                   H, W, C, K, N);
        
        double start = timestamp();
        for (int i = 0; i < LOOP_NUM; i++) {
            naive_conv(image, filter, out,
                       U, V, M,
                       H, W, C, K, N);
        }
        results[l].baseline_time = (timestamp() - start) / LOOP_NUM;
        
        // Save baseline output for later comparison
        baseline_outputs[l] = aligned_alloc(64, sizeof(float) * N * K * sizeO);
        memcpy(baseline_outputs[l], out, sizeof(float) * N * K * sizeO);
        
        long flops = (long)N * K * C * (H-2) * (W-2) * 18;
        results[l].baseline_gflops = flops * 1e-9 / results[l].baseline_time;
        
        printf("Layer %2d: %.2f ms (%.1f GFLOPS)\n", l, results[l].baseline_time * 1000, results[l].baseline_gflops);
        
        baseline_total_time += results[l].baseline_time;
        total_flops += flops;
        
        free(out); free(U); free(V); free(M);
    }
    
    printf("Baseline Total: %.2f ms (%.1f GFLOPS)\n", baseline_total_time * 1000, total_flops * 1e-9 / baseline_total_time);
    
    printf("\n=== Running Winograd Convolution ===\n");
    double winograd_total_time = 0;
    
    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeI = H * W, sizeF = 9, sizeO = (H-2) * (W-2);
        long P = N * (H-2)/2 * (W-2)/2;
        
        float* image = all_images[l];
        float* filter = all_filters[l];
        float* out = aligned_alloc(64, sizeof(float) * N * K * sizeO);
        float* U = aligned_alloc(64, sizeof(float) * 16 * K * C);
        float* V = aligned_alloc(64, sizeof(float) * 16 * C * P);
        float* M = aligned_alloc(64, sizeof(float) * 16 * K * P);
        
        // Warmup
        winograd_conv(image, filter, out,
                      U, V, M,
                      H, W, C, K, N);
        
        double start = timestamp();
        for (int i = 0; i < LOOP_NUM; i++) {
            winograd_conv(image, filter, out,
                          U, V, M,
                          H, W, C, K, N);
        }
        results[l].winograd_time = (timestamp() - start) / LOOP_NUM;
        
        // Save custom output for later comparison
        custom_outputs[l] = aligned_alloc(64, sizeof(float) * N * K * sizeO);
        memcpy(custom_outputs[l], out, sizeof(float) * N * K * sizeO);
        
        long flops = (long)N * K * C * (H-2) * (W-2) * 18;
        results[l].winograd_gflops = flops * 1e-9 / results[l].winograd_time;
        results[l].speedup = results[l].baseline_time / results[l].winograd_time;
        
        printf("Layer %2d: %.2f ms (%.1f GFLOPS)\n", l, results[l].winograd_time * 1000, results[l].winograd_gflops);
        
        winograd_total_time += results[l].winograd_time;
        
        free(out); free(U); free(V); free(M);
    }
    
    printf("Custom Total: %.2f ms (%.1f GFLOPS)\n", winograd_total_time * 1000, total_flops * 1e-9 / winograd_total_time);
    
    printf("\n=== Correctness Check ===\n");
    int all_correct = 1;
    for (int l = 0; l < cfg->layer_num; l++) {
        int H = cfg->H[l], W = cfg->W[l], C = cfg->C[l], K = cfg->K[l], N = cfg->Batch[l];
        long sizeO = (H-2) * (W-2);
        
        results[l].passed = 1;
        for (long i = 0; i < N * K * sizeO; i++) {
            if (fabs((custom_outputs[l][i] - baseline_outputs[l][i]) / (baseline_outputs[l][i] + 1e-8)) > 1e-4) {
                results[l].passed = 0;
                all_correct = 0;
                break;
            }
        }
        
        printf("Layer %2d: %s (Speedup: %.2fx)\n", 
               l, results[l].passed ? GREEN "CORRECT" RESET : RED "INCORRECT" RESET, 
               results[l].speedup);
        
        free(baseline_outputs[l]);
        free(custom_outputs[l]);
    }
    
    free(baseline_outputs);
    free(custom_outputs);
    
    // Free pre-generated data
    for (int l = 0; l < cfg->layer_num; l++) {
        free(all_images[l]);
        free(all_filters[l]);
    }
    free(all_images);
    free(all_filters);
    
    double overall_speedup = baseline_total_time / winograd_total_time;
    printf("\n=== Final Results ===\n");
    if (all_correct) {
        printf(GREEN "Results are correct!\n" RESET);
    } else {
        printf(RED "Results are different!\n" RESET);
    }

    printf("Baseline: %.2f ms (%.1f GFLOPS)\n", baseline_total_time * 1000, total_flops * 1e-9 / baseline_total_time);
    printf("Custom:   %.2f ms (%.1f GFLOPS)\n", winograd_total_time * 1000, total_flops * 1e-9 / winograd_total_time);
    printf("Overall speedup: %.2fx\n", overall_speedup);
    
    free(results);
    return all_correct ? 0 : -1;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <config_file>\n", argv[0]);
        return -1;
    }
    
    config_t *cfg = load_config(argv[1]);
    if (!cfg) return -1;
    
    int result = validate_and_benchmark(cfg);
    
    free(cfg->C); free(cfg->H); free(cfg->W); free(cfg->K); free(cfg->Batch);
    free(cfg);
    return result;
}