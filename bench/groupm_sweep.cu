// GROUPM sweep benchmark for 4090 L2 optimization
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// Minimal kernel to test GROUPM impact on L2 hit rate
template<int GROUPM>
__global__ void grouped_raster_kernel(int8_t* A, int8_t* B, int* C, int M, int N, int K) {
    const int BM = 128, BN = 256;
    int nbm = (M + BM - 1) / BM;
    int nbn = (N + BN - 1) / BN;
    int pid = blockIdx.x;
    
    int band   = pid / (GROUPM * nbn);
    int first  = band * GROUPM;
    int gsz    = (nbm - first < GROUPM) ? (nbm - first) : GROUPM;
    int in_band = pid % (GROUPM * nbn);
    int bm = first + (in_band / nbn);
    int bn = in_band % nbn;
    
    if (bm >= nbm || bn >= nbn) return;
    
    int row = bm * BM + threadIdx.y;
    int col = bn * BN + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    int sum = 0;
    for (int k = 0; k < K; k += 16) {
        int8_t a = A[row * K + k + (threadIdx.x % 16)];
        int8_t b = B[col * K + k + (threadIdx.y % 16)];
        sum += a * b;
    }
    C[row * N + col] = sum;
}

template<int GROUPM>
float bench(int M, int N, int K, int8_t* d_A, int8_t* d_B, int* d_C) {
    const int BM = 128, BN = 256;
    int nbm = (M + BM - 1) / BM;
    int nbn = (N + BN - 1) / BN;
    int grid = nbm * nbn;
    dim3 block(16, 16);
    
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    
    for (int iter = 0; iter < 10; iter++) {
        grouped_raster_kernel<GROUPM><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count() / 10.0f;
    return ms;
}

int main() {
    const int M = 131072, N = 131072, K = 4096;
    
    int8_t *d_A, *d_B;
    int *d_C;
    
    cudaMalloc(&d_A, (size_t)M * K);
    cudaMalloc(&d_B, (size_t)N * K);
    cudaMalloc(&d_C, (size_t)M * N * sizeof(int));
    
    printf("GROUPM sweep on real Pearl config (M=N=%d, K=%d)\n\n", M, K);
    printf("GROUPM    Time(ms)    Speedup\n");
    printf("--------------------------------\n");
    
    float base = bench<8>(M, N, K, d_A, d_B, d_C);
    printf("  8       %.1f       1.00x (baseline)\n", base);
    
    float t16 = bench<16>(M, N, K, d_A, d_B, d_C);
    printf(" 16       %.1f       %.2fx\n", t16, base / t16);
    
    float t32 = bench<32>(M, N, K, d_A, d_B, d_C);
    printf(" 32       %.1f       %.2fx\n", t32, base / t32);
    
    float t64 = bench<64>(M, N, K, d_A, d_B, d_C);
    printf(" 64       %.1f       %.2fx\n", t64, base / t64);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
