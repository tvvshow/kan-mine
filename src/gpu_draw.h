// gpu_draw.h — GPU-side draw generation interface
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Generate A[m×k] and Bt[n×k] on GPU via splitmix64 RNG (matches CPU produce_draw step a)
extern "C" void gpu_produce_draw(
    signed char* dA, signed char* dBt, int m, int n, int k,
    uint64_t seed, uint64_t draw, cudaStream_t stream = 0);
