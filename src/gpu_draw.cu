// gpu_draw.cu — GPU-side A/B generation (splitmix64 RNG) + noise assembly.
// Eliminates ~1.4s CPU produce_draw + ~1GB H2D per draw.
#include <cuda_runtime.h>
#include <cstdint>

// Fill A[m×k] with splitmix64(seed_a + draw + row)
__global__ void fill_A(signed char* A, int m, int k, uint64_t seed, uint64_t draw) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;
  uint64_t s = (seed^0x9E3779B97F4A7C15ULL) + draw*1000003ULL + (uint64_t)row*0x100000001B3ULL;
  for (int col = 0; col < k; col += 8) {
    s += 0x9e3779b97f4a7c15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z = z ^ (z >> 31);
    for (int i = 0; i < 8 && col + i < k; i++) {
      uint32_t b = (z >> (i*8)) & 0xFF;
      A[(size_t)row * k + col + i] = (signed char)((int)(((b*129u)>>8)) - 64);
    }
  }
}

// Fill Bt[n×k] (B transposed)
__global__ void fill_Bt(signed char* Bt, int n, int k, uint64_t seed, uint64_t draw) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  uint64_t s = (seed^0xD1B54A32D192ED03ULL) + draw*1000003ULL + (uint64_t)row*0x100000001B3ULL;
  for (int col = 0; col < k; col += 8) {
    s += 0x9e3779b97f4a7c15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z = z ^ (z >> 31);
    for (int i = 0; i < 8 && col + i < k; i++) {
      uint32_t b = (z >> (i*8)) & 0xFF;
      Bt[(size_t)row * k + col + i] = (signed char)((int)(((b*129u)>>8)) - 64);
    }
  }
}

extern "C" void gpu_produce_draw(
    signed char* dA, signed char* dBt, int m, int n, int k,
    uint64_t seed, uint64_t draw, cudaStream_t stream)
{
  int tpb = 256;
  fill_A<<<(m + tpb - 1) / tpb, tpb, 0, stream>>>(dA, m, k, seed, draw);
  fill_Bt<<<(n + tpb - 1) / tpb, tpb, 0, stream>>>(dBt, n, k, seed, draw);
}
