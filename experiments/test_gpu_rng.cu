// test_gpu_rng.cu — verify GPU RNG produces same output as CPU
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

__device__ __forceinline__ uint64_t splitmix64(uint64_t& s) {
  s += 0x9e3779b97f4a7c15ull;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

__global__ void fill_A(signed char* A, int m, int k, uint64_t seed, uint64_t draw) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;
  uint64_t st = (seed^0x9E3779B97F4A7C15ULL) + draw*1000003ULL + (uint64_t)row*0x100000001B3ULL;
  for (int col = 0; col < k; col += 8) {
    uint64_t z = splitmix64(st);
    for (int i = 0; i < 8 && col + i < k; i++) {
      uint32_t b = (z >> (i*8)) & 0xFF;
      A[(size_t)row * k + col + i] = (signed char)((int)(((b*129u)>>8)) - 64);
    }
  }
}

void cpu_fill_A(signed char* A, int m, int k, uint64_t seed, uint64_t draw) {
  for (int row = 0; row < m; row++) {
    uint64_t st = (seed^0x9E3779B97F4A7C15ULL) + draw*1000003ULL + (uint64_t)row*0x100000001B3ULL;
    for (int col = 0; col < k; col += 8) {
      st += 0x9e3779b97f4a7c15ull;
      uint64_t z = st;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      z = z ^ (z >> 31);
      for (int i = 0; i < 8 && col + i < k; i++) {
        uint32_t b = (z >> (i*8)) & 0xFF;
        A[row * k + col + i] = (signed char)((int)(((b*129u)>>8)) - 64);
      }
    }
  }
}

int main() {
  const int m = 1024, k = 4096;
  uint64_t seed = 12345, draw = 0;

  std::vector<signed char> cpu_A(m*k), gpu_A(m*k);
  cpu_fill_A(cpu_A.data(), m, k, seed, draw);

  signed char* dA;
  cudaMalloc(&dA, m*k);
  fill_A<<<(m+255)/256, 256>>>(dA, m, k, seed, draw);
  cudaMemcpy(gpu_A.data(), dA, m*k, cudaMemcpyDeviceToHost);
  cudaFree(dA);

  int diff = 0;
  for (int i = 0; i < m*k && diff < 10; i++) {
    if (cpu_A[i] != gpu_A[i]) {
      printf("DIFF [%d]: cpu=%d gpu=%d\n", i, (int)cpu_A[i], (int)gpu_A[i]);
      diff++;
    }
  }
  printf(diff ? "FAIL: %d diffs\n" : "PASS: GPU RNG matches CPU\n", diff);
  return diff ? 1 : 0;
}
