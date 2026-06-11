// mma_ldmatrix_test.cu — validate ldmatrix.x4/x2 addressing feeds the SAME
// registers the (already-validated) manual layout did. Builds on the proven
// mma.sync.m16n8k32 core; only the A/B load path changes to ldmatrix.
//
// A 16x32 int8 -> smem -> ldmatrix.x4 (four 8x8 b16 tiles: TL,BL,TR,BR = a0..a3)
// Bt 8x32 int8 -> smem -> ldmatrix.x2 (two 8x8 b16 tiles: L,R = b0,b1)
// int8 viewed as b16: 32 int8 cols = 16 b16 cols, row stride = 32 bytes.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void mma_ld(const signed char* A, const signed char* Bt, int32_t* C) {
  int lane = threadIdx.x & 31;
  int group = lane >> 2, tig = lane & 3;

  __shared__ __align__(16) signed char As[16 * 32];
  __shared__ __align__(16) signed char Bs[8 * 32];
  for (int i = threadIdx.x; i < 16 * 32; i += 32) As[i] = A[i];
  for (int i = threadIdx.x; i < 8 * 32; i += 32) Bs[i] = Bt[i];
  __syncwarp();

  // A: lane supplies row (lane&15), right-half byte offset (lane>>4)*16.
  uint32_t addrA = __cvta_generic_to_shared(&As[(lane & 15) * 32 + (lane >> 4) * 16]);
  // Bt: lane supplies row (lane&7), right-half byte offset ((lane>>3)&1)*16.
  uint32_t addrB = __cvta_generic_to_shared(&Bs[(lane & 7) * 32 + ((lane >> 3) & 1) * 16]);

  uint32_t a0, a1, a2, a3, b0, b1;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addrA));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b0), "=r"(b1) : "r"(addrB));

  int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

  C[(group)   * 8 + tig * 2 + 0] = c0;
  C[(group)   * 8 + tig * 2 + 1] = c1;
  C[(group+8) * 8 + tig * 2 + 0] = c2;
  C[(group+8) * 8 + tig * 2 + 1] = c3;
}

int main() {
  const int M = 16, N = 8, K = 32;
  signed char hA[M*K], hBt[N*K];
  int32_t hC[M*N], ref[M*N];
  uint64_t s = 0x12345678abcdef01ull;
  auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (signed char)((int)(s & 0x7f) - 64); };
  for (int i = 0; i < M*K; i++) hA[i] = rnd();
  for (int i = 0; i < N*K; i++) hBt[i] = rnd();
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      int acc = 0;
      for (int l = 0; l < K; l++) acc += (int)hA[i*K+l] * (int)hBt[j*K+l];
      ref[i*N+j] = acc;
    }
  signed char *dA, *dBt; int32_t *dC;
  cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dBt, sizeof(hBt)); cudaMalloc(&dC, sizeof(hC));
  cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
  cudaMemcpy(dBt, hBt, sizeof(hBt), cudaMemcpyHostToDevice);
  mma_ld<<<1, 32>>>(dA, dBt, dC);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA err: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      if (hC[i*N+j] != ref[i*N+j]) {
        if (bad < 12) printf("MISMATCH C[%2d][%d] gpu=%-7d cpu=%-7d\n", i, j, hC[i*N+j], ref[i*N+j]);
        bad++;
      }
  printf(bad ? "\nLDMATRIX WRONG: %d/%d mismatch\n" : "\nLDMATRIX CORRECT: all %d match\n",
         bad ? bad : M*N, M*N);
  return bad ? 1 : 0;
}
