// mma_microtest.cu — nail the mma.sync.m16n8k32.s8 register/fragment layout
// with RANDOM data + a CPU reference, BEFORE building any kernel around it.
//
// This is the step tc_imma never did: an all-ones test (C=K=32 everywhere)
// cannot distinguish a correct fragment layout from a transposed/permuted one.
// Random int8 in [-64,64] makes ANY layout error show up as a mismatch.
//
// One warp computes ONE 16x8 output tile:
//   C[i][j] = sum_{l=0..31} A[i][l] * Bt[j][l]        (i in 0..15, j in 0..7)
// where A is 16x32 row-major and Bt is 8x32 (n-by-k) — exactly our A'*Bt'^T
// orientation. Instruction:
//   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//
// Hypothesised layout (validated/refuted by this test):
//   group = lane>>2 (0..7), tig = lane&3 (0..3)
//   A: a0(row=group,    k=tig*4+0..3)   a1(row=group+8, k=tig*4+0..3)
//      a2(row=group,    k=tig*4+16..)   a3(row=group+8, k=tig*4+16..)
//   B: b0(n=group, k=tig*4+0..3)        b1(n=group, k=tig*4+16..)
//   C: c0(r=group,  c=tig*2)  c1(r=group,  c=tig*2+1)
//      c2(r=group+8,c=tig*2)  c3(r=group+8,c=tig*2+1)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t pack4(const signed char* p) {
  // little-endian pack of 4 int8 (reinterpreted as bytes) into a b32 register
  return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) |
         ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}

// A: 16x32 row-major (lda=32). Bt: 8x32 row-major (ldb=32, = n-by-k). C: 16x8 row-major.
__global__ void mma_one(const signed char* A, const signed char* Bt, int32_t* C) {
  int lane = threadIdx.x & 31;
  int group = lane >> 2;
  int tig = lane & 3;

  uint32_t a0 = pack4(&A[(group)   * 32 + tig * 4 +  0]);
  uint32_t a1 = pack4(&A[(group+8) * 32 + tig * 4 +  0]);
  uint32_t a2 = pack4(&A[(group)   * 32 + tig * 4 + 16]);
  uint32_t a3 = pack4(&A[(group+8) * 32 + tig * 4 + 16]);

  uint32_t b0 = pack4(&Bt[(group) * 32 + tig * 4 +  0]);
  uint32_t b1 = pack4(&Bt[(group) * 32 + tig * 4 + 16]);

  int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

  // store per hypothesised C layout
  C[(group)   * 8 + tig * 2 + 0] = c0;
  C[(group)   * 8 + tig * 2 + 1] = c1;
  C[(group+8) * 8 + tig * 2 + 0] = c2;
  C[(group+8) * 8 + tig * 2 + 1] = c3;
}

int main() {
  const int M = 16, N = 8, K = 32;
  signed char hA[M*K], hBt[N*K];
  int32_t hC[M*N], ref[M*N];

  // deterministic pseudo-random int8 in [-64,64]
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
  mma_one<<<1, 32>>>(dA, dBt, dC);
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
  printf(bad ? "\nLAYOUT WRONG: %d/%d cells mismatch\n" : "\nLAYOUT CORRECT: all %d cells match\n",
         bad ? bad : M*N, M*N);
  return bad ? 1 : 0;
}
