// mma_warpgemm_test.cu — final inner-loop primitive: ONE warp computes a full
// 16x16 output tile = A[16xK] * Bt[16xK]^T over the real K=4096, streaming K
// through smem in RK=256 chunks (one rank), ldmatrix+mma per 32-wide k-step,
// N=16 handled as two mma n-tiles (Bt rows 0-7 and 8-15). Random data, CPU ref.
//
// Proves: K-accumulation, multi-N-tile, ldmatrix streaming along K. After this,
// only threadblock tiling + cp.async pipeline + the rank-boundary fold remain.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define K 4096
#define RK 256          // smem k-chunk (= one rank)

__global__ void warpgemm(const signed char* A, const signed char* Bt, int32_t* C) {
  int lane = threadIdx.x & 31;
  int group = lane >> 2, tig = lane & 3;

  __shared__ __align__(16) signed char As[16 * RK];
  __shared__ __align__(16) signed char Bs[16 * RK];

  int32_t acc[2][4] = {{0,0,0,0},{0,0,0,0}};   // 2 n-tiles, 4 C regs each

  for (int kb = 0; kb < K; kb += RK) {
    for (int i = threadIdx.x; i < 16 * RK; i += 32) {
      int r = i / RK, c = i % RK;
      As[i] = A[r * K + kb + c];
      Bs[i] = Bt[r * K + kb + c];
    }
    __syncwarp();

    for (int k0 = 0; k0 < RK; k0 += 32) {
      uint32_t addrA = __cvta_generic_to_shared(&As[(lane & 15) * RK + k0 + (lane >> 4) * 16]);
      uint32_t a0, a1, a2, a3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addrA));
      #pragma unroll
      for (int t = 0; t < 2; t++) {
        uint32_t addrB = __cvta_generic_to_shared(&Bs[(8 * t + (lane & 7)) * RK + k0 + ((lane >> 3) & 1) * 16]);
        uint32_t b0, b1;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(b0), "=r"(b1) : "r"(addrB));
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+r"(acc[t][0]), "+r"(acc[t][1]), "+r"(acc[t][2]), "+r"(acc[t][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
    __syncwarp();
  }

  #pragma unroll
  for (int t = 0; t < 2; t++) {
    C[(group)   * 16 + 8 * t + tig * 2 + 0] = acc[t][0];
    C[(group)   * 16 + 8 * t + tig * 2 + 1] = acc[t][1];
    C[(group+8) * 16 + 8 * t + tig * 2 + 0] = acc[t][2];
    C[(group+8) * 16 + 8 * t + tig * 2 + 1] = acc[t][3];
  }
}

int main() {
  const int M = 16, N = 16;
  static signed char hA[M*K], hBt[N*K];
  static int32_t hC[M*N], ref[M*N];
  uint64_t s = 0xfeed1234cafe5678ull;
  auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (signed char)((int)(s & 0x7f) - 64); };
  for (int i = 0; i < M*K; i++) hA[i] = rnd();
  for (int i = 0; i < N*K; i++) hBt[i] = rnd();
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      long acc = 0;
      for (int l = 0; l < K; l++) acc += (int)hA[i*K+l] * (int)hBt[j*K+l];
      ref[i*N+j] = (int32_t)acc;
    }
  signed char *dA, *dBt; int32_t *dC;
  cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dBt, sizeof(hBt)); cudaMalloc(&dC, sizeof(hC));
  cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
  cudaMemcpy(dBt, hBt, sizeof(hBt), cudaMemcpyHostToDevice);
  warpgemm<<<1, 32>>>(dA, dBt, dC);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA err: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      if (hC[i*N+j] != ref[i*N+j]) {
        if (bad < 12) printf("MISMATCH C[%2d][%2d] gpu=%-8d cpu=%-8d\n", i, j, hC[i*N+j], ref[i*N+j]);
        bad++;
      }
  printf(bad ? "\nWARP-GEMM WRONG: %d/%d mismatch\n" : "\nWARP-GEMM CORRECT: all %d match (K=%d)\n",
         bad ? bad : M*N, M*N, K);
  return bad ? 1 : 0;
}
