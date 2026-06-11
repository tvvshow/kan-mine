// CUTLASS int8 GEMM roofline benchmark for RTX 3080 Ti (sm_86).
//
// Purpose: measure the SUSTAINED int8 tensor-core throughput a production-grade
// GEMM (CUTLASS, same class lpminer/SRBMiner use) reaches on THIS card, so we
// know the ceiling our Pearl jackpot kernel could ever approach. Pearl "TH/s"
// here == TMAC/s (1 tera multiply-accumulate / s), since work/draw is counted
// in MACs. SRBMiner logged 106 TH/s, lpminer ~91, our WMMA kernel ~30.
//
// We bench a compute-bound shape that fits 12 GB (C int32 must be stored):
//   m = n = 16384, k = 4096  -> C = 16384^2 * 4 B = 1.07 GB, A=B=64 MB.
//   MAC = m*n*k = 1.0995e12 per GEMM.
// Real mining is m=n=131072 (C would be 68 GB, never materialized — folded in
// the epilogue), but tensor throughput is shape-independent once tiles saturate,
// so 16384 is a faithful roofline proxy.
//
// A row-major, B column-major (Bt row-major), C row-major — the canonical int8
// tensorop layout, identical to what our kernel computes (A * Bt^T).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementAcc = int32_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// sm_80 int8 tensorop, 16x8x32 IMMA instruction — the exact MMA lpminer uses.
template <typename TBShape, typename WarpShape, int Stages>
using GemmT = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TBShape, WarpShape,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc, ElementAcc>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

static const char* cublasStatusName(cutlass::Status s) { return cutlass::cutlassGetStatusString(s); }

template <typename Gemm>
double run_one(const char* tag, int m, int n, int k, int iters,
               int8_t* dA, int8_t* dB, int32_t* dC) {
  typename Gemm::Arguments args(
      {m, n, k},
      {dA, k},          // A row-major, lda = k
      {dB, k},          // B column-major, ldb = k
      {dC, n},          // C row-major, ldc = n
      {dC, n},
      {1, 0});

  Gemm gemm;
  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) { printf("  %-28s can_implement FAIL: %s\n", tag, cublasStatusName(st)); return -1; }

  size_t ws = Gemm::get_workspace_size(args);
  void* dWs = nullptr;
  if (ws) cudaMalloc(&dWs, ws);

  st = gemm.initialize(args, dWs);
  if (st != cutlass::Status::kSuccess) { printf("  %-28s initialize FAIL: %s\n", tag, cublasStatusName(st)); if(dWs)cudaFree(dWs); return -1; }

  // warmup
  for (int i = 0; i < 3; ++i) { st = gemm(); }
  if (st != cutlass::Status::kSuccess) { printf("  %-28s run FAIL: %s\n", tag, cublasStatusName(st)); if(dWs)cudaFree(dWs); return -1; }
  cudaDeviceSynchronize();

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) gemm();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  if (dWs) cudaFree(dWs);

  double sec = ms / 1e3 / iters;
  double mac = (double)m * n * k;
  double tmac = mac / sec / 1e12;   // TMAC/s == Pearl TH/s
  double tops = 2.0 * mac / sec / 1e12;
  printf("  %-28s %8.3f ms/gemm   %7.2f TMAC/s (= %6.2f Pearl-TH/s)   %7.1f int8-TOPS\n",
         tag, ms / iters, tmac, tmac, tops);
  return tmac;
}

int main(int argc, char** argv) {
  int m = 16384, n = 16384, k = 4096, iters = 20;
  if (argc > 1) m = atoi(argv[1]);
  if (argc > 2) n = atoi(argv[2]);
  if (argc > 3) k = atoi(argv[3]);
  if (argc > 4) iters = atoi(argv[4]);

  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("GPU: %s  sm_%d%d  SMs=%d  %.0f MHz  mem %.0f GB/s\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         p.clockRate / 1e3, 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1e6);
  printf("shape m=%d n=%d k=%d  iters=%d  MAC/gemm=%.3e\n\n", m, n, k, iters, (double)m * n * k);

  int8_t *dA, *dB; int32_t *dC;
  cudaMalloc(&dA, (size_t)m * k);
  cudaMalloc(&dB, (size_t)k * n);
  cudaMalloc(&dC, (size_t)m * n * sizeof(int32_t));
  cudaMemset(dA, 3, (size_t)m * k);
  cudaMemset(dB, 5, (size_t)k * n);

  // Sweep a few production tile shapes. <128,256,64> and <256,128,64> are the
  // workhorse int8 sm_80 tiles (lpminer uses 128x256). 3-4 pipeline stages.
  run_one<GemmT<cutlass::gemm::GemmShape<256,128,64>, cutlass::gemm::GemmShape<64,64,64>, 3>>("TB256x128x64 W64x64 s3", m,n,k,iters,dA,dB,dC);
  run_one<GemmT<cutlass::gemm::GemmShape<128,256,64>, cutlass::gemm::GemmShape<64,64,64>, 3>>("TB128x256x64 W64x64 s3", m,n,k,iters,dA,dB,dC);
  run_one<GemmT<cutlass::gemm::GemmShape<128,128,64>, cutlass::gemm::GemmShape<64,64,64>, 3>>("TB128x128x64 W64x64 s3", m,n,k,iters,dA,dB,dC);
  run_one<GemmT<cutlass::gemm::GemmShape<128,256,64>, cutlass::gemm::GemmShape<64,64,64>, 4>>("TB128x256x64 W64x64 s4", m,n,k,iters,dA,dB,dC);
  run_one<GemmT<cutlass::gemm::GemmShape<256,128,128>,cutlass::gemm::GemmShape<64,64,128>,3>>("TB256x128x128 W64x64 s3",m,n,k,iters,dA,dB,dC);
  run_one<GemmT<cutlass::gemm::GemmShape<128,128,128>,cutlass::gemm::GemmShape<64,64,128>,3>>("TB128x128x128 W64x64 s3",m,n,k,iters,dA,dB,dC);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
