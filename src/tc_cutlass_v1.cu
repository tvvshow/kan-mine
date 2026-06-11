// tc_cutlass_v1.cu — Day3 of PLAN_cutlass.md: CUTLASS threadblock mainloop +
// per-rank-chunk jackpot fold.
//
// WHY: the hand-written mainloop (tc_deep_pipeline.cu) is sweep-exhausted at
// ~61 TH/s on sm_86 while CUTLASS device::Gemm hits 131.7 TH/s (96% int8 peak)
// on the same card with the same 128x256x64 tile (bench/cutlass_int8_bench.cu).
// device::Gemm is unusable as a black box (its epilogue fires once after ALL of
// K; our fold must read accumulators after EVERY rank-chunk of K=256), so this
// file drives the CUTLASS *threadblock-level* MmaMultistage directly:
//
//   for chunk c in 0..k/rank:                     (16 chunks, k=4096 rank=256)
//     Mma mma(...); mma(rank/64, accum, itA, itB) (CUTLASS multistage mainloop)
//     fold accum -> jp_sh[tile][c%16]             (XOR + warp shuffle + rotl13)
//   blake3(jp) <= bound  ->  win                  (unchanged from tc_deep)
//
// Accumulators are NOT cleared between chunks (cumulative partial sums), which
// matches the CPU reference / official verifier — same semantics as the
// POSTCHECK-ok=1 tc_deep_pipeline.cu fold this is ported from.
//
// CUTLASS FragmentC layout (warp 64x64, inst m16n8k32, mma_tensor_op.h):
//   per-warp Array<int32,128> viewed as MmaOperandC[mIter + nIter*4], each an
//   Array<int32,4>; within a 16x8 instruction tile, lane = groupID*4 + tid:
//     e0,e1 -> row groupID,   cols tid*2, tid*2+1
//     e2,e3 -> row groupID+8, cols tid*2, tid*2+1
//   (identical to the empirically validated layout in bench/mma_microtest.cu).
// A jackpot tile is h=8 rows x w=16 cols, so within a warp:
//   8-row band jr (0..7): mIter = jr/2, half = jr&1 (e-pair: half*2, half*2+1)
//   16-col band jc (0..3): nIters 2*jc and 2*jc+1
//   XOR the 4 fragment values, warp-shuffle-XOR over 32 lanes = whole 8x16 tile.
//
// ABI: exports the same extern "C" tc_jackpot_search as tc_block/tc_deep, so
// build.sh / plainproof_gen / POSTCHECK need no changes — just link this .o.
//
// Build (box): nvcc -O3 -arch=sm_86 -std=c++17 -I$HOME/cutlass/include \
//              -c src/tc_cutlass_v1.cu -o build/tc_cutlass.o
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/mma_sm80.h"          // arch::Mma specialization for IMMA 16x8x32 s8
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

// ---- CUTLASS threadblock mainloop: the exact 131.7 TH/s bench recipe --------
using TBShape   = cutlass::gemm::GemmShape<128, 256, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kStages = 3;

using DefaultMmaT = cutlass::gemm::threadblock::DefaultMma<
    int8_t,  cutlass::layout::RowMajor,    16,   // A  (gathered A',  lda = k)
    int8_t,  cutlass::layout::ColumnMajor, 16,   // B  (gathered Bt', ldb = k)
    int32_t, cutlass::layout::RowMajor,          // C  (never materialized)
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TBShape, WarpShape, InstShape,
    // satfinite variant: the only s8 16x8x32 arch::Mma specialization in CUTLASS
    // (plain OpMultiplyAdd is not specialized for this shape). Identical math
    // for us: |acc| <= k*64*64 = 1.7e7 << 2^31, saturation can never trigger.
    kStages, cutlass::arch::OpMultiplyAddSaturate>;
using Mma = typename DefaultMmaT::ThreadblockMma;

// fold geometry is specialized to the real config jackpot tile (h=8, w=16)
constexpr int BM    = TBShape::kM;     // 128
constexpr int BN    = TBShape::kN;     // 256
constexpr int H     = 8,  W = 16;
constexpr int RTOFF = BM / H;          // jackpot-tile rows per block: 16
constexpr int CTOFF = BN / W;          // jackpot-tile cols per block: 16
constexpr int NJT   = RTOFF * CTOFF;   // 256
constexpr int TPB   = 32 * Mma::WarpCount::kM * Mma::WarpCount::kN * Mma::WarpCount::kK;

static_assert(Mma::WarpCount::kM == 2 && Mma::WarpCount::kN == 4 && Mma::WarpCount::kK == 1,
              "fold mapping assumes a 2x4 warp grid (TB 128x256 / warp 64x64)");
static_assert(TPB == 256, "kernel is written for 256 threads");

// ---- blake3 jackpot hash + bound check (unchanged from tc_deep_pipeline) ----
static __device__ __forceinline__ uint32_t rotr32(uint32_t x,int n){ return (x>>n)|(x<<(32-n)); }
static __device__ __forceinline__ uint32_t rotl32d(uint32_t x,int n){ return (x<<n)|(x>>(32-n)); }
static __constant__ uint32_t IVc[8] = {
  0x6A09E667u,0xBB67AE85u,0x3C6EF372u,0xA54FF53Au,
  0x510E527Fu,0x9B05688Cu,0x1F83D9ABu,0x5BE0CD19u};
static __constant__ unsigned char MS[7][16] = {
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
  {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
  {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
  {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
  {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
  {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
  {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}};
static __device__ void jackpot_blake3(const uint32_t key[8], const uint32_t msg[16], uint32_t out[8]){
  uint32_t v[16];
  for(int i=0;i<8;i++) v[i]=key[i];
  v[8]=IVc[0];v[9]=IVc[1];v[10]=IVc[2];v[11]=IVc[3];
  v[12]=0;v[13]=0;v[14]=64;v[15]=27;
  for(int r=0;r<7;r++){
    const unsigned char* s=MS[r];
    #define MIX(a,b,c,d,x,y) \
      v[a]+=v[b]+(x); v[d]=rotr32(v[d]^v[a],16); v[c]+=v[d]; v[b]=rotr32(v[b]^v[c],12); \
      v[a]+=v[b]+(y); v[d]=rotr32(v[d]^v[a],8);  v[c]+=v[d]; v[b]=rotr32(v[b]^v[c],7);
    MIX(0,4, 8,12, msg[s[0]],  msg[s[1]]);
    MIX(1,5, 9,13, msg[s[2]],  msg[s[3]]);
    MIX(2,6,10,14, msg[s[4]],  msg[s[5]]);
    MIX(3,7,11,15, msg[s[6]],  msg[s[7]]);
    MIX(0,5,10,15, msg[s[8]],  msg[s[9]]);
    MIX(1,6,11,12, msg[s[10]], msg[s[11]]);
    MIX(2,7, 8,13, msg[s[12]], msg[s[13]]);
    MIX(3,4, 9,14, msg[s[14]], msg[s[15]]);
    #undef MIX
  }
  for(int i=0;i<8;i++) out[i]=v[i]^v[i+8];
}
static __device__ __forceinline__ bool le_u256(const uint32_t a[8], const uint32_t b[8]){
  for(int i=7;i>=0;i--){ if(a[i]!=b[i]) return a[i]<b[i]; }
  return true;
}

// --- GATHER: build contiguous A'/Bt' from scattered (offset+pattern) indices ---
static __global__ void gather_rows(const signed char* __restrict__ src, signed char* __restrict__ dst,
                                   int k, const int* __restrict__ off, const int* __restrict__ pat,
                                   int h, int noff)
{
  int rprime = blockIdx.x;
  int i = rprime / h, u = rprime % h;
  if (i >= noff) return;
  size_t s = (size_t)(off[i] + pat[u]) * k;
  size_t d = (size_t)rprime * k;
  for (int l = threadIdx.x; l < k; l += blockDim.x) dst[d+l] = src[s+l];
}

// ---- the kernel: CUTLASS mainloop per rank-chunk + register fold ------------
__global__ void __launch_bounds__(TPB, 1) tc_cutlass_jackpot(
    const int8_t* __restrict__ Ap, const int8_t* __restrict__ Btp,
    int k, int rank, int nrow_off, int ncol_off,
    const uint32_t* __restrict__ key, const uint32_t* __restrict__ bound,
    int* __restrict__ win_flag, int* __restrict__ win_rt, int* __restrict__ win_ct)
{
  extern __shared__ __align__(16) char smem_raw[];
  auto* mma_smem = reinterpret_cast<typename Mma::SharedStorage*>(smem_raw);
  uint32_t* jp_sh = reinterpret_cast<uint32_t*>(smem_raw + sizeof(typename Mma::SharedStorage));
#define JPS(t,q) jp_sh[(t)*16+(q)]

  const int thread_idx = threadIdx.x;
  const int warp_idx   = threadIdx.x >> 5;
  const int lane       = threadIdx.x & 31;
  // CUTLASS MmaBase warp-raking convention (mma_base.h): m fastest, then n.
  const int warp_m = warp_idx % Mma::WarpCount::kM;
  const int warp_n = warp_idx / Mma::WarpCount::kM;

  const int bi = blockIdx.y * RTOFF;      // first jackpot-tile row of this block
  const int bj = blockIdx.x * CTOFF;      // first jackpot-tile col of this block
  const int M  = nrow_off * H;            // gathered A' rows
  const int N  = ncol_off * W;            // gathered Bt' rows (= B cols)

  for (int t = thread_idx; t < NJT*16; t += blockDim.x) jp_sh[t] = 0;
  __syncthreads();

  typename Mma::FragmentC accum;
  accum.clear();                          // cumulative across ALL chunks (spec)

  // iterator params are layout-only (stride k); same for every chunk
  // (braces, not parens: Params p(Layout(k)) is C++'s most vexing parse)
  typename Mma::IteratorA::Params paramsA{cutlass::layout::RowMajor(k)};
  typename Mma::IteratorB::Params paramsB{cutlass::layout::ColumnMajor(k)};

  const int nchunks = (k - (k % rank)) / rank;
  const int kiters  = rank / Mma::Shape::kK;     // mainloop iterations per chunk

  for (int c = 0; c < nchunks; ++c) {
    // K-window [c*rank, (c+1)*rank): shift the base pointer along k, extent=rank.
    // A row-major (row,kk)->row*k+kk and B col-major (kk,col)->col*k+kk, so the
    // window shift is +c*rank elements for both.
    typename Mma::IteratorA itA(paramsA, const_cast<int8_t*>(Ap) + (size_t)c*rank,
                                {M, rank}, thread_idx,
                                cutlass::MatrixCoord((int)blockIdx.y * BM, 0));
    typename Mma::IteratorB itB(paramsB, const_cast<int8_t*>(Btp) + (size_t)c*rank,
                                {rank, N}, thread_idx,
                                cutlass::MatrixCoord(0, (int)blockIdx.x * BN));

    Mma mma(*mma_smem, thread_idx, warp_idx, lane);
    mma(kiters, accum, itA, itB, accum);

    // drain any in-flight cp.async before the next chunk reuses the smem stages,
    // and make the fold's JPS writes safely ordered between chunks
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    // ---- fold: one XOR word per 8x16 jackpot tile per chunk (registers only)
    const int32_t* f = reinterpret_cast<const int32_t*>(accum.data());
    #pragma unroll
    for (int jr = 0; jr < 8; ++jr) {            // 8-row bands within warp's 64 rows
      const int m = jr >> 1, half = jr & 1;
      const int e0 = half*2, e1 = half*2 + 1;
      #pragma unroll
      for (int jc = 0; jc < 4; ++jc) {          // 16-col bands within warp's 64 cols
        const int n0 = jc*2, n1 = n0 + 1;
        uint32_t x = (uint32_t)f[(m + n0*4)*4 + e0] ^ (uint32_t)f[(m + n0*4)*4 + e1]
                   ^ (uint32_t)f[(m + n1*4)*4 + e0] ^ (uint32_t)f[(m + n1*4)*4 + e1];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
        if (lane == 0) {
          const int jtrib = warp_m*8 + jr;      // 0..RTOFF-1
          const int jtcib = warp_n*4 + jc;      // 0..CTOFF-1
          if (bi + jtrib < nrow_off && bj + jtcib < ncol_off) {
            const int local_jt = jtrib*CTOFF + jtcib;
            JPS(local_jt, c % 16) = rotl32d(JPS(local_jt, c % 16), 13) ^ x;
          }
        }
      }
    }
    __syncthreads();
  }

  // ---- jackpot: blake3 every tile's transcript, flag a winner ---------------
  for (int t = thread_idx; t < NJT; t += blockDim.x) {
    int jt_i = bi + t / CTOFF, jt_j = bj + t % CTOFF;
    if (jt_i >= nrow_off || jt_j >= ncol_off) continue;
    uint32_t jp[16];
    for (int q = 0; q < 16; q++) jp[q] = JPS(t, q);
    uint32_t out[8];
    jackpot_blake3(key, jp, out);
    if (le_u256(out, bound)) {
      if (atomicCAS(win_flag, 0, 1) == 0) { *win_rt = jt_i; *win_ct = jt_j; }
    }
  }
}

// ---- host wrapper: identical ABI/flow to tc_deep_pipeline -------------------
static inline void words_from_le32(const unsigned char* b, uint32_t w[8]){
  for(int i=0;i<8;i++)
    w[i]=(uint32_t)b[i*4]|((uint32_t)b[i*4+1]<<8)|((uint32_t)b[i*4+2]<<16)|((uint32_t)b[i*4+3]<<24);
}

struct DevBufs {
  signed char *dA=nullptr,*dBt=nullptr,*dAp=nullptr,*dBtp=nullptr;
  int *dpr=nullptr,*dpc=nullptr,*droff=nullptr,*dcoff=nullptr;
  uint32_t *dk=nullptr,*db=nullptr;
  int *df=nullptr,*dr=nullptr,*dc=nullptr;
  bool ok=false;
};
static DevBufs g_bufs;

static bool ensure_dev_bufs(int m, int n, int k, int h, int w, int nrow_off, int ncol_off) {
  if (!g_bufs.ok) {
    size_t apk = (size_t)nrow_off*h*k, bpk = (size_t)ncol_off*w*k;
    if (cudaMalloc(&g_bufs.dA,(size_t)m*k) || cudaMalloc(&g_bufs.dBt,(size_t)n*k) ||
        cudaMalloc(&g_bufs.dAp,apk) || cudaMalloc(&g_bufs.dBtp,bpk) ||
        cudaMalloc(&g_bufs.dpr,h*4) || cudaMalloc(&g_bufs.dpc,w*4) ||
        cudaMalloc(&g_bufs.droff,nrow_off*4) || cudaMalloc(&g_bufs.dcoff,ncol_off*4) ||
        cudaMalloc(&g_bufs.dk,32) || cudaMalloc(&g_bufs.db,32) ||
        cudaMalloc(&g_bufs.df,4) || cudaMalloc(&g_bufs.dr,4) || cudaMalloc(&g_bufs.dc,4)) {
      fprintf(stderr,"tc_cutlass: persistent malloc fail\n");
      g_bufs.ok=false; return false;
    }
    g_bufs.ok=true;
    fprintf(stderr,"tc_cutlass: persistent device buffers allocated (%zu MB)\n",
            ((size_t)m*k+(size_t)n*k+apk+bpk)/1024/1024);
  }
  return true;
}

extern "C" int tc_jackpot_search(
    const signed char* a_noised, const signed char* b_noised_t,
    int m,int n,int k,int rank,
    const int* pat_rows, const int* pat_cols, int h,int w,
    const int* row_off, int nrow_off, const int* col_off, int ncol_off,
    const unsigned char* a_noise_seed32, const unsigned char* bound_le32,
    unsigned int* /*out_hashes_host*/, unsigned int* /*dbg*/,
    int* out_rt, int* out_ct)
{
  if (rank % Mma::Shape::kK) { fprintf(stderr,"tc_cutlass: rank %d not a multiple of %d\n", rank, (int)Mma::Shape::kK); return -3; }
  // The register-fold geometry is specialized to the real config tile shape.
  if (h != H || w != W) {
    fprintf(stderr,"tc_cutlass: this geometry needs h=%d w=%d (got h=%d w=%d)\n",H,W,h,w); return -2; }

  if (!ensure_dev_bufs(m,n,k,h,w,nrow_off,ncol_off)) return -1;
  DevBufs& B = g_bufs;

  size_t ntiles = (size_t)nrow_off * ncol_off;
  cudaMemcpy(B.dA,a_noised,(size_t)m*k,cudaMemcpyHostToDevice);
  cudaMemcpy(B.dBt,b_noised_t,(size_t)n*k,cudaMemcpyHostToDevice);
  cudaMemcpy(B.dpr,pat_rows,h*4,cudaMemcpyHostToDevice);
  cudaMemcpy(B.dpc,pat_cols,w*4,cudaMemcpyHostToDevice);
  cudaMemcpy(B.droff,row_off,nrow_off*4,cudaMemcpyHostToDevice);
  cudaMemcpy(B.dcoff,col_off,ncol_off*4,cudaMemcpyHostToDevice);
  uint32_t kw[8],bw[8]; words_from_le32(a_noise_seed32,kw); words_from_le32(bound_le32,bw);
  cudaMemcpy(B.dk,kw,32,cudaMemcpyHostToDevice);
  cudaMemcpy(B.db,bw,32,cudaMemcpyHostToDevice);
  cudaMemset(B.df,0,4);

  gather_rows<<<nrow_off*h, 256>>>(B.dA, B.dAp, k, B.droff, B.dpr, h, nrow_off);
  gather_rows<<<ncol_off*w, 256>>>(B.dBt, B.dBtp, k, B.dcoff, B.dpc, w, ncol_off);

  dim3 grid((ncol_off + CTOFF - 1)/CTOFF, (nrow_off + RTOFF - 1)/RTOFF);
  size_t smem_bytes = sizeof(typename Mma::SharedStorage) + (size_t)NJT*16*4;
  static bool attr_set=false;
  if (!attr_set) {
    cudaError_t ae = cudaFuncSetAttribute(tc_cutlass_jackpot,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    if (ae!=cudaSuccess) fprintf(stderr,"tc_cutlass: smem attr (%zu B) err %s\n", smem_bytes, cudaGetErrorString(ae));
    attr_set=true;
  }
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  tc_cutlass_jackpot<<<grid, TPB, smem_bytes>>>(
      (const int8_t*)B.dAp,(const int8_t*)B.dBtp,k,rank,nrow_off,ncol_off,B.dk,B.db,B.df,B.dr,B.dc);
  cudaError_t le = cudaGetLastError();
  if (le!=cudaSuccess) fprintf(stderr,"tc_cutlass: LAUNCH err %s (tpb=%d, grid=%dx%d, smem=%zuB)\n",
      cudaGetErrorString(le), TPB, grid.x, grid.y, smem_bytes);
  cudaError_t err=cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  if (err==cudaSuccess) err=le;
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  double work_hashes=(double)ntiles*h*w*(k-(k%rank));
  fprintf(stderr,
          "tc(cutlass): TB=%dx%dx%d W=64x64 s%d %zu tiles, %.3f ms, %.2f TH/s\n",
          BM, BN, (int)TBShape::kK, kStages, ntiles, ms, work_hashes / (ms * 1e-3) / 1e12);
  if (err!=cudaSuccess) fprintf(stderr,"tc_cutlass: err %s\n",cudaGetErrorString(err));

  int wf=0;
  if (err==cudaSuccess){
    cudaMemcpy(&wf,B.df,4,cudaMemcpyDeviceToHost);
    if (wf){ cudaMemcpy(out_rt,B.dr,4,cudaMemcpyDeviceToHost); cudaMemcpy(out_ct,B.dc,4,cudaMemcpyDeviceToHost); }
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (err==cudaSuccess) ? wf : -1;
}
