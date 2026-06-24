// tc_block.cu — M3: BLOCKED int8 tensor-core GEMM with fused jackpot fold +
// cp.async MULTI-STAGE (STAGES-deep) global->shared pipeline.
//
// WHY (the 123x gap): tc_gemm = one warp per tile -> every one of 134M tiles
// re-reads its inputs from GLOBAL = ~13 TB/draw -> memory-bound, ~0.5% util, 95s.
// Compute floor ~0.5s. lpminer (0.77s) does a shared-memory-BLOCKED GEMM.
//
// THIS KERNEL:
//   (1) GATHER pre-pass: row_off[i]+pat[u] is a bijection onto [0,m) -> build
//       contiguous A'/Bt' once; the GEMM is a clean C'=A'*Bt'^T whose contiguous
//       h x w blocks ARE the jackpot tiles.
//   (2) BLOCKED + WARP-REGISTER-TILED + MULTI-STAGE: threadblock computes
//       BM x BN (128x128). 8 warps; each owns a 2x4 grid of 16x16 WMMA subtiles
//       (reuses 6 shared loads for 8 MMAs). cp.async keeps STAGES-1 k-slice
//       prefetches in flight across STAGES rotating shared buffers, hiding global
//       latency. Accumulators persist; per rank-chunk the h x w cells of each
//       contained jackpot tile are XOR-folded (rotl13 chain) — same math as the
//       verified tc_gemm/CPU reference.
//   (3) PERSISTENT DEVICE BUFFERS: allocated once on first call, reused across
//       draws. Avoids ~2.5 GB cudaMalloc/cudaFree per call.
//
// WMMA orientation copied verbatim from verified tc_gemm.cu.
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

#define BM 128
#define BN 128
#define SM (BM/16)        // 8 subtiles down
#define SN (BN/16)        // 8 subtiles across
#define WGM 4             // warp grid rows
#define WGN 2             // warp grid cols
#define WPB (WGM*WGN)     // 8 warps / 256 threads (best: WSUBN=4 maximises B-frag reuse,
                          // 6 shared loads feed 8 MMAs; 512-thr/2x2 was slower at 26.7)
#define WSUBM (SM/WGM)    // 2 subtiles per warp (down)
#define WSUBN (SN/WGN)    // 4 subtiles per warp (across)
#define RSUB 32           // k staging width
#define STAGES 2          // cp.async pipeline depth; STAGES=2 -> 32KB smem -> 3 blocks/SM on sm_86
#define MAXR 256
#define MAXJT 128         // (BM/h)*(BN/w) real cfg = 16*8 = 128

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
__global__ void gather_rows(const signed char* __restrict__ src, signed char* __restrict__ dst,
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

__global__ void __launch_bounds__(WPB*32) tc_block_jackpot(
    const signed char* __restrict__ Ap,  const signed char* __restrict__ Btp,
    int k, int rank, int h, int w,
    int nrow_off, int ncol_off,
    const uint32_t* __restrict__ key, const uint32_t* __restrict__ bound,
    int* __restrict__ win_flag, int* __restrict__ win_rt, int* __restrict__ win_ct)
{
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wgr = warp / WGN, wgc = warp % WGN;

  const int RToff = BM / h;
  const int CToff = BN / w;
  const int bi = blockIdx.y * RToff;
  const int bj = blockIdx.x * CToff;
  const int jtr_per_sub = 16 / h;
  const int jtc_per_sub = 16 / w;
  const int njt = RToff * CToff;

  __shared__ __align__(16) signed char a_sh[STAGES][BM][RSUB];
  __shared__ __align__(16) signed char b_sh[STAGES][BN][RSUB];
  __shared__ int32_t  o_sh[WPB][16][16];
  __shared__ uint32_t jp_sh[MAXJT][16];

  for (int t = threadIdx.x; t < njt*16; t += blockDim.x) jp_sh[t/16][t%16] = 0;

  wmma::fragment<wmma::accumulator,16,16,16,int32_t> acc[WSUBM][WSUBN];
  for (int a = 0; a < WSUBM; a++) for (int b = 0; b < WSUBN; b++) wmma::fill_fragment(acc[a][b], 0);

  const int nchunks = (k - (k % rank)) / rank;
  const int nsub    = rank / RSUB;
  const int ks_sub  = RSUB / 16;
  const int nstages = nchunks * nsub;

  // cp.async copy of one RSUB-wide k-slice into a shared buffer. Strided over all
  // threads (works for any blockDim): BM*(RSUB/16) 16B segments for A, same for B.
#define LOAD_SLICE(buf, kbexpr) do { \
    int _kb = (kbexpr); \
    for (int e = threadIdx.x; e < BM*(RSUB/16); e += blockDim.x) { \
      int r = e/(RSUB/16), o = (e%(RSUB/16))*16; \
      __pipeline_memcpy_async(&a_sh[buf][r][o], &Ap[(size_t)(bi*h+r)*k + _kb + o], 16); } \
    for (int e = threadIdx.x; e < BN*(RSUB/16); e += blockDim.x) { \
      int cc = e/(RSUB/16), o = (e%(RSUB/16))*16; \
      __pipeline_memcpy_async(&b_sh[buf][cc][o], &Btp[(size_t)(bj*w+cc)*k + _kb + o], 16); } \
    __pipeline_commit(); \
  } while(0)

  // prologue: kick off STAGES-1 async stages (0..STAGES-2) so the steady state
  // always has STAGES-1 cp.async groups in flight to hide global latency.
  #pragma unroll
  for (int s = 0; s < STAGES-1; s++) {
    if (s < nstages) LOAD_SLICE(s % STAGES, s*RSUB);
    else __pipeline_commit();
  }

  for (int st = 0; st < nstages; st++) {
    const int buf = st % STAGES;
    const int pf  = st + (STAGES-1);          // stage to prefetch this iteration
    if (pf < nstages) LOAD_SLICE(pf % STAGES, pf*RSUB);
    else __pipeline_commit();                 // empty group keeps wait_prior arithmetic uniform
    __pipeline_wait_prior(STAGES-1);          // <= STAGES-1 in flight => buf `st` has arrived
    __syncthreads();

    for (int ks = 0; ks < ks_sub; ks++) {
      wmma::fragment<wmma::matrix_a,16,16,16, signed char, wmma::row_major> af[WSUBN];
      wmma::fragment<wmma::matrix_b,16,16,16, signed char, wmma::col_major> bf[WSUBM];
      for (int tc = 0; tc < WSUBN; tc++)
        wmma::load_matrix_sync(af[tc], &b_sh[buf][(wgc*WSUBN+tc)*16][ks*16], RSUB);
      for (int tr = 0; tr < WSUBM; tr++)
        wmma::load_matrix_sync(bf[tr], &a_sh[buf][(wgr*WSUBM+tr)*16][ks*16], RSUB);
      for (int tr = 0; tr < WSUBM; tr++)
        for (int tc = 0; tc < WSUBN; tc++)
          wmma::mma_sync(acc[tr][tc], af[tc], bf[tr], acc[tr][tc]);
    }

    // fold after every full rank-chunk
    if ((st+1) % nsub == 0) {
      const int c = st / nsub;
      for (int tr = 0; tr < WSUBM; tr++) {
        for (int tc = 0; tc < WSUBN; tc++) {
          wmma::store_matrix_sync(&o_sh[warp][0][0], acc[tr][tc], 16, wmma::mem_row_major);
          __syncwarp();
          const int sgr = wgr*WSUBM + tr, sgc = wgc*WSUBN + tc;
          for (int sr = 0; sr < jtr_per_sub; sr++) {
            for (int sc = 0; sc < jtc_per_sub; sc++) {
              int jt_i = bi + sgr*jtr_per_sub + sr;
              int jt_j = bj + sgc*jtc_per_sub + sc;
              if (jt_i >= nrow_off || jt_j >= ncol_off) continue;
              uint32_t x = 0;
              for (int idx = lane; idx < h*w; idx += 32) {
                int rr = idx / w, ccc = idx % w;
                x ^= (uint32_t)o_sh[warp][sc*w + ccc][sr*h + rr];
              }
              for (int off = 16; off > 0; off >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, off);
              if (lane == 0) {
                int local_jt = (sgr*jtr_per_sub + sr)*CToff + (sgc*jtc_per_sub + sc);
                jp_sh[local_jt][c % 16] = rotl32d(jp_sh[local_jt][c % 16], 13) ^ x;
              }
            }
          }
          __syncwarp();
        }
      }
    }
    __syncthreads();   // current buffer fully consumed before it is re-prefetched
  }

  for (int t = threadIdx.x; t < njt; t += blockDim.x) {
    int jt_i = bi + t / CToff, jt_j = bj + t % CToff;
    if (jt_i >= nrow_off || jt_j >= ncol_off) continue;
    uint32_t jp[16];
    for (int q = 0; q < 16; q++) jp[q] = jp_sh[t][q];
    uint32_t out[8];
    jackpot_blake3(key, jp, out);
    if (le_u256(out, bound)) {
      if (atomicCAS(win_flag, 0, 1) == 0) { *win_rt = jt_i; *win_ct = jt_j; }
    }
  }
}

static inline void words_from_le32(const unsigned char* b, uint32_t w[8]){
  for(int i=0;i<8;i++)
    w[i]=(uint32_t)b[i*4]|((uint32_t)b[i*4+1]<<8)|((uint32_t)b[i*4+2]<<16)|((uint32_t)b[i*4+3]<<24);
}

// Persistent device buffers: allocated once on first call, reused across draws.
// Avoids ~2.5 GB cudaMalloc/cudaFree per call (saves ~100-200ms/draw and prevents
// fragmentation). Sizes are capped at the real config (m=n=131072, k=4096, h=8, w=16).
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
      fprintf(stderr,"tc_block: persistent malloc fail\n");
      g_bufs.ok=false; return false;
    }
    g_bufs.ok=true;
    fprintf(stderr,"tc_block: persistent device buffers allocated (%zu MB)\n",
            ((size_t)m*k+(size_t)n*k+apk+bpk)/1024/1024);
  }
  return true;
}

// Persistent-buffer accessor so the GPU-resident draw pipeline (gpu_prep.cu) can
// write the noised matrices straight into THIS kernel's dA/dBt and then call
// tc_jackpot_search(NULL,NULL,...) — the same drop-in contract tc_cutlass_v2
// exposes. Without this symbol linked, plainproof_gen keeps the CPU producer path.
extern "C" int tc_alloc_bufs(int m,int n,int k,int h,int w,int nrow_off,int ncol_off,
                             signed char** dA, signed char** dBt)
{
  if (!ensure_dev_bufs(m,n,k,h,w,nrow_off,ncol_off)) return -1;
  *dA = g_bufs.dA; *dBt = g_bufs.dBt;
  return 0;
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
  if (rank > MAXR || (rank % RSUB)) { fprintf(stderr,"tc_block: bad rank %d\n", rank); return -3; }
  if ((16 % h) || (16 % w) || (BM % h) || (BN % w) ||
      (size_t)(BM/h)*(BN/w) > MAXJT) {
    fprintf(stderr,"tc_block: h=%d w=%d unsupported by this block geometry\n",h,w); return -2; }

  if (!ensure_dev_bufs(m,n,k,h,w,nrow_off,ncol_off)) return -1;
  DevBufs& B = g_bufs;

  size_t ntiles = (size_t)nrow_off * ncol_off;
  // a_noised==NULL => the noised matrices were produced GPU-side directly in
  // B.dA/B.dBt by gpu_prep (the GPU-resident draw pipeline), so skip the H2D.
  if (a_noised)   cudaMemcpy(B.dA,a_noised,(size_t)m*k,cudaMemcpyHostToDevice);
  if (b_noised_t) cudaMemcpy(B.dBt,b_noised_t,(size_t)n*k,cudaMemcpyHostToDevice);

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

  dim3 grid((ncol_off + (BN/w) - 1)/(BN/w), (nrow_off + (BM/h) - 1)/(BM/h));
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  tc_block_jackpot<<<grid, WPB*32>>>(B.dAp,B.dBtp,k,rank,h,w,nrow_off,ncol_off,B.dk,B.db,B.df,B.dr,B.dc);
  cudaError_t le = cudaGetLastError();
  if (le!=cudaSuccess) fprintf(stderr,"tc_block: LAUNCH err %s (tpb=%d, grid=%dx%d)\n",
      cudaGetErrorString(le), WPB*32, grid.x, grid.y);
  cudaError_t err=cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  if (err==cudaSuccess) err=le;
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  // SRBMiner-MULTI, lpminer, and pools display this PearlHash/PoUW work-rate as
  // TH/s. Numerically this is the closed-source-compatible PRL work counter:
  // tiles * h * w * dot_len / second / 1e12.
  double work_hashes=(double)ntiles*h*w*(k-(k%rank));
  fprintf(stderr,
          "tc(block): BM=%d BN=%d RSUB=%d %zu tiles, %.3f ms, %.2f TH/s\n",
          BM, BN, RSUB, ntiles, ms, work_hashes / (ms * 1e-3) / 1e12);
  if (err!=cudaSuccess) fprintf(stderr,"tc_block: err %s\n",cudaGetErrorString(err));

  int wf=0;
  if (err==cudaSuccess){
    cudaMemcpy(&wf,B.df,4,cudaMemcpyDeviceToHost);
    if (wf){ cudaMemcpy(out_rt,B.dr,4,cudaMemcpyDeviceToHost); cudaMemcpy(out_ct,B.dc,4,cudaMemcpyDeviceToHost); }
  }
  // DO NOT free — reuse across draws. Buffers live until process exit.
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (err==cudaSuccess) ? wf : -1;
}
