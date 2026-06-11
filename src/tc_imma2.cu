// tc_imma2.cu — FAST Pearl jackpot kernel: mma.sync.m16n8k32 (IMMA) + ldmatrix,
// register-only per-rank-chunk fold. Drop-in replacement for tc_jackpot_search.
//
// Built on EMPIRICALLY VALIDATED primitives (peral/bench/mma_*.cu, random-data
// CPU-checked): mma.sync.m16n8k32.row.col.s32.s8.s8.s32 layout + ldmatrix.x4/x2
// addressing + full-K accumulation. See memory reference_imma_layout_validated.
//
// Orientation: natural C[i][j] = A'[i]·Bt'[j]  (i = A' row, j = Bt' row).
// Threadblock computes BM(i) x BN(j). 8 warps (4x2). Each warp owns WSUBM x WSUBN
// mma subtiles (16x8 each) -> 32 i-rows x 64 j-cols. Accumulators stay in regs.
//
// FOLD (register-only, no o_sh): a jackpot tile is h x w = 8 i-rows x 16 j-cols.
//   - 8 i-rows = the upper (c0,c1) or lower (c2,c3) half of one 16-row M-subtile.
//   - 16 j-cols = two adjacent 8-col N-subtiles (tc, tc+1).
//   So XOR of the tile's 128 cells = warp-XOR-reduce of
//     acc[tr][tc].{off0,off1} ^ acc[tr][tc+1].{off0,off1}
//   with (off0,off1) = (0,1) upper / (2,3) lower. Math identical to tc_block.
//
// gather / blake3 / le_u256 / persistent buffers / host wrapper copied verbatim
// from the verified tc_block.cu.
//
// STATUS (2026-06-10): this file is the VALIDATED 56 TH/s BN=128 REFERENCE,
// restored after a brief in-place BN=256 edit (which measured only 22.6 TH/s:
// BN=256 at STAGES=2 forces 1 block/SM with nothing to hide latency, see
// DESIGN_speedup.md §0.4). Do NOT experiment in this file again — parameter
// exploration lives in tc_deep_pipeline.cu, where this exact geometry is
// reproduced by -DBN=128 -DSTAGES=2 -DRSUB=64 -DREG_PIPE=0 -DMINBLOCKS=2
// (the sweep's anchor config). After the sweep crowns a winner this file
// moves to archive/.
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include <cstdio>

#define BM 128
#define BN 128
#define MMA_K 32
#define SM (BM/16)        // 8 M-subtiles (16 rows each)
#define SN (BN/8)         // 16 N-subtiles (8 cols each)
#define WGM 4
#define WGN 2
#define WPB (WGM*WGN)      // 8 warps / 256 threads
#define WSUBM (SM/WGM)     // 2 M-subtiles per warp (down)
#define WSUBN (SN/WGN)     // 8 N-subtiles per warp (across)
#define RSUB 64            // k staging width (RSUB/32 mma-k per cp.async stage)
#define STAGES 2           // 40KB smem (a_sh 16KB + b_sh 16KB + jp_sh 8KB)
#define MAXR 256
#define MAXJT 128          // (BM/h)*(BN/w) = 16*8 = 128 at real cfg
#define MINBLOCKS 2        // 40KB smem x2 fits 100KB/SM -> 2 blocks/SM hide latency

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

__global__ void __launch_bounds__(WPB*32, MINBLOCKS) tc_imma2_jackpot(
    const signed char* __restrict__ Ap,  const signed char* __restrict__ Btp,
    int k, int rank, int h, int w,
    int nrow_off, int ncol_off,
    const uint32_t* __restrict__ key, const uint32_t* __restrict__ bound,
    int* __restrict__ win_flag, int* __restrict__ win_rt, int* __restrict__ win_ct)
{
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2;        // 0..7  (mma C row within subtile half)
  const int tig   = lane & 3;         // 0..3
  const int wgr = warp / WGN, wgc = warp % WGN;

  const int RToff = BM / h;           // jackpot-tile rows per block = 16
  const int CToff = BN / w;           // jackpot-tile cols per block = 8
  const int bi = blockIdx.y * RToff;
  const int bj = blockIdx.x * CToff;
  const int njt = RToff * CToff;       // 128

  // dynamic shared memory (lets RSUB exceed the 48KB static cap)
  extern __shared__ __align__(16) char smem_raw[];
  signed char* a_sh = (signed char*)smem_raw;
  signed char* b_sh = a_sh + (size_t)STAGES*BM*RSUB;
  uint32_t* jp_sh = (uint32_t*)(b_sh + (size_t)STAGES*BN*RSUB);
#define A_SH(buf,r,o)  a_sh[((size_t)(buf)*BM+(r))*RSUB+(o)]
#define B_SH(buf,cc,o) b_sh[((size_t)(buf)*BN+(cc))*RSUB+(o)]
#define JPS(t,q)       jp_sh[(t)*16+(q)]

  for (int t = threadIdx.x; t < njt*16; t += blockDim.x) JPS(t/16, t%16) = 0;

  int32_t acc[WSUBM][WSUBN][4];
  #pragma unroll
  for (int a=0;a<WSUBM;a++) for(int b=0;b<WSUBN;b++) { acc[a][b][0]=acc[a][b][1]=acc[a][b][2]=acc[a][b][3]=0; }

  const int nchunks = (k - (k % rank)) / rank;
  const int nsub    = rank / RSUB;     // stages per rank-chunk = 4
  const int nstages = nchunks * nsub;

#define LOAD_SLICE(buf, kbexpr) do { \
    int _kb = (kbexpr); \
    for (int e = threadIdx.x; e < BM*(RSUB/16); e += blockDim.x) { \
      int r = e/(RSUB/16), o = (e%(RSUB/16))*16; \
      __pipeline_memcpy_async(&A_SH(buf,r,o), &Ap[(size_t)(bi*h+r)*k + _kb + o], 16); } \
    for (int e = threadIdx.x; e < BN*(RSUB/16); e += blockDim.x) { \
      int cc = e/(RSUB/16), o = (e%(RSUB/16))*16; \
      __pipeline_memcpy_async(&B_SH(buf,cc,o), &Btp[(size_t)(bj*w+cc)*k + _kb + o], 16); } \
    __pipeline_commit(); \
  } while(0)

  #pragma unroll
  for (int s = 0; s < STAGES-1; s++) {
    if (s < nstages) LOAD_SLICE(s % STAGES, s*RSUB);
    else __pipeline_commit();
  }

  for (int st = 0; st < nstages; st++) {
    const int buf = st % STAGES;
    const int pf  = st + (STAGES-1);
    if (pf < nstages) LOAD_SLICE(pf % STAGES, pf*RSUB);
    else __pipeline_commit();
    __pipeline_wait_prior(STAGES-1);
    __syncthreads();

    // load A subtiles (ldmatrix.x4) and B subtiles (ldmatrix.x2), then mma.
    // RSUB/32 mma-k steps live in this one cp.async stage (fewer barriers).
    #pragma unroll
    for (int ks=0; ks<RSUB/32; ks++) {
      uint32_t af[WSUBM][4];
      #pragma unroll
      for (int tr=0; tr<WSUBM; tr++) {
        uint32_t addrA = __cvta_generic_to_shared(
            &A_SH(buf, (wgr*WSUBM+tr)*16 + (lane&15), ks*32 + (lane>>4)*16));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(af[tr][0]),"=r"(af[tr][1]),"=r"(af[tr][2]),"=r"(af[tr][3]) : "r"(addrA));
      }
      uint32_t bf[WSUBN][2];
      #pragma unroll
      for (int tc=0; tc<WSUBN; tc++) {
        uint32_t addrB = __cvta_generic_to_shared(
            &B_SH(buf, (wgc*WSUBN+tc)*8 + (lane&7), ks*32 + ((lane>>3)&1)*16));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(bf[tc][0]),"=r"(bf[tc][1]) : "r"(addrB));
      }
      #pragma unroll
      for (int tr=0; tr<WSUBM; tr++)
        #pragma unroll
        for (int tc=0; tc<WSUBN; tc++)
          asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(acc[tr][tc][0]),"+r"(acc[tr][tc][1]),"+r"(acc[tr][tc][2]),"+r"(acc[tr][tc][3])
            : "r"(af[tr][0]),"r"(af[tr][1]),"r"(af[tr][2]),"r"(af[tr][3]),
              "r"(bf[tc][0]),"r"(bf[tc][1]));
    }

    // fold after every full rank-chunk (register-only)
    if ((st+1) % nsub == 0) {
      const int c = st / nsub;
      #pragma unroll
      for (int tr=0; tr<WSUBM; tr++) {
        #pragma unroll
        for (int half=0; half<2; half++) {
          const int off0 = half*2, off1 = half*2+1;
          const int jtrib = (wgr*WSUBM+tr)*2 + half;     // 0..15
          #pragma unroll
          for (int jtc=0; jtc<WSUBN/2; jtc++) {
            const int tc = jtc*2;
            uint32_t x = (uint32_t)acc[tr][tc][off0] ^ (uint32_t)acc[tr][tc][off1]
                       ^ (uint32_t)acc[tr][tc+1][off0] ^ (uint32_t)acc[tr][tc+1][off1];
            #pragma unroll
            for (int o=16;o>0;o>>=1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
            if (lane==0) {
              const int jtcib = wgc*(WSUBN/2) + jtc;      // 0..7
              const int jt_i = bi + jtrib, jt_j = bj + jtcib;
              if (jt_i < nrow_off && jt_j < ncol_off) {
                int local_jt = jtrib*CToff + jtcib;
                JPS(local_jt, c % 16) = rotl32d(JPS(local_jt, c % 16), 13) ^ x;
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

  for (int t = threadIdx.x; t < njt; t += blockDim.x) {
    int jt_i = bi + t / CToff, jt_j = bj + t % CToff;
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
      fprintf(stderr,"tc_imma2: persistent malloc fail\n");
      g_bufs.ok=false; return false;
    }
    g_bufs.ok=true;
    fprintf(stderr,"tc_imma2: persistent device buffers allocated (%zu MB)\n",
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
  if (rank > MAXR || (rank % RSUB)) { fprintf(stderr,"tc_imma2: bad rank %d\n", rank); return -3; }
  // This register-fold geometry is specialised to the real config tile shape.
  if ((h != 8) || (w != 16) || (BM % h) || (BN % w) || (size_t)(BM/h)*(BN/w) > MAXJT) {
    fprintf(stderr,"tc_imma2: this geometry needs h=8 w=16 (got h=%d w=%d)\n",h,w); return -2; }

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

  dim3 grid((ncol_off + (BN/w) - 1)/(BN/w), (nrow_off + (BM/h) - 1)/(BM/h));
  size_t smem_bytes = (size_t)STAGES*BM*RSUB + (size_t)STAGES*BN*RSUB + (size_t)MAXJT*16*4;
  static bool attr_set=false;
  if (!attr_set) {
    cudaError_t ae = cudaFuncSetAttribute(tc_imma2_jackpot,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    if (ae!=cudaSuccess) fprintf(stderr,"tc_imma2: smem attr (%zu B) err %s\n", smem_bytes, cudaGetErrorString(ae));
    attr_set=true;
  }
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  tc_imma2_jackpot<<<grid, WPB*32, smem_bytes>>>(B.dAp,B.dBtp,k,rank,h,w,nrow_off,ncol_off,B.dk,B.db,B.df,B.dr,B.dc);
  cudaError_t le = cudaGetLastError();
  if (le!=cudaSuccess) fprintf(stderr,"tc_imma2: LAUNCH err %s (tpb=%d, grid=%dx%d)\n",
      cudaGetErrorString(le), WPB*32, grid.x, grid.y);
  cudaError_t err=cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  if (err==cudaSuccess) err=le;
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  double work_hashes=(double)ntiles*h*w*(k-(k%rank));
  fprintf(stderr,
          "tc(imma2): BM=%d BN=%d RSUB=%d %zu tiles, %.3f ms, %.2f TH/s\n",
          BM, BN, RSUB, ntiles, ms, work_hashes / (ms * 1e-3) / 1e12);
  if (err!=cudaSuccess) fprintf(stderr,"tc_imma2: err %s\n",cudaGetErrorString(err));

  int wf=0;
  if (err==cudaSuccess){
    cudaMemcpy(&wf,B.df,4,cudaMemcpyDeviceToHost);
    if (wf){ cudaMemcpy(out_rt,B.dr,4,cudaMemcpyDeviceToHost); cudaMemcpy(out_ct,B.dc,4,cudaMemcpyDeviceToHost); }
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (err==cudaSuccess) ? wf : -1;
}
