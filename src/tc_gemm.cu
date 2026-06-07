// tc_gemm.cu — M2a FUSED int8 tensor-core jackpot search for Pearl PoUW.
//
// WHY THIS REPLACES THE OLD CUTLASS PATH: the previous version ran a device-level
// CUTLASS GEMM per rank-chunk into a full m*n int32 C matrix. At the REAL network
// config (m=n=131072) that C is 131072^2*4 = 68 GB — impossible. lpminer/official
// never materialize C; they accumulate each tile IN REGISTERS and fold the hash on
// the fly. This kernel does the same with WMMA (Ampere/Ada/Blackwell int8 tensor
// cores), so memory use is O(A)+O(Bt) = ~1 GB at real config, not O(m*n).
//
// MODEL (one WARP per tile, identical math to the dp4a reference jackpot_kernel.cu
// and the CPU reference plainproof_gen.cpp):
//   tile (i,j): rows R[u]=row_off[i]+pat_rows[u] (u<h), cols C[v]=col_off[j]+pat_cols[v] (v<w)
//   acc[M=16 cols][N=16 rows] persists across ALL rank-chunks, so after chunk c it
//   holds the CUMULATIVE dot P_c[col][row]=sum_{l<=(c+1)*rank} b_noised_t[C[col]][l]*a_noised[R[row]][l]
//   (note O[col][row]==jackpot_tile[row][col]; dot is symmetric so the XOR is identical).
//   After each chunk: x_c = XOR over the h*w REAL cells of (uint32)acc; jp[c]=rotl13(jp[c])^x_c.
//   (real config has exactly dot_len/rank=16 chunks => each jp[c] written once => jp[c]=x_c.)
//   After all chunks: out = keyed-blake3(jp[16], a_noise_seed); win if U256_LE(out) <= bound.
//
// WMMA tile is 16x16x16 s8. We map M<-w cols (pad to 16), N<-h rows (pad to 16),
// K<-16 (rank/16 wmma steps per chunk). Padding lanes are zeroed and EXCLUDED from
// the XOR (only m<w && n<h cells fold), so they never affect the result.
//
// Build: nvcc -arch=sm_80+ (no CUTLASS needed). int8 WMMA needs sm_72+.
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

// --- BLAKE3 single 64-byte block, keyed, root XOF, first 8 words (== jackpot_kernel.cu) ---
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
  v[12]=0;v[13]=0;v[14]=64;v[15]=27; // counter=0, block_len=64, flags=KEYED|CHUNK_START|CHUNK_END|ROOT
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

#define WPB 4          // warps per block
#define MAXR 256       // max rank supported (real=256, golden=128)

// One warp per tile. blockDim = WPB*32.
__global__ void tc_fused_jackpot(
    const signed char* __restrict__ A,  const signed char* __restrict__ Bt,
    int m, int n, int k, int rank,
    const int* __restrict__ pat_rows, const int* __restrict__ pat_cols, int h, int w,
    const int* __restrict__ row_off, const int* __restrict__ col_off,
    int nrow_off, int ncol_off,
    const uint32_t* __restrict__ key, const uint32_t* __restrict__ bound,
    uint32_t* __restrict__ out_hashes,            // optional: ntiles*8, may be null
    int* __restrict__ win_flag, int* __restrict__ win_rt, int* __restrict__ win_ct)
{
  const int warp = threadIdx.x >> 5;          // 0..WPB-1
  const int lane = threadIdx.x & 31;
  const size_t tile = (size_t)blockIdx.x * WPB + warp;
  const size_t ntiles = (size_t)nrow_off * ncol_off;
  if (tile >= ntiles) return;
  const int i = (int)(tile / ncol_off);       // row_off index
  const int j = (int)(tile % ncol_off);       // col_off index

  // per-warp shared staging: A-rows (N, padded 16) and B-cols (M, padded 16) for one chunk.
  // 16-byte aligned: wmma::load_matrix_sync needs aligned base addresses for int8 frags.
  __shared__ __align__(16) signed char a_sh[WPB][16][MAXR];  // a_sh[warp][row][kk]  rows 0..h-1 real
  __shared__ __align__(16) signed char b_sh[WPB][16][MAXR];  // b_sh[warp][col][kk]  cols 0..w-1 real
  __shared__ int32_t     o_sh[WPB][16][16];    // acc store, [M=col][N=row]
  __shared__ uint32_t    jp_sh[WPB][16];

  // zero the padding rows/cols once (they never get gathered, must contribute 0 / be excluded)
  for (int idx = lane; idx < 16*MAXR; idx += 32) {
    int row = idx / MAXR, kk = idx % MAXR;
    if (row >= h) a_sh[warp][row][kk] = 0;
    if (row >= w) b_sh[warp][row][kk] = 0;
  }
  if (lane < 16) jp_sh[warp][lane] = 0;
  __syncwarp();

  // resolve this tile's scattered row/col indices (h<=16, w<=16)
  int Rr[16], Cc[16];
  for (int u = 0; u < h; u++) Rr[u] = row_off[i] + pat_rows[u];
  for (int v = 0; v < w; v++) Cc[v] = col_off[j] + pat_cols[v];

  wmma::fragment<wmma::accumulator, 16,16,16, int32_t> acc;
  wmma::fill_fragment(acc, 0);

  const int nchunks = (k - (k % rank)) / rank;  // dot_len/rank
  const int ksteps  = rank / 16;                // wmma K-steps per chunk

  for (int c = 0; c < nchunks; c++) {
    const int kbase = c * rank;
    // gather this chunk's real A-rows and B-cols into shared (coalesced-ish per row)
    for (int e = lane; e < h*rank; e += 32) {
      int row = e / rank, kk = e % rank;
      a_sh[warp][row][kk] = A[(size_t)Rr[row]*k + kbase + kk];
    }
    for (int e = lane; e < w*rank; e += 32) {
      int col = e / rank, kk = e % rank;
      b_sh[warp][col][kk] = Bt[(size_t)Cc[col]*k + kbase + kk];
    }
    __syncwarp();

    // accumulate this chunk into acc (acc PERSISTS -> cumulative across chunks)
    for (int ks = 0; ks < ksteps; ks++) {
      wmma::fragment<wmma::matrix_a, 16,16,16, signed char, wmma::row_major> af; // M=col,K
      wmma::fragment<wmma::matrix_b, 16,16,16, signed char, wmma::col_major> bf; // K,N=row
      wmma::load_matrix_sync(af, &b_sh[warp][0][ks*16], MAXR); // O[M=col][.] from b-cols
      wmma::load_matrix_sync(bf, &a_sh[warp][0][ks*16], MAXR); // [.][N=row]   from a-rows
      wmma::mma_sync(acc, af, bf, acc);
    }
    // snapshot cumulative acc and XOR-fold the h*w real cells
    wmma::store_matrix_sync(&o_sh[warp][0][0], acc, 16, wmma::mem_row_major);
    __syncwarp();
    uint32_t x = 0;
    for (int idx = lane; idx < 256; idx += 32) {  // 8 cells/lane
      int M = idx >> 4, N = idx & 15;             // M=col, N=row
      if (M < w && N < h) x ^= (uint32_t)o_sh[warp][M][N];
    }
    for (int off = 16; off > 0; off >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, off);
    if (lane == 0) { int tid = c % 16; jp_sh[warp][tid] = rotl32d(jp_sh[warp][tid], 13) ^ x; }
    __syncwarp();
  }

  if (lane == 0) {
    uint32_t jp[16];
    for (int t = 0; t < 16; t++) jp[t] = jp_sh[warp][t];
    uint32_t out[8];
    jackpot_blake3(key, jp, out);
    if (out_hashes) for (int t = 0; t < 8; t++) out_hashes[tile*8 + t] = out[t];
    if (le_u256(out, bound)) {
      if (atomicCAS(win_flag, 0, 1) == 0) { *win_rt = i; *win_ct = j; }
    }
  }
}

static inline void words_from_le32(const unsigned char* b, uint32_t w[8]){
  for(int i=0;i<8;i++)
    w[i]=(uint32_t)b[i*4]|((uint32_t)b[i*4+1]<<8)|((uint32_t)b[i*4+2]<<16)|((uint32_t)b[i*4+3]<<24);
}

// Drop-in replacement for the old tc_jackpot_search, now with SEPARATE row/col
// patterns (real config: h=8 != w=16) and NO m*n C buffer.
extern "C" int tc_jackpot_search(
    const signed char* a_noised, const signed char* b_noised_t,
    int m,int n,int k,int rank,
    const int* pat_rows, const int* pat_cols, int h,int w,
    const int* row_off, int nrow_off, const int* col_off, int ncol_off,
    const unsigned char* a_noise_seed32, const unsigned char* bound_le32,
    unsigned int* out_hashes_host, unsigned int* /*dbg_j0_host*/,
    int* out_rt, int* out_ct)
{
  if (rank > MAXR) { fprintf(stderr,"tc: rank %d > MAXR %d\n", rank, MAXR); return -3; }
  size_t ntiles = (size_t)nrow_off * ncol_off;
  signed char *dA=nullptr,*dBt=nullptr;
  if (cudaMalloc(&dA,(size_t)m*k)!=cudaSuccess || cudaMalloc(&dBt,(size_t)n*k)!=cudaSuccess) {
    fprintf(stderr,"tc: cudaMalloc A/Bt failed\n"); return -1; }
  cudaMemcpy(dA,a_noised,(size_t)m*k,cudaMemcpyHostToDevice);
  cudaMemcpy(dBt,b_noised_t,(size_t)n*k,cudaMemcpyHostToDevice);

  int *dpr=nullptr,*dpc=nullptr,*droff=nullptr,*dcoff=nullptr;
  cudaMalloc(&dpr,h*sizeof(int)); cudaMemcpy(dpr,pat_rows,h*sizeof(int),cudaMemcpyHostToDevice);
  cudaMalloc(&dpc,w*sizeof(int)); cudaMemcpy(dpc,pat_cols,w*sizeof(int),cudaMemcpyHostToDevice);
  cudaMalloc(&droff,nrow_off*sizeof(int)); cudaMemcpy(droff,row_off,nrow_off*sizeof(int),cudaMemcpyHostToDevice);
  cudaMalloc(&dcoff,ncol_off*sizeof(int)); cudaMemcpy(dcoff,col_off,ncol_off*sizeof(int),cudaMemcpyHostToDevice);

  uint32_t keyw[8],bndw[8]; words_from_le32(a_noise_seed32,keyw); words_from_le32(bound_le32,bndw);
  uint32_t *dkey=nullptr,*dbnd=nullptr; cudaMalloc(&dkey,32); cudaMalloc(&dbnd,32);
  cudaMemcpy(dkey,keyw,32,cudaMemcpyHostToDevice); cudaMemcpy(dbnd,bndw,32,cudaMemcpyHostToDevice);

  uint32_t* dhash=nullptr; if (out_hashes_host) cudaMalloc(&dhash, ntiles*8*sizeof(uint32_t));
  int *dwf=nullptr,*drt=nullptr,*dct=nullptr; cudaMalloc(&dwf,4);cudaMalloc(&drt,4);cudaMalloc(&dct,4); cudaMemset(dwf,0,4);

  int blocks = (int)((ntiles + WPB - 1) / WPB);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  tc_fused_jackpot<<<blocks, WPB*32>>>(dA,dBt,m,n,k,rank,dpr,dpc,h,w,droff,dcoff,
                                       nrow_off,ncol_off,dkey,dbnd,dhash,dwf,drt,dct);
  cudaError_t err=cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  double mac = (double)ntiles*(double)h*(double)w*(double)(k-(k%rank));
  fprintf(stderr,"tc(fused): %zu tiles, %.3f ms, %.2f TMAC/s\n", ntiles, ms, mac/(ms*1e-3)/1e12);
  if (err!=cudaSuccess){ fprintf(stderr,"tc: cuda err %s\n",cudaGetErrorString(err)); }

  int wf=0;
  if (err==cudaSuccess){
    cudaMemcpy(&wf,dwf,4,cudaMemcpyDeviceToHost);
    if (wf){ cudaMemcpy(out_rt,drt,4,cudaMemcpyDeviceToHost); cudaMemcpy(out_ct,dct,4,cudaMemcpyDeviceToHost); }
    if (out_hashes_host) cudaMemcpy(out_hashes_host,dhash,ntiles*8*sizeof(uint32_t),cudaMemcpyDeviceToHost);
  }
  cudaFree(dA);cudaFree(dBt);cudaFree(dpr);cudaFree(dpc);cudaFree(droff);cudaFree(dcoff);
  cudaFree(dkey);cudaFree(dbnd);cudaFree(dwf);cudaFree(drt);cudaFree(dct);
  if (dhash) cudaFree(dhash);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (err==cudaSuccess) ? wf : -1;
}
