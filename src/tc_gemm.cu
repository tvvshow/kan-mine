// tc_gemm.cu — M1b tensor-core compute path for Pearl jackpot search.
// Dense int8 tensor-core GEMM (CUTLASS legacy Sm80 tensorop, compiled -arch=sm_120a),
// chunked over k in rank-sized steps with beta=1 accumulation so C holds the
// CUMULATIVE partial sum after each rank-chunk — exactly what the jackpot fold needs.
// After each chunk, fold all tiles; after all chunks, keyed-blake3 + LE<=bound win check.
// Bit-exact to the dp4a reference because int7 base + bounded noise fits int8 [-128,126].
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"

using Gemm = cutlass::gemm::device::Gemm<
    int8_t,  cutlass::layout::RowMajor,
    int8_t,  cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80>;

// --- BLAKE3 single-block keyed (root XOF first 8 words) — identical to jackpot_kernel.cu ---
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

// Fold one rank-chunk into the per-tile jackpot lanes.
// One block per tile, h*w threads. Reads CUMULATIVE C (m x n int32, row-major).
__global__ void tc_fold_kernel(
    const int32_t* __restrict__ C, int n,
    const int* __restrict__ row_off, const int* __restrict__ col_off,
    const int* __restrict__ pat, int h, int w, int ncol_off,
    int chunk_idx, uint32_t* __restrict__ jackpot /* ntiles*16 */)
{
  int tile = blockIdx.x;
  int i = tile / ncol_off, j = tile % ncol_off;
  int hw = h*w;
  int cell = threadIdx.x;
  __shared__ uint32_t sh[256];
  if (cell < hw) {
    int u = cell / w, vv = cell % w;
    size_t a_idx = (size_t)(row_off[i] + pat[u]);
    int    b_idx = col_off[j] + pat[vv];
    sh[cell] = (uint32_t)C[a_idx*(size_t)n + b_idx];
  }
  __syncthreads();
  if (cell == 0) {
    uint32_t x = 0;
    for (int t = 0; t < hw; t++) x ^= sh[t];
    int tid = chunk_idx % 16;
    size_t base = (size_t)tile*16 + tid;
    jackpot[base] = rotl32d(jackpot[base], 13) ^ x;
  }
}

// One thread per tile: keyed-blake3(jackpot[16]) then LE compare <= bound.
__global__ void tc_hash_kernel(
    const uint32_t* __restrict__ jackpot, int ntiles, int ncol_off,
    const uint32_t* __restrict__ key, const uint32_t* __restrict__ bound,
    int* __restrict__ win_flag, int* __restrict__ win_rt, int* __restrict__ win_ct)
{
  int tile = blockIdx.x*blockDim.x + threadIdx.x;
  if (tile >= ntiles) return;
  uint32_t msg[16];
  for (int i=0;i<16;i++) msg[i] = jackpot[(size_t)tile*16 + i];
  uint32_t out[8];
  jackpot_blake3(key, msg, out);
  if (le_u256(out, bound)) {
    if (atomicCAS(win_flag, 0, 1) == 0) { *win_rt = tile/ncol_off; *win_ct = tile%ncol_off; }
  }
}

static inline void words_from_le32(const unsigned char* b, uint32_t w[8]){
  for(int i=0;i<8;i++)
    w[i]=(uint32_t)b[i*4]|((uint32_t)b[i*4+1]<<8)|((uint32_t)b[i*4+2]<<16)|((uint32_t)b[i*4+3]<<24);
}

extern "C" int tc_jackpot_search(
    const signed char* a_noised, const signed char* b_noised_t,
    int m,int n,int k,int rank,
    const int* pat, int h,int w,
    const int* row_off, int nrow_off, const int* col_off, int ncol_off,
    const unsigned char* a_noise_seed32, const unsigned char* bound_le32,
    unsigned int* /*out_hashes_host*/, unsigned int* /*dbg_j0_host*/,
    int* out_rt, int* out_ct)
{
  size_t ntiles = (size_t)nrow_off * ncol_off;
  int dot_len = k - (k % rank);
  int nchunks = dot_len / rank;

  int8_t  *dA=nullptr,*dBt=nullptr; int32_t *dC=nullptr;
  cudaMalloc(&dA,(size_t)m*k); cudaMalloc(&dBt,(size_t)n*k); cudaMalloc(&dC,(size_t)m*n*sizeof(int32_t));
  cudaMemcpy(dA,a_noised,(size_t)m*k,cudaMemcpyHostToDevice);
  cudaMemcpy(dBt,b_noised_t,(size_t)n*k,cudaMemcpyHostToDevice);
  cudaMemset(dC,0,(size_t)m*n*sizeof(int32_t));

  int *dpat=nullptr,*droff=nullptr,*dcoff=nullptr;
  int npat=(h>w?h:w);
  cudaMalloc(&dpat,npat*sizeof(int));   cudaMemcpy(dpat,pat,npat*sizeof(int),cudaMemcpyHostToDevice);
  cudaMalloc(&droff,nrow_off*sizeof(int)); cudaMemcpy(droff,row_off,nrow_off*sizeof(int),cudaMemcpyHostToDevice);
  cudaMalloc(&dcoff,ncol_off*sizeof(int)); cudaMemcpy(dcoff,col_off,ncol_off*sizeof(int),cudaMemcpyHostToDevice);

  uint32_t keyw[8],bndw[8]; words_from_le32(a_noise_seed32,keyw); words_from_le32(bound_le32,bndw);
  uint32_t *dkey=nullptr,*dbnd=nullptr; cudaMalloc(&dkey,32); cudaMalloc(&dbnd,32);
  cudaMemcpy(dkey,keyw,32,cudaMemcpyHostToDevice); cudaMemcpy(dbnd,bndw,32,cudaMemcpyHostToDevice);

  uint32_t* djp=nullptr; cudaMalloc(&djp,ntiles*16*sizeof(uint32_t)); cudaMemset(djp,0,ntiles*16*sizeof(uint32_t));
  int *dwf=nullptr,*drt=nullptr,*dct=nullptr; cudaMalloc(&dwf,4);cudaMalloc(&drt,4);cudaMalloc(&dct,4); cudaMemset(dwf,0,4);

  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);

  Gemm op;
  for (int c=0;c<nchunks;c++){
    cutlass::TensorRef<int8_t,  cutlass::layout::RowMajor>    refA((int8_t*)dA + (size_t)c*rank, cutlass::layout::RowMajor(k));
    cutlass::TensorRef<int8_t,  cutlass::layout::ColumnMajor> refB((int8_t*)dBt + (size_t)c*rank, cutlass::layout::ColumnMajor(k));
    cutlass::TensorRef<int32_t, cutlass::layout::RowMajor>    refC(dC, cutlass::layout::RowMajor(n));
    typename Gemm::Arguments args({m,n,rank}, refA, refB, refC, refC, {1,1});
    cutlass::Status st = op(args);
    if (st != cutlass::Status::kSuccess){ fprintf(stderr,"tc: gemm chunk %d status=%d\n",c,(int)st); return -2; }
    tc_fold_kernel<<<(unsigned)ntiles, (unsigned)(h*w)>>>(dC,n,droff,dcoff,dpat,h,w,ncol_off,c,djp);
  }
  int threads=128, blocks=(int)((ntiles+threads-1)/threads);
  tc_hash_kernel<<<blocks,threads>>>(djp,(int)ntiles,ncol_off,dkey,dbnd,dwf,drt,dct);
  cudaError_t err=cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  fprintf(stderr,"tc: %d chunks, %zu tiles, %.3f ms (compute+fold+hash)\n", nchunks, ntiles, ms);
  if (err!=cudaSuccess){ fprintf(stderr,"tc: cuda err %s\n",cudaGetErrorString(err)); return -1; }

  int wf=0; cudaMemcpy(&wf,dwf,4,cudaMemcpyDeviceToHost);
  if (wf){ cudaMemcpy(out_rt,drt,4,cudaMemcpyDeviceToHost); cudaMemcpy(out_ct,dct,4,cudaMemcpyDeviceToHost); }
  cudaFree(dA);cudaFree(dBt);cudaFree(dC);cudaFree(dpat);cudaFree(droff);cudaFree(dcoff);
  cudaFree(dkey);cudaFree(dbnd);cudaFree(djp);cudaFree(dwf);cudaFree(drt);cudaFree(dct);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return wf;
}
