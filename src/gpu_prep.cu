// gpu_prep.cu — Phase2 of DESIGN_speedup.md: move the ENTIRE per-draw CPU prep
// (RNG fill, blake3 commitments, noise generation) onto the GPU.
//
// WHY (measured on the 3090 box, 2026-06-11): the fused CUTLASS kernel does a
// draw in 880 ms, but the CPU prep for each draw costs ~1490 ms
// (RNG 154 + blake3 567 + noise 480 + build 290) plus a 1 GB H2D copy. Even
// with the producer-thread overlap the steady state is max(1490, 880) =
// 1490 ms/draw => only ~52-59 TH/s wall-clock (pool-side it shows even lower).
// lpminer generates noise on-GPU and never moves matrices over PCIe.
//
// This file implements the same pipeline byte-exactly on the GPU:
//   phase1: splitmix64 RNG fill of A[m*k] / Bt[n*k] in device memory
//           + full BLAKE3 keyed tree hash of both (512 MiB each = 2^19 chunks,
//           perfectly balanced binary tree) -> hash_a / hash_b (D2H 64 B)
//   (host:  derive a/b_noise_seed via 2 tiny blake3 calls + the permutation
//           matrices via 1024 tiny blake3 calls — sub-millisecond)
//   phase2: per-row uniform noise (8 keyed single-block blake3 per row, the
//           SAME compression our validated jackpot_blake3 uses: t=0 b=64
//           flags=27) + sparse-perm matvec + in-place add into A / Bt.
//
// After phase2 the noised matrices are already in the tc kernel's persistent
// dA/dBt buffers, so tc_jackpot_search is called with a_noised=NULL and skips
// its two 512 MB H2D copies entirely.
//
// CORRECTNESS GATE: the existing loosened-target POSTCHECK re-derives the
// whole draw on the CPU and independently recomputes the winning tile's
// jackpot. Any GPU/CPU mismatch in RNG, tree hash, seeds or noise changes the
// jackpot transcript and fails ok=1. POSTCHECK ok=1 == full-pipeline
// equivalence proof.
//
// Build: nvcc -O3 -arch=sm_86 -std=c++17 -c src/gpu_prep.cu -o build/gpu_prep.o
#include <cuda_runtime.h>
#include "platform.h"   // KAN_WEAK / KAN_NO_ASYNC_SEARCH
#include <cstdint>
#include <cstdio>

// ---------------- BLAKE3 device compression (full, parameterized) ------------
static __device__ __forceinline__ uint32_t rotr32g(uint32_t x,int n){ return (x>>n)|(x<<(32-n)); }
static __constant__ uint32_t IVg[8] = {
  0x6A09E667u,0xBB67AE85u,0x3C6EF372u,0xA54FF53Au,
  0x510E527Fu,0x9B05688Cu,0x1F83D9ABu,0x5BE0CD19u};
static __constant__ unsigned char MSg[7][16] = {
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
  {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
  {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
  {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
  {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
  {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
  {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}};

// blake3 flag bits
#define B3_CHUNK_START 1u
#define B3_CHUNK_END   2u
#define B3_PARENT      4u
#define B3_ROOT        8u
#define B3_KEYED      16u

static __device__ void b3_compress(const uint32_t cv[8], const uint32_t msg[16],
                                   uint64_t t, uint32_t blen, uint32_t flags,
                                   uint32_t out[8]) {
  uint32_t v[16];
  #pragma unroll
  for (int i=0;i<8;i++) v[i]=cv[i];
  v[8]=IVg[0]; v[9]=IVg[1]; v[10]=IVg[2]; v[11]=IVg[3];
  v[12]=(uint32_t)t; v[13]=(uint32_t)(t>>32); v[14]=blen; v[15]=flags;
  #pragma unroll
  for (int r=0;r<7;r++){
    const unsigned char* s=MSg[r];
    #define MIXP(a,b,c,d,x,y) \
      v[a]+=v[b]+(x); v[d]=rotr32g(v[d]^v[a],16); v[c]+=v[d]; v[b]=rotr32g(v[b]^v[c],12); \
      v[a]+=v[b]+(y); v[d]=rotr32g(v[d]^v[a],8);  v[c]+=v[d]; v[b]=rotr32g(v[b]^v[c],7);
    MIXP(0,4, 8,12, msg[s[0]],  msg[s[1]]);
    MIXP(1,5, 9,13, msg[s[2]],  msg[s[3]]);
    MIXP(2,6,10,14, msg[s[4]],  msg[s[5]]);
    MIXP(3,7,11,15, msg[s[6]],  msg[s[7]]);
    MIXP(0,5,10,15, msg[s[8]],  msg[s[9]]);
    MIXP(1,6,11,12, msg[s[10]], msg[s[11]]);
    MIXP(2,7, 8,13, msg[s[12]], msg[s[13]]);
    MIXP(3,4, 9,14, msg[s[14]], msg[s[15]]);
    #undef MIXP
  }
  #pragma unroll
  for (int i=0;i<8;i++) out[i]=v[i]^v[i+8];
}

// ---------------- RNG fill (byte-exact port of produce_draw step a) ----------
// CPU reference: per row i, st = (seed^XORC) + d*1000003 + i*0x100000001B3;
// per 8-byte group q: z = splitmix64_mix(st += GOLDEN)  ==  mix(base+(q+1)*GOLDEN)
// (closed form -> embarrassingly parallel). byte b -> ((z>>8b & 0xFF)*129>>8)-64.
__global__ void rng_fill_kernel(int8_t* __restrict__ out, size_t rows, int k,
                                uint64_t seed_xored, uint64_t d) {
  const size_t ngroups = rows * (size_t)(k/8);
  const uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;
  for (size_t g = (size_t)blockIdx.x*blockDim.x + threadIdx.x; g < ngroups;
       g += (size_t)gridDim.x*blockDim.x) {
    size_t i = g / (size_t)(k/8);
    size_t q = g % (size_t)(k/8);
    uint64_t st = seed_xored + d*1000003ULL + i*0x100000001B3ULL + (q+1)*GOLDEN;
    uint64_t z = st;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; z=z^(z>>31);
    uint64_t pack = 0;
    #pragma unroll
    for (int b=0;b<8;b++) {
      int8_t v = (int8_t)((int)((((uint32_t)((z>>(8*b))&0xFF))*129u)>>8)-64);
      pack |= ((uint64_t)(uint8_t)v) << (8*b);
    }
    reinterpret_cast<uint64_t*>(out)[g] = pack;   // k%8==0 -> 8B-aligned
  }
}

// ---------------- BLAKE3 keyed tree hash of a device buffer ------------------
// nbytes must be a multiple of 1024 with a power-of-two chunk count (real
// config: 131072*4096 = 2^29 B = 2^19 chunks -> perfectly balanced tree, no
// partial chunks). Key = job_key (KEYED_HASH on every compression).
static __constant__ uint32_t c_hkey[8];

__global__ void b3_chunk_kernel(const uint8_t* __restrict__ data, size_t nchunks,
                                uint32_t* __restrict__ cvs) {
  size_t c = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if (c >= nchunks) return;
  uint32_t cv[8];
  #pragma unroll
  for (int i=0;i<8;i++) cv[i]=c_hkey[i];
  const uint32_t* p = reinterpret_cast<const uint32_t*>(data + c*1024);
  for (int b=0;b<16;b++) {
    uint32_t msg[16];
    #pragma unroll
    for (int w=0;w<16;w++) msg[w] = p[b*16+w];     // bytes are LE words already
    uint32_t flags = B3_KEYED | (b==0 ? B3_CHUNK_START:0) | (b==15 ? B3_CHUNK_END:0);
    b3_compress(cv, msg, (uint64_t)c, 64, flags, cv);
  }
  #pragma unroll
  for (int i=0;i<8;i++) cvs[c*8+i]=cv[i];
}

__global__ void b3_parent_kernel(const uint32_t* __restrict__ in, size_t nparents,
                                 uint32_t* __restrict__ outv, int root) {
  size_t pidx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if (pidx >= nparents) return;
  uint32_t msg[16];
  #pragma unroll
  for (int i=0;i<16;i++) msg[i]=in[pidx*16+i];     // left cv (8) || right cv (8)
  uint32_t flags = B3_KEYED | B3_PARENT | (root ? B3_ROOT : 0);
  uint32_t cv[8];
  #pragma unroll
  for (int i=0;i<8;i++) cv[i]=c_hkey[i];
  uint32_t o[8];
  b3_compress(cv, msg, 0, 64, flags, o);
  #pragma unroll
  for (int i=0;i<8;i++) outv[pidx*8+i]=o[i];
}

// ---------------- noise: 8 keyed hashes/row -> e[256] -> perm-diff add -------
// uniform_row + matvec_sparse_perm + the noised-build, fused, in-place.
// rank is fixed at 256 (real config): start_idx = row*256 is 32-aligned, so
// each row consumes exactly 8 whole hash blocks. Message layout
// (get_random_hash, prepend_index=0): bytes[0..3]=LE(1+block_index), 4..31=0,
// 32..63 = seed label. Keyed by the noise seed; single 64-byte block ->
// flags = CHUNK_START|CHUNK_END|ROOT|KEYED = 27, t=0, blen=64 (identical
// constants to the validated jackpot_blake3).
static __constant__ uint32_t c_nzkey[8];    // a_/b_noise_seed words
static __constant__ uint32_t c_nzlabel[8];  // SEED_LABEL_A/B words

__global__ void noise_add_kernel(int8_t* __restrict__ mat, size_t rows, int k,
                                 const uint32_t* __restrict__ perm /* k pairs */) {
  __shared__ int8_t e[256];
  size_t row = blockIdx.x;
  if (row >= rows) return;
  if (threadIdx.x < 8) {
    uint64_t block = row*8 + threadIdx.x;
    uint32_t msg[16];
    msg[0] = (uint32_t)(1 + block);                // LE i32 at byte 0
    #pragma unroll
    for (int i=1;i<8;i++) msg[i]=0;
    #pragma unroll
    for (int i=0;i<8;i++) msg[8+i]=c_nzlabel[i];
    uint32_t h[8];
    b3_compress(c_nzkey, msg, 0, 64, 27, h);
    #pragma unroll
    for (int wd=0; wd<8; wd++) {
      uint32_t x = h[wd];
      #pragma unroll
      for (int by=0; by<4; by++)
        e[threadIdx.x*32 + wd*4 + by] = (int8_t)((int)((x>>(8*by)) & 63u) - 32);
    }
  }
  __syncthreads();
  for (int l = threadIdx.x; l < k; l += blockDim.x) {
    uint32_t p0 = perm[2*l], p1 = perm[2*l+1];
    mat[row*(size_t)k + l] = (int8_t)(mat[row*(size_t)k + l] + e[p0] - e[p1]);
  }
}

// ---------------- host orchestration ----------------------------------------
struct PrepBufs {
  uint32_t *cv0=nullptr, *cv1=nullptr;   // tree ping-pong (16 MB + 8 MB)
  uint32_t *dperm=nullptr;               // k pairs x 2 matrices
  bool ok=false;
};
static PrepBufs g_prep;

// All prep work runs on a private HIGH-PRIORITY non-blocking stream so it can
// execute UNDER the in-flight search kernel (tc_cutlass_v2 launches async on
// its own stream; prep writes dA/dBt while the search only reads the gathered
// dAp/dBtp panels — no conflict). High priority lets prep blocks slot into SMs
// as search waves retire instead of starving behind the 512k-block search grid.
// Both phases still cudaStreamSynchronize THIS stream before returning, so the
// host-visible contract (buffers ready on return) is unchanged.
static cudaStream_t g_prep_stream = nullptr;
static bool ensure_prep_stream() {
  if (g_prep_stream) return true;
  int lo = 0, hi = 0;
  cudaDeviceGetStreamPriorityRange(&lo, &hi);   // hi = greatest priority
  if (cudaStreamCreateWithPriority(&g_prep_stream, cudaStreamNonBlocking, hi)
      == cudaSuccess) return true;
  return cudaStreamCreateWithFlags(&g_prep_stream, cudaStreamNonBlocking)
      == cudaSuccess;
}

// Recorded by tc_search_launch right after the gather kernels (the last
// READERS of dA/dBt for draw N). Phase1 orders its dA/dBt writes behind it so
// prep(N+1) can run under search(N) without clobbering the gathers' input.
// Only tc_cutlass_v2 defines it; builds without the async split define
// KAN_NO_ASYNC_SEARCH so the reference vanishes (no weak needed, which MSVC lacks).
#ifndef KAN_NO_ASYNC_SEARCH
extern "C" void* tc_gather_done_event() KAN_WEAK;
#endif

static bool ensure_prep_bufs(size_t max_chunks, int k) {
  if (g_prep.ok) return true;
  if (cudaMalloc(&g_prep.cv0, max_chunks*8*4) ||
      cudaMalloc(&g_prep.cv1, (max_chunks/2)*8*4) ||
      cudaMalloc(&g_prep.dperm, (size_t)k*2*4*2)) {
    fprintf(stderr, "gpu_prep: scratch malloc fail\n");
    return false;
  }
  g_prep.ok = true;
  return true;
}

// keyed tree hash of dbuf[nbytes] -> 32-byte digest (host out). Assumes
// power-of-two chunk count (checked by caller).
static int tree_hash(const uint8_t* dbuf, size_t nbytes, uint8_t out32[32]) {
  size_t nchunks = nbytes / 1024;
  b3_chunk_kernel<<<(unsigned)((nchunks+255)/256), 256, 0, g_prep_stream>>>(dbuf, nchunks, g_prep.cv0);
  uint32_t *src = g_prep.cv0, *dst = g_prep.cv1;
  size_t ncv = nchunks;
  while (ncv > 1) {
    size_t npar = ncv/2;
    b3_parent_kernel<<<(unsigned)((npar+255)/256), 256, 0, g_prep_stream>>>(src, npar, dst, npar==1);
    uint32_t* t = src; src = dst; dst = t;
    ncv = npar;
  }
  return cudaMemcpyAsync(out32, src, 32, cudaMemcpyDeviceToHost, g_prep_stream) == cudaSuccess ? 0 : -1;
}

// phase1: RNG fill + commitments. seed/draw/job_key in; hash_a/hash_b out.
extern "C" int gpu_prep_phase1(
    signed char* dA, signed char* dBt, int m, int n, int k,
    uint64_t seed, uint64_t draw, const unsigned char job_key32[32],
    unsigned char hash_a32[32], unsigned char hash_b32[32],
    double* ms_rng, double* ms_hash)
{
  size_t abytes = (size_t)m*k, bbytes = (size_t)n*k;
  if (abytes % 1024 || bbytes % 1024) return -2;
  size_t ach = abytes/1024, bch = bbytes/1024;
  if ((ach & (ach-1)) || (bch & (bch-1))) return -2;   // need power-of-two chunks
  if (k % 8) return -2;
  if (!ensure_prep_bufs(ach > bch ? ach : bch, k)) return -1;
  if (!ensure_prep_stream()) return -1;
  // Don't write dA/dBt until search(N)'s gathers have read them.
#ifndef KAN_NO_ASYNC_SEARCH
  if (tc_gather_done_event) {
    cudaEvent_t ge = (cudaEvent_t)tc_gather_done_event();
    if (ge) cudaStreamWaitEvent(g_prep_stream, ge, 0);
  }
#endif

  uint32_t kw[8];
  for (int i=0;i<8;i++)
    kw[i]=(uint32_t)job_key32[i*4]|((uint32_t)job_key32[i*4+1]<<8)|
          ((uint32_t)job_key32[i*4+2]<<16)|((uint32_t)job_key32[i*4+3]<<24);
  cudaMemcpyToSymbolAsync(c_hkey, kw, 32, 0, cudaMemcpyHostToDevice, g_prep_stream);

  cudaEvent_t e0,e1,e2; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  cudaEventRecord(e0, g_prep_stream);
  {
    size_t ng = (size_t)m*(k/8);
    rng_fill_kernel<<<(unsigned)((ng+255)/256), 256, 0, g_prep_stream>>>((int8_t*)dA, m, k,
        seed ^ 0x9E3779B97F4A7C15ULL, draw);
    ng = (size_t)n*(k/8);
    rng_fill_kernel<<<(unsigned)((ng+255)/256), 256, 0, g_prep_stream>>>((int8_t*)dBt, n, k,
        seed ^ 0xD1B54A32D192ED03ULL, draw);
  }
  cudaEventRecord(e1, g_prep_stream);
  if (tree_hash((const uint8_t*)dA, abytes, hash_a32)) return -1;
  if (tree_hash((const uint8_t*)dBt, bbytes, hash_b32)) return -1;
  cudaEventRecord(e2, g_prep_stream);
  cudaStreamSynchronize(g_prep_stream);
  float f01=0,f12=0;
  cudaEventElapsedTime(&f01,e0,e1); cudaEventElapsedTime(&f12,e1,e2);
  if (ms_rng) *ms_rng=f01;
  if (ms_hash) *ms_hash=f12;
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { fprintf(stderr,"gpu_prep1: %s\n",cudaGetErrorString(err)); return -1; }
  return 0;
}

// phase2: noise + in-place add. Perms are host pointers (k pairs each).
extern "C" int gpu_prep_phase2(
    signed char* dA, signed char* dBt, int m, int n, int k, int rank,
    const unsigned char a_noise_seed32[32], const unsigned char b_noise_seed32[32],
    const unsigned char label_a32[32], const unsigned char label_b32[32],
    const unsigned int* perm_a, const unsigned int* perm_b,
    double* ms_noise)
{
  if (rank != 256) { fprintf(stderr,"gpu_prep2: rank must be 256 (got %d)\n",rank); return -2; }
  if (!g_prep.ok) return -1;
  if (!ensure_prep_stream()) return -1;

  uint32_t* dpa = g_prep.dperm;
  uint32_t* dpb = g_prep.dperm + (size_t)k*2;
  cudaMemcpyAsync(dpa, perm_a, (size_t)k*2*4, cudaMemcpyHostToDevice, g_prep_stream);
  cudaMemcpyAsync(dpb, perm_b, (size_t)k*2*4, cudaMemcpyHostToDevice, g_prep_stream);

  uint32_t kwa[8], kwb[8], la[8], lb[8];
  for (int i=0;i<8;i++) {
    kwa[i]=(uint32_t)a_noise_seed32[i*4]|((uint32_t)a_noise_seed32[i*4+1]<<8)|
           ((uint32_t)a_noise_seed32[i*4+2]<<16)|((uint32_t)a_noise_seed32[i*4+3]<<24);
    kwb[i]=(uint32_t)b_noise_seed32[i*4]|((uint32_t)b_noise_seed32[i*4+1]<<8)|
           ((uint32_t)b_noise_seed32[i*4+2]<<16)|((uint32_t)b_noise_seed32[i*4+3]<<24);
    la[i]=(uint32_t)label_a32[i*4]|((uint32_t)label_a32[i*4+1]<<8)|
          ((uint32_t)label_a32[i*4+2]<<16)|((uint32_t)label_a32[i*4+3]<<24);
    lb[i]=(uint32_t)label_b32[i*4]|((uint32_t)label_b32[i*4+1]<<8)|
          ((uint32_t)label_b32[i*4+2]<<16)|((uint32_t)label_b32[i*4+3]<<24);
  }

  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, g_prep_stream);
  cudaMemcpyToSymbolAsync(c_nzkey, kwa, 32, 0, cudaMemcpyHostToDevice, g_prep_stream);
  cudaMemcpyToSymbolAsync(c_nzlabel, la, 32, 0, cudaMemcpyHostToDevice, g_prep_stream);
  noise_add_kernel<<<(unsigned)m, 256, 0, g_prep_stream>>>((int8_t*)dA, m, k, dpa);
  cudaMemcpyToSymbolAsync(c_nzkey, kwb, 32, 0, cudaMemcpyHostToDevice, g_prep_stream);
  cudaMemcpyToSymbolAsync(c_nzlabel, lb, 32, 0, cudaMemcpyHostToDevice, g_prep_stream);
  noise_add_kernel<<<(unsigned)n, 256, 0, g_prep_stream>>>((int8_t*)dBt, n, k, dpb);
  cudaEventRecord(e1, g_prep_stream);
  cudaStreamSynchronize(g_prep_stream);
  float f=0; cudaEventElapsedTime(&f,e0,e1);
  if (ms_noise) *ms_noise=f;
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { fprintf(stderr,"gpu_prep2: %s\n",cudaGetErrorString(err)); return -1; }
  return 0;
}
