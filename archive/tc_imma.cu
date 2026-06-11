// tc_imma.cu — IMMA.16832 int8 tensor-core GEMM with fused jackpot fold.
//
// Replaces tc_block.cu's WMMA 16x16x16 with mma.sync PTX IMMA 16x8x32.
// Each IMMA instruction does 16*8*32 = 4096 MAC (same count as WMMA but
// covers K=32 in one shot and uses registers far more efficiently).
//
// Reverse-engineered from lpminer's SASS (512× IMMA.16832.S8.S8 on sm_86).
// Tile: BM=128, BN=128, BK=32. 8 warps, each warp: 16 rows × 128 cols.
// 8 "super-subtiles" of 16×16 (each from 2 IMMA.16832 calls).
// Persistent device buffers (allocated once, reused across draws).
//
// WMMA vs IMMA on Ampere:
//   WMMA 16x16x16 s8: opaque fragment, 256 bytes per warp per load
//   IMMA 16x8x32  s8: 4 regs A + 2 regs B, K=32 in one instruction
//   Same MAC per inst (4096), but IMMA has 2× K coverage and much better
//   register density → higher effective throughput.

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include <cstdio>

#define IBM 128
#define IBN 128
#define IBK 32
#define IWPB 8          // warps per block
#define ISTAGES 2       // cp.async pipeline depth
#define IMAXR 256
#define IMAXJT 128      // (IBM/8)*(IBN/16) = 16*8 = 128

// ---------- device helpers ----------
static __device__ __forceinline__ uint32_t pack4s8(const signed char* p) {
    return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1]<<8)
         | ((uint32_t)(uint8_t)p[2]<<16) | ((uint32_t)(uint8_t)p[3]<<24);
}

// IMMA.16832: D[16x8] += A[16x32] * B[32x8], all int8, accum s32
static __device__ void imma16832(
    int &d0, int &d1, int &d2, int &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int c0, int c1, int c2, int c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n\t"
        "  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}

// ---------- jackpot blake3 (identical to tc_block) ----------
static __device__ __forceinline__ uint32_t rotr32(uint32_t x,int n){return(x>>n)|(x<<(32-n));}
static __device__ __forceinline__ uint32_t rotl32d(uint32_t x,int n){return(x<<n)|(x>>(32-n));}
static __constant__ uint32_t IVc[8]={
  0x6A09E667u,0xBB67AE85u,0x3C6EF372u,0xA54FF53Au,
  0x510E527Fu,0x9B05688Cu,0x1F83D9ABu,0x5BE0CD19u};
static __constant__ unsigned char MS[7][16]={
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
  {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
  {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
  {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
  {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
  {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
  {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}};
static __device__ void jackpot_blake3(const uint32_t key[8],const uint32_t msg[16],uint32_t out[8]){
  uint32_t v[16]; for(int i=0;i<8;i++)v[i]=key[i];
  v[8]=IVc[0];v[9]=IVc[1];v[10]=IVc[2];v[11]=IVc[3];v[12]=0;v[13]=0;v[14]=64;v[15]=27;
  for(int r=0;r<7;r++){const unsigned char*s=MS[r];
    #define MIX(a,b,c,d,x,y) v[a]+=v[b]+(x);v[d]=rotr32(v[d]^v[a],16);v[c]+=v[d];v[b]=rotr32(v[b]^v[c],12);v[a]+=v[b]+(y);v[d]=rotr32(v[d]^v[a],8);v[c]+=v[d];v[b]=rotr32(v[b]^v[c],7);
    MIX(0,4,8,12,msg[s[0]],msg[s[1]]);MIX(1,5,9,13,msg[s[2]],msg[s[3]]);
    MIX(2,6,10,14,msg[s[4]],msg[s[5]]);MIX(3,7,11,15,msg[s[6]],msg[s[7]]);
    MIX(0,5,10,15,msg[s[8]],msg[s[9]]);MIX(1,6,11,12,msg[s[10]],msg[s[11]]);
    MIX(2,7,8,13,msg[s[12]],msg[s[13]]);MIX(3,4,9,14,msg[s[14]],msg[s[15]]);
    #undef MIX
  }
  for(int i=0;i<8;i++)out[i]=v[i]^v[i+8];
}
static __device__ __forceinline__ bool le_u256(const uint32_t a[8],const uint32_t b[8]){
  for(int i=7;i>=0;i--){if(a[i]!=b[i])return a[i]<b[i];}return true;
}

// ---------- GATHER (identical to tc_block) ----------
__global__ void gather_rows(const signed char*__restrict__ src,signed char*__restrict__ dst,
                            int k,const int*__restrict__ off,const int*__restrict__ pat,int h,int noff){
  int rp=blockIdx.x,i=rp/h,u=rp%h; if(i>=noff)return;
  size_t s=(size_t)(off[i]+pat[u])*k,d=(size_t)rp*k;
  for(int l=threadIdx.x;l<k;l+=blockDim.x)dst[d+l]=src[s+l];
}

// ---------- main kernel ----------
__global__ void __launch_bounds__(IWPB*32) tc_imma_jackpot(
    const signed char*__restrict__ Ap,const signed char*__restrict__ Btp,
    int k,int rank,int h,int w,
    int nrow_off,int ncol_off,
    const uint32_t*__restrict__ key,const uint32_t*__restrict__ bound,
    int*__restrict__ win_flag,int*__restrict__ win_rt,int*__restrict__ win_ct)
{
  // Warp geometry: 8 warps, each owns 16 consecutive rows × all 128 cols
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warp_row_base = warp * 16;   // this warp's starting row in the BM×BN block
  const int bi = blockIdx.y * (IBM/h);   // row-tile base (in row_off units)
  const int bj = blockIdx.x * (IBN/w);   // col-tile base

  const int RToff = IBM / h;   // 16
  const int CToff = IBN / w;   // 8
  const int jtr = 16 / h;      // 2 jackpot-tile rows per 16-row group
  const int jtc = 16 / w;      // 1 jackpot-tile col per 16-col group
  const int njt_warp = (16/h) * (IBN/w); // 2*8=16 jackpot tiles per warp
  const int njt = RToff * CToff; // 128 total per block

  __shared__ __align__(32) signed char a_sh[ISTAGES][IBM][IBK];
  __shared__ __align__(32) signed char b_sh[ISTAGES][IBN][IBK];
  __shared__ int32_t  o_sh[IWPB][16][16];
  __shared__ uint32_t jp_sh[IMAXJT][16];

  for (int t=threadIdx.x; t<njt*16; t+=blockDim.x) jp_sh[t/16][t%16]=0;

  // Accumulator: 8 super-subtiles × 8 s32 (2 IMMA × 4 regs each) = 64 regs
  // acc[s][0..3] = first IMMA (cols s*16+0..7)
  // acc[s][4..7] = second IMMA (cols s*16+8..15)
  int acc[8][8];
  for(int s=0;s<8;s++) for(int r=0;r<8;r++) acc[s][r]=0;

  const int nchunks = (k-(k%rank))/rank;  // 16 for real config
  const int nsub = rank/IBK;               // 8 BK steps per rank-chunk
  const int nstages = nchunks * nsub;      // 128 total k-steps

  // --- cp.async load macro (same pattern as tc_block) ---
  #define LOAD_SLICE(buf, kbexpr) do { \
    int _kb=(kbexpr); \
    for(int e=threadIdx.x;e<IBM*(IBK/16);e+=blockDim.x){ \
      int r=e/(IBK/16),o=(e%(IBK/16))*16; \
      __pipeline_memcpy_async(&a_sh[buf][r][o],&Ap[(size_t)(bi*h+warp_row_base+r)*k+_kb+o],16);} \
    for(int e=threadIdx.x;e<IBN*(IBK/16);e+=blockDim.x){ \
      int cc=e/(IBK/16),o=(e%(IBK/16))*16; \
      __pipeline_memcpy_async(&b_sh[buf][cc][o],&Btp[(size_t)(bj*w+cc)*k+_kb+o],16);} \
    __pipeline_commit(); \
  } while(0)

  // NOTE: the LOAD_SLICE above loads ONLY this warp's 16 rows of A, not all 128.
  // That's wrong — all warps share a_sh. Let me fix: load full BM rows.
  // Actually, all threads in the block cooperate on the load, so threadIdx.x
  // strides over all 256 threads. The a_sh loads BM rows (all 128), b_sh loads
  // BN rows (all 128). Each warp then reads its own 16-row slice during MMA.

  // Reload macro that loads FULL BM×BN (correct for multi-warp sharing)
  #undef LOAD_SLICE
  #define LOAD_SLICE(buf, kbexpr) do { \
    int _kb=(kbexpr); \
    for(int e=threadIdx.x;e<IBM*(IBK/16);e+=blockDim.x){ \
      int r=e/(IBK/16),o=(e%(IBK/16))*16; \
      __pipeline_memcpy_async(&a_sh[buf][r][o],&Ap[(size_t)(bi*h+r)*k+_kb+o],16);} \
    for(int e=threadIdx.x;e<IBN*(IBK/16);e+=blockDim.x){ \
      int cc=e/(IBK/16),o=(e%(IBK/16))*16; \
      __pipeline_memcpy_async(&b_sh[buf][cc][o],&Btp[(size_t)(bj*w+cc)*k+_kb+o],16);} \
    __pipeline_commit(); \
  } while(0)

  // prologue
  #pragma unroll
  for(int s=0;s<ISTAGES-1;s++){
    if(s<nstages) LOAD_SLICE(s%ISTAGES,s*IBK);
    else __pipeline_commit();
  }

  for(int st=0;st<nstages;st++){
    const int buf=st%ISTAGES;
    const int pf=st+(ISTAGES-1);
    if(pf<nstages) LOAD_SLICE(pf%ISTAGES,pf*IBK);
    else __pipeline_commit();
    __pipeline_wait_prior(ISTAGES-1);
    __syncthreads();

    // --- Load A & B fragments via ldmatrix PTX (layout guaranteed correct for mma.sync) ---
    // ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 loads 4 regs (16×8 s8 transposed)
    // from shared memory in the exact layout expected by mma.sync.
    // For a 16×32 A tile: we load from our warp's 16-row slice in a_sh.
    // For B: we load from the appropriate 8-col slice in b_sh.
    {
      // A fragment: 16 rows × 32 cols, loaded from a_sh[buf][warp_row_base + r][0..31]
      // ldmatrix.m8n8.x4 loads 4 registers covering a 16×8 region.
      // For 16×32: need 4 ldmatrix.x2 calls (each loads 2 regs covering 8×8).
      // Or: load as 4 rows of 8 elements using ldmatrix.x4 with .trans

      // Use ldmatrix.x4.trans to load a 16×32 tile (4 calls of 16×8 each)
      // Actually ldmatrix.m8n8.x4 loads 4 consecutive registers from a shared mem ptr.
      // For row-major A, we DON'T want transpose (.trans).
      // For col-major B, we DO want transpose (.trans) since b_sh is row-major.

      // A: row-major 16×32. ldmatrix.m8n8.x4 loads 4 regs from smem.
      // Each reg = 4 s8, 4 regs = 16 s8. This covers one 8-column slice of 16 rows.
      // Need 4 slices (cols 0-7, 8-15, 16-23, 24-31).
      // But wait: ldmatrix.x4 with .trans loads a TRANSPOSED view.
      // Without .trans: loads 16 rows × 8 cols (row-major) into 4 regs.
      //   But that's 128 elements = 128 s8, while we have 4 regs × 4 s8 = 16 s8. Mismatch!

      // Actually ldmatrix.x4 for s8:
      //   m8n8 = 8×8 matrix = 64 elements. x4 = 4 matrices = 256 elements.
      //   Each element is 1 byte (s8). 256 bytes = 8 registers (4 pairs).
      //   But IMMA needs only 4 A regs (16 s8). So ldmatrix.x4 is too much.

      // ldmatrix.x1: loads 1 reg (4 s8 from 8×8 view). Not enough.
      // ldmatrix.x2: loads 2 regs (8 s8 from 2×(8×8) view). Two calls = 4 regs = 16 s8.

      // CORRECT approach: use ldmatrix.m8n8.x2 for A (2 calls for 4 regs)
      // and ldmatrix.m8n8.x2 for B (1 call for 2 regs)

      // The shared memory must be laid out so that ldmatrix reads the right elements.
      // For row-major A in a_sh[BM][BK]:
      //   ldmatrix.m8n8.x2 without .trans reads 2 consecutive 8-byte rows from smem
      //   starting at [row][col], where the thread reads its own row.
      //   Thread lane reads: row = lane (for x2, threads 0-15 each read 2 rows)

      // The IMMA A fragment has 16 rows. ldmatrix.x2 maps:
      //   thread 0 → rows 0,1, thread 1 → rows 2,3, ..., thread 15 → rows 30,31
      //   BUT we only have 16 rows, not 32. The mapping is:
      //   thread t reads row 2*t and 2*t+1 for the first x2 call
      //   and rows 2*t+16 and 2*t+17 for the second x2 call (to cover K=32)

      // B fragment: col-major 32×8. b_sh is row-major (BN rows × BK cols).
      // We want B[k][n] = b_sh[n][k]. With .trans, ldmatrix reads column-wise.
      // ldmatrix.m8n8.x2.trans: thread t reads column t from 8 rows of smem
      //   into 2 regs (each 4 s8, covering 8 consecutive rows).

      // For 32 rows of B: need 4 ldmatrix.x2.trans calls (8+8+8+8 = 32 rows)
      // But B fragment is only 2 regs (8 s8 = 1 column of 8 rows).
      // So thread t covers 8 rows of its column, and the hardware handles the rest.

      // PRAGMATIC: use PTX inline for ldmatrix with the correct addressing.
      // The shared memory layout a_sh[BM][BK] is row-major with BK=32.
      // For warp's A (16 rows at warp_row_base):
      //   Base address = &a_sh[buf][warp_row_base][0]

      uint32_t a0, a1, a2, a3;
      uint32_t b0, b1;

      // Load A fragment using ldmatrix.x4 (covers 16×8 = 128 s8 → 4 regs of 4 s8 each... no)
      // Actually for s8 with ldmatrix:
      //   Each matrix element = 1 byte (int8)
      //   ldmatrix.m8n8.x1 = 8×8 = 64 bytes = 2 regs of 32 bytes each? No.
      //
      // ldmatrix encodes element size in the .b16 suffix (16-bit per element).
      // For int8, we reinterpret: pack 2 s8 as 1 b16.
      // ldmatrix.m8n8.x4.b16 loads 4 regs of b16 (4×16 bits = 8 bytes) per thread.
      // But we need 16 s8 = 16 bytes = 4 regs of 32 bits.

      // NVIDIA's documentation for ldmatrix:
      //   .b16: each element is 16 bits (2 bytes). For s8 data, each "element" packs 2 s8.
      //   m8n8 = 8×8 matrix of b16 = 64 × 2 = 128 bytes per matrix.
      //   x4 = 4 matrices = 512 bytes. 32 threads × 16 bytes = 512 bytes ✓.
      //   Each thread gets 4 regs × 4 bytes = 16 bytes = 8 b16 elements = 16 s8. ✓

      // So: ldmatrix.sync.aligned.m8n8.x4.shared.b16 loads 4 regs per thread
      // from shared memory, covering a 16×32 s8 region (2 s8 per b16 element).

      // For A: use base of warp's 16-row slice at column 0
      // ldmatrix.x4.b16 without .trans covers 16×32 s8 (8 b16 per element pair)
      uint32_t a_base = (uint32_t)__cvta_generic_to_shared(&a_sh[buf][warp_row_base][0]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16"
                   " {%0,%1,%2,%3}, [%4];"
                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                   : "r"(a_base));

      for(int ss=0;ss<8;ss++){
        int col_base = ss * 16;

        // B fragment: 32 rows × 8 cols, stored as b_sh[col][k] (row-major).
        // ldmatrix.x2.trans.b16: loads 2 regs per thread from 8×16 s8 = 8×8 b16,
        // reading COLUMN-wise (.trans) from row-major smem.
        // Address must be 16-byte aligned -> use row start (&b_sh[buf][col][0]).
        int b_col = col_base + (lane / 4);
        uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(&b_sh[buf][b_col][0]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"
                     " {%0,%1}, [%2];"
                     : "=r"(b0), "=r"(b1)
                     : "r"(b_addr));

        // IMMA 1: columns 0..7
        {
          int d0=acc[ss][0],d1=acc[ss][1],d2=acc[ss][2],d3=acc[ss][3];
          imma16832(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, d0,d1,d2,d3);
          acc[ss][0]=d0; acc[ss][1]=d1; acc[ss][2]=d2; acc[ss][3]=d3;
        }
        // IMMA 2: columns 8..15 (different B, same A)
        {
          int b_col2 = col_base + 8 + (lane / 4);
          uint32_t b_addr2 = (uint32_t)__cvta_generic_to_shared(&b_sh[buf][b_col2][0]);
          uint32_t b0b, b1b;
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"
                       " {%0,%1}, [%2];"
                       : "=r"(b0b), "=r"(b1b)
                       : "r"(b_addr2));
          int d0=acc[ss][4],d1=acc[ss][5],d2=acc[ss][6],d3=acc[ss][7];
          imma16832(d0,d1,d2,d3, a0,a1,a2,a3, b0b,b1b, d0,d1,d2,d3);
          acc[ss][4]=d0; acc[ss][5]=d1; acc[ss][6]=d2; acc[ss][7]=d3;
        }
      }
    }

    // --- Fold after every rank-chunk ---
    if((st+1)%nsub==0){
      const int c = st/nsub;
      for(int ss=0;ss<8;ss++){
        int sgr = warp;       // super-subtile row group index
        int sgc = ss;         // super-subtile col group index

        // Store accumulator to o_sh (16×16 row-major)
        // Thread lane holds:
        //   IMMA1: D[lane%16][(lane/16)*4+j] for j=0..3 → cols ss*16+0..7
        //   IMMA2: D[lane%16][(lane/16)*4+j] for j=0..3 → cols ss*16+8..15
        int row = lane % 16;
        int cb1 = (lane/16)*4;       // 0 or 4
        int cb2 = (lane/16)*4 + 8;   // 8 or 12
        o_sh[warp][row][cb1+0]=acc[ss][0]; o_sh[warp][row][cb1+1]=acc[ss][1];
        o_sh[warp][row][cb1+2]=acc[ss][2]; o_sh[warp][row][cb1+3]=acc[ss][3];
        o_sh[warp][row][cb2+0]=acc[ss][4]; o_sh[warp][row][cb2+1]=acc[ss][5];
        o_sh[warp][row][cb2+2]=acc[ss][6]; o_sh[warp][row][cb2+3]=acc[ss][7];
        __syncwarp();

        // XOR-fold jackpot tiles within this 16×16 super-subtile
        for(int sr=0;sr<jtr;sr++){
          for(int sc=0;sc<jtc;sc++){
            int jt_i = bi + sgr*jtr + sr;
            int jt_j = bj + sgc*jtc + sc;
            if(jt_i>=nrow_off||jt_j>=ncol_off) continue;
            uint32_t x=0;
            for(int idx=lane;idx<h*w;idx+=32){
              int rr=idx/w, ccc=idx%w;
              x ^= (uint32_t)o_sh[warp][sc*w+ccc][sr*h+rr];
            }
            for(int off=16;off>0;off>>=1) x ^= __shfl_xor_sync(0xffffffffu,x,off);
            if(lane==0){
              int local_jt = (sgr*jtr+sr)*CToff + (sgc*jtc+sc);
              jp_sh[local_jt][c%16] = rotl32d(jp_sh[local_jt][c%16],13)^x;
            }
          }
        }
        __syncwarp();
      }
    }
    __syncthreads();
  }

  // --- Final jackpot blake3 + compare ---
  for(int t=threadIdx.x;t<njt;t+=blockDim.x){
    int jt_i=bi+t/CToff, jt_j=bj+t%CToff;
    if(jt_i>=nrow_off||jt_j>=ncol_off) continue;
    uint32_t jp[16]; for(int q=0;q<16;q++) jp[q]=jp_sh[t][q];
    uint32_t out[8]; jackpot_blake3(key,jp,out);
    if(le_u256(out,bound)){
      if(atomicCAS(win_flag,0,1)==0){*win_rt=jt_i;*win_ct=jt_j;}
    }
  }
}

// ---------- Host-side API (same extern "C" as tc_block) ----------
static inline void words_from_le32(const unsigned char*b,uint32_t w[8]){
  for(int i=0;i<8;i++)
    w[i]=(uint32_t)b[i*4]|((uint32_t)b[i*4+1]<<8)|((uint32_t)b[i*4+2]<<16)|((uint32_t)b[i*4+3]<<24);
}

struct IDevBufs {
  signed char *dA=nullptr,*dBt=nullptr,*dAp=nullptr,*dBtp=nullptr;
  int *dpr=nullptr,*dpc=nullptr,*droff=nullptr,*dcoff=nullptr;
  uint32_t *dk=nullptr,*db=nullptr;
  int *df=nullptr,*dr=nullptr,*dc=nullptr;
  bool ok=false;
};
static IDevBufs g_ibufs;

static bool ensure_idev_bufs(int m,int n,int k,int h,int w,int nrow_off,int ncol_off){
  if(!g_ibufs.ok){
    size_t apk=(size_t)nrow_off*h*k, bpk=(size_t)ncol_off*w*k;
    if(cudaMalloc(&g_ibufs.dA,(size_t)m*k)||cudaMalloc(&g_ibufs.dBt,(size_t)n*k)||
       cudaMalloc(&g_ibufs.dAp,apk)||cudaMalloc(&g_ibufs.dBtp,bpk)||
       cudaMalloc(&g_ibufs.dpr,h*4)||cudaMalloc(&g_ibufs.dpc,w*4)||
       cudaMalloc(&g_ibufs.droff,nrow_off*4)||cudaMalloc(&g_ibufs.dcoff,ncol_off*4)||
       cudaMalloc(&g_ibufs.dk,32)||cudaMalloc(&g_ibufs.db,32)||
       cudaMalloc(&g_ibufs.df,4)||cudaMalloc(&g_ibufs.dr,4)||cudaMalloc(&g_ibufs.dc,4)){
      fprintf(stderr,"tc_imma: persistent malloc fail\n"); g_ibufs.ok=false; return false;
    }
    g_ibufs.ok=true;
    fprintf(stderr,"tc_imma: persistent device buffers allocated (%zu MB)\n",
            ((size_t)m*k+(size_t)n*k+apk+bpk)/1024/1024);
  }
  return true;
}

extern "C" int tc_jackpot_search(
    const signed char* a_noised,const signed char* b_noised_t,
    int m,int n,int k,int rank,
    const int* pat_rows,const int* pat_cols,int h,int w,
    const int* row_off,int nrow_off,const int* col_off,int ncol_off,
    const unsigned char* a_noise_seed32,const unsigned char* bound_le32,
    unsigned int*,unsigned int*,
    int* out_rt,int* out_ct)
{
  if(rank>IMAXR||(rank%IBK)){fprintf(stderr,"tc_imma: bad rank %d\n",rank);return -3;}
  if((16%h)||(16%w)||(IBM%h)||(IBN%w)||(size_t)(IBM/h)*(IBN/w)>IMAXJT){
    fprintf(stderr,"tc_imma: h=%d w=%d unsupported\n",h,w);return -2;}

  if(!ensure_idev_bufs(m,n,k,h,w,nrow_off,ncol_off)) return -1;
  IDevBufs& B=g_ibufs;

  size_t ntiles=(size_t)nrow_off*ncol_off;
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

  gather_rows<<<nrow_off*h,256>>>(B.dA,B.dAp,k,B.droff,B.dpr,h,nrow_off);
  gather_rows<<<ncol_off*w,256>>>(B.dBt,B.dBtp,k,B.dcoff,B.dpc,w,ncol_off);

  dim3 grid((ncol_off+(IBN/w)-1)/(IBN/w),(nrow_off+(IBM/h)-1)/(IBM/h));
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  tc_imma_jackpot<<<grid,IWPB*32>>>(B.dAp,B.dBtp,k,rank,h,w,nrow_off,ncol_off,
                                     B.dk,B.db,B.df,B.dr,B.dc);
  cudaError_t le=cudaGetLastError();
  if(le!=cudaSuccess) fprintf(stderr,"tc_imma: LAUNCH err %s (tpb=%d, grid=%dx%d)\n",
      cudaGetErrorString(le),IWPB*32,grid.x,grid.y);
  cudaError_t err=cudaDeviceSynchronize();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  if(err==cudaSuccess) err=le;
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  // Same PearlHash/PoUW TH/s formula used by SRBMiner-MULTI/lpminer/pools.
  double mac=(double)ntiles*h*w*(k-(k%rank));
  fprintf(stderr,
          "tc(imma): BM=%d BN=%d BK=%d %zu tiles, %.3f ms, %.2f TH/s\n",
          IBM, IBN, IBK, ntiles, ms, mac / (ms * 1e-3) / 1e12);
  if(err!=cudaSuccess) fprintf(stderr,"tc_imma: err %s\n",cudaGetErrorString(err));

  int wf=0;
  if(err==cudaSuccess){
    cudaMemcpy(&wf,B.df,4,cudaMemcpyDeviceToHost);
    if(wf){cudaMemcpy(out_rt,B.dr,4,cudaMemcpyDeviceToHost);cudaMemcpy(out_ct,B.dc,4,cudaMemcpyDeviceToHost);}
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (err==cudaSuccess)?wf:-1;
}
