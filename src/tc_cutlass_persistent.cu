// tc_cutlass_persistent.cu — Phase 8 persistent scheduler variant
// Based on tc_cutlass_v2.cu with minimal changes:
//   1. Each threadblock processes MULTIPLE tiles in a loop
//   2. Grid size = num_SM * occupancy (128 or 256 blocks, not 524k)
//   3. Dynamic tile assignment: tile_idx = blockIdx.x; tile_idx += gridDim.x
//
// Expected: +5-15% over v2 (eliminate launch overhead, better L2 reuse)
//
// Build: PERSISTENT=1 ./build.sh

#define PERSISTENT 1
#include "tc_cutlass_v2.cu"  // Reuse all type definitions and host code
#undef tc_cutlass_jackpot    // We'll redefine the kernel

// ---- Tile ID mapping: preserve GROUPM strategy ----
__device__ __forceinline__ void tile_idx_to_block_coord(
    int tile_idx, int nbm, int nbn, int& bm_, int& bn_)
{
  // IDENTICAL logic to the GROUPM mapping in tc_cutlass_v2.cu kernel
  int pid    = tile_idx;
  int band   = pid / (GROUPM * nbn);
  int first  = band * GROUPM;
  int gsz    = (nbm - first < GROUPM) ? (nbm - first) : GROUPM;
  int rem    = pid - first * nbn;
  bm_ = first + rem % gsz;
  bn_ = rem / gsz;
}

// ---- PERSISTENT kernel: loop over multiple tiles ----
__global__ void __launch_bounds__(TPB, 1) tc_cutlass_jackpot(
    const int8_t* __restrict__ Ap, const int8_t* __restrict__ Btp,
    int k, int rank, int nrow_off, int ncol_off,
    const uint32_t* __restrict__ key, const uint32_t* __restrict__ bound,
    int* __restrict__ win_flag, int* __restrict__ win_rt, int* __restrict__ win_ct,
    int nbm, int nbn)  // NEW: pass grid dimensions
{
  extern __shared__ __align__(16) char smem_raw[];
  auto* mma_smem = reinterpret_cast<typename FoldMma::Base::SharedStorage*>(smem_raw);
  uint32_t* jp_sh = reinterpret_cast<uint32_t*>(smem_raw + sizeof(typename FoldMma::Base::SharedStorage));
#define JPS(t,q) jp_sh[(t)*17+(q)]

  const int thread_idx = threadIdx.x;
  const int warp_idx   = threadIdx.x >> 5;
  const int lane       = threadIdx.x & 31;
  const int warp_m = warp_idx % Mma::WarpCount::kM;
  const int warp_n = warp_idx / Mma::WarpCount::kM;

  const int M = nrow_off * H;
  const int N = ncol_off * W;
  const int kfold = k - (k % rank);
  const int gemm_k_iterations = kfold / FoldMma::Shape::kK;
  const int fold_every = rank / FoldMma::Shape::kK;

  typename FoldMma::IteratorA::Params paramsA{cutlass::layout::RowMajor(k)};
  typename FoldMma::IteratorB::Params paramsB{cutlass::layout::ColumnMajor(k)};

  const int total_tiles = nbm * nbn;

  // ---- PERSISTENT LOOP: each TB processes multiple tiles ----
  for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
    // Map tile_idx to (bm_, bn_) using GROUPM strategy
    int bm_, bn_;
    tile_idx_to_block_coord(tile_idx, nbm, nbn, bm_, bn_);

    const int bi = bm_ * RTOFF;
    const int bj = bn_ * CTOFF;

    // Clear jackpot shared memory
    for (int t = thread_idx; t < NJT*17; t += blockDim.x) jp_sh[t] = 0;
    __syncthreads();

    // Clear accumulator
    typename FoldMma::FragmentC accum;
    accum.clear();

    // Construct iterators for this tile
    typename FoldMma::IteratorA itA(paramsA, const_cast<int8_t*>(Ap), {M, kfold}, thread_idx,
                                    cutlass::MatrixCoord(bm_ * BM, 0));
    typename FoldMma::IteratorB itB(paramsB, const_cast<int8_t*>(Btp), {kfold, N}, thread_idx,
                                    cutlass::MatrixCoord(0, bn_ * BN));

    // Fold callback (same as v2, generalized for any warp shape)
    auto fold = [&](typename FoldMma::FragmentC const& acc, int c) {
      const int32_t* f = reinterpret_cast<const int32_t*>(acc.data());
      uint32_t myx = 0;
      #pragma unroll
      for (int jr = 0; jr < JR; ++jr) {
        const int m = jr >> 1, half = jr & 1;
        const int e0 = half*2, e1 = half*2 + 1;
        #pragma unroll
        for (int jc = 0; jc < JC; ++jc) {
          const int n0 = jc*2, n1 = n0 + 1;
          uint32_t x = (uint32_t)f[(m + n0*ROW_ITERS)*4 + e0]
                     ^ (uint32_t)f[(m + n0*ROW_ITERS)*4 + e1]
                     ^ (uint32_t)f[(m + n1*ROW_ITERS)*4 + e0]
                     ^ (uint32_t)f[(m + n1*ROW_ITERS)*4 + e1];
          x = __shfl_xor_sync(0xffffffffu, x, 16);
          x = __shfl_xor_sync(0xffffffffu, x, 8);
          x = __shfl_xor_sync(0xffffffffu, x, 4);
          x = __shfl_xor_sync(0xffffffffu, x, 2);
          x = __shfl_xor_sync(0xffffffffu, x, 1);
          if (lane == jr*JC + jc) myx = x;
        }
      }
      const int my_jr = lane / JC, my_jc = lane % JC;
      if (my_jr >= JR) return;
      const int jtrib = warp_m*JR + my_jr;
      const int jtcib = warp_n*JC + my_jc;
      if (bi + jtrib < nrow_off && bj + jtcib < ncol_off) {
        const int local_jt = jtrib*CTOFF + jtcib;
        JPS(local_jt, c % 16) = rotl32d(JPS(local_jt, c % 16), 13) ^ myx;
      }
    };

    // GEMM + fold
    FoldMma mma(*mma_smem, thread_idx, warp_idx, lane);
    mma(gemm_k_iterations, accum, itA, itB, accum, fold_every, fold);

    // Jackpot check
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

    // Sync before next tile (reuse shared memory)
    __syncthreads();

    // Optional: early exit if winner found (saves work but adds divergence)
    // if (*win_flag) break;  // Disabled for now
  }
#undef JPS
}

// ---- Override host launch code to use persistent grid ----
#undef tc_jackpot_search  // Undefine the one from tc_cutlass_v2.cu

extern "C" int tc_jackpot_search(
    const signed char* A, const signed char* Bt, int m, int n, int k, int rank,
    const int* pat_rows, const int* pat_cols, int h, int w,
    const int* row_off, const int* col_off, int nrow_off, int ncol_off,
    const unsigned char* a_noise_seed32, const unsigned char* bound_le32,
    int* out_rt, int* out_ct, void* stream_)
{
  cudaStream_t s = stream_ ? *((cudaStream_t*)stream_) : g_search_stream;
  DevBufs& B = g_bufs;
  if (!ensure_dev_bufs(m,n,k,h,w,nrow_off,ncol_off)) return -1;

  g_inflight_tiles = (size_t)((nrow_off + RTOFF - 1)/RTOFF) * ((ncol_off + CTOFF - 1)/CTOFF);
  g_inflight_work = (size_t)nrow_off * ncol_off * k * 256 * 2;

  cudaStreamWaitEvent(s, g_gather_evt);
  cudaMemcpyAsync(B.dA,A,(size_t)m*k,cudaMemcpyHostToDevice,s);
  cudaMemcpyAsync(B.dBt,Bt,(size_t)n*k,cudaMemcpyHostToDevice,s);
  cudaMemcpyAsync(B.dpr,pat_rows,h*4,cudaMemcpyHostToDevice,s);
  cudaMemcpyAsync(B.dpc,pat_cols,w*4,cudaMemcpyHostToDevice,s);
  cudaMemcpyAsync(B.droff,row_off,nrow_off*4,cudaMemcpyHostToDevice,s);
  cudaMemcpyAsync(B.dcoff,col_off,ncol_off*4,cudaMemcpyHostToDevice,s);
  uint32_t kw[8],bw[8]; words_from_le32(a_noise_seed32,kw); words_from_le32(bound_le32,bw);
  cudaMemcpyAsync(B.dk,kw,32,cudaMemcpyHostToDevice,s);
  cudaMemcpyAsync(B.db,bw,32,cudaMemcpyHostToDevice,s);
  cudaMemsetAsync(B.df,0,4,s);

  gather_rows<<<nrow_off*h, 256, 0, s>>>(B.dA, B.dAp, k, B.droff, B.dpr, h, nrow_off);
  gather_rows<<<ncol_off*w, 256, 0, s>>>(B.dBt, B.dBtp, k, B.dcoff, B.dpc, w, ncol_off);
  cudaEventRecord(g_gather_evt, s);

  // ---- PERSISTENT GRID: query SM count, compute persistent block count ----
  static int num_sm = 0;
  if (num_sm == 0) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
  }

  #ifdef SMALL_TILE
  const int occupancy = 2;  // 2 TB/SM with small tile
  #else
  const int occupancy = 1;  // 1 TB/SM with standard tile
  #endif

  const int persistent_blocks = num_sm * occupancy;
  const int nbm = (nrow_off + RTOFF - 1) / RTOFF;
  const int nbn = (ncol_off + CTOFF - 1) / CTOFF;

  dim3 grid(persistent_blocks, 1);  // 1D grid, loop inside kernel
  size_t smem_bytes = sizeof(typename FoldMma::Base::SharedStorage) + (size_t)NJT*17*4;

  static bool attr_set=false;
  if (!attr_set) {
    cudaError_t ae = cudaFuncSetAttribute(tc_cutlass_jackpot,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    if (ae!=cudaSuccess) fprintf(stderr,"tc_cutlass_persistent: smem attr (%zu B) err %s\n", smem_bytes, cudaGetErrorString(ae));
    attr_set=true;
  }

  if (g_miner_verbose) {
    fprintf(stderr, "tc_cutlass_persistent: launching %d persistent blocks (SM=%d, occ=%d) for %d×%d=%d tiles\n",
            persistent_blocks, num_sm, occupancy, nbm, nbn, nbm*nbn);
  }

  cudaEventRecord(g_se0, s);
  tc_cutlass_jackpot<<<grid, TPB, smem_bytes, s>>>(
      (const int8_t*)B.dAp,(const int8_t*)B.dBtp,k,rank,nrow_off,ncol_off,B.dk,B.db,B.df,B.dr,B.dc,
      nbm, nbn);  // NEW: pass grid dimensions
  cudaEventRecord(g_se1, s);

  cudaError_t le = cudaGetLastError();
  if (le!=cudaSuccess) {
    fprintf(stderr,"tc_cutlass_persistent: LAUNCH err %s (tpb=%d, grid=%d, smem=%zuB)\n",
            cudaGetErrorString(le), TPB, grid.x, smem_bytes);
    return -1;
  }
  return 0;
}

// tc_search_wait is unchanged, reuse from tc_cutlass_v2.cu
