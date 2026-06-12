// tc_cutlass_v2.cu — Day5 of PLAN_cutlass.md: ONE fused CUTLASS multistage
// mainloop over the FULL K, with the jackpot fold hooked between k-iterations.
//
// v1 (tc_cutlass_v1.cu, 72-73 TH/s, POSTCHECK ok=1) ran one MmaMultistage per
// rank-chunk and drained the cp.async pipeline between chunks:
//     16 chunks x (prologue refill + cp_async_wait<0> + syncthreads)
// killing the 3-stage overlap 16x per draw — measured 56% of the 128.6 TH/s
// roofline. v2 removes ALL of it:
//
//   * FoldMmaMultistage = byte-for-byte copy of CUTLASS MmaMultistage
//     (gemm/threadblock/mma_multistage.h, v3.5.1) with ONE addition: gemm_iters
//     takes a fold callback invoked every `fold_every` mac_loop_iter calls.
//     Each mac_loop_iter consumes exactly one TB-K=64 slice in order, so after
//     call 4c+4 the accumulators hold precisely chunks 0..c — the fold for
//     chunk c reads them there. The callback is warp-local registers + shfl +
//     a lane-0 store to the JPS smem region (disjoint from the A/B stage
//     buffers): NO barrier, NO pipeline drain. The mainloop's cp.async
//     pipeline stays full across all 16 chunk boundaries.
//   * One prologue + one drain per DRAW (vs 16 of each in v1).
//
// Accumulators are cumulative across chunks (never cleared) — identical
// semantics to v1 / tc_deep_pipeline / the official verifier (POSTCHECK ok=1).
//
// Template plumbing: MmaMultistage's PipeState is private, so subclassing
// can't reuse its loop. Rebind<> pattern-matches DefaultMma::ThreadblockMma
// to re-instantiate our copied class with CUTLASS's own deduced iterators /
// policy / cache-ops — zero hand-written iterator types to drift out of sync.
//
// ABI: same extern "C" tc_jackpot_search as tc_block / v1 — drop-in link.
//
// Build (box): nvcc -O3 -arch=sm_86 -std=c++17 -I$HOME/cutlass/include \
//              -c src/tc_cutlass_v2.cu -o build/tc_cutlass.o
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

extern "C" int g_miner_verbose;  // defined in miner_main.cpp (or plainproof_gen for standalone)

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"

// ---- CUTLASS recipe (identical to v1: the 128.6 TH/s bench config) ----------
#ifdef SMALL_TILE
using TBShape   = cutlass::gemm::GemmShape<64, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
#else
using TBShape   = cutlass::gemm::GemmShape<128, 256, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
#endif
using InstShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kStages = 3;

using DefaultMmaT = cutlass::gemm::threadblock::DefaultMma<
    int8_t,  cutlass::layout::RowMajor,    16,   // A  (gathered A',  lda = k)
    int8_t,  cutlass::layout::ColumnMajor, 16,   // B  (gathered Bt', ldb = k)
    int32_t, cutlass::layout::RowMajor,          // C  (never materialized)
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TBShape, WarpShape, InstShape,
    // satfinite: the only s8 16x8x32 arch::Mma specialization; saturation can
    // never trigger here (|acc| <= k*64*64 = 1.7e7 << 2^31)
    kStages, cutlass::arch::OpMultiplyAddSaturate>;

// =============================================================================
// FoldMmaMultistage — CUTLASS MmaMultistage (v3.5.1) + a fold hook in the
// mainloop. Everything except gemm_iters/operator() is copied verbatim.
// =============================================================================
template <
    typename Shape_, typename IteratorA_, typename SmemIteratorA_,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB_, typename SmemIteratorB_,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC_, typename LayoutC_, typename Policy_, int Stages,
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    typename Enable = bool>
class FoldMmaMultistage :
  public cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, Stages> {
public:
  using Base = cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, Stages>;
  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  using FragmentC = typename Policy::Operator::FragmentC;
  using Operator = typename Policy::Operator;
  using ArchTag = cutlass::arch::Sm80;

  struct Detail {
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;
    static int const kStages = Stages;
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
    static bool const kStagedAccumulation =
        cutlass::arch::detail::UseStagedAccumulation<Operator>::value;
  };

private:
  struct PipeState {
    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
    FragmentC tmp_accum_;
    WarpLoadedFragmentA warp_loaded_frag_A_[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];
    WarpLoadedFragmentB warp_loaded_frag_B_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];
  };

  Operator warp_mma_;
  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  int smem_write_stage_idx_;
  int smem_read_stage_idx_;

public:
  CUTLASS_DEVICE
  FoldMmaMultistage(typename Base::SharedStorage &shared_storage,
                    int thread_idx, int warp_idx, int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        smem_write_stage_idx_(0),
        smem_read_stage_idx_(0) {
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE
  void advance_smem_read_stage() {
    ++smem_read_stage_idx_;
    if (smem_read_stage_idx_ == Base::kStages) {
      this->warp_tile_iterator_A_.add_tile_offset({0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset({-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      smem_read_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void advance_smem_write_stage(IteratorA &iterator_A, IteratorB &iterator_B) {
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});
    ++smem_write_stage_idx_;
    if (smem_write_stage_idx_ == Base::kStages) {
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get());
        int const kSrcBytes = cutlass::sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();
          if (SharedMemoryClear == cutlass::gemm::SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr, iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr, iterator_A.valid());
          }
          ++iterator_A;
        }
        ++this->smem_iterator_A_;
      }
    }
    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get());
        int const kSrcBytes = cutlass::sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B.get();
          if (SharedMemoryClear == cutlass::gemm::SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr, iterator_B.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr, iterator_B.valid());
          }
          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  CUTLASS_DEVICE
  void prologue(IteratorA &iterator_A, IteratorB &iterator_B, int &gemm_k_iterations) {
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);
      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get());
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes = cutlass::sizeof_bits<typename IteratorA::Element>::value *
                                IteratorA::ThreadMap::kElementsPerAccess /
                                IteratorA::kAccessesPerVector / 8;
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator_A.get(), iterator_A.valid());
          ++iterator_A;
        }
        ++this->smem_iterator_A_;
      }
      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get());
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes = cutlass::sizeof_bits<typename IteratorB::Element>::value *
                                IteratorB::ThreadMap::kElementsPerAccess /
                                IteratorB::kAccessesPerVector / 8;
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B.get(), iterator_B.valid());
          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
      advance_smem_write_stage(iterator_A, iterator_B);
      cutlass::arch::cp_async_fence();
    }
  }

  CUTLASS_DEVICE
  void gmem_wait() {
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();
  }

  CUTLASS_DEVICE
  void mac_loop_iter(PipeState &pipe_state, FragmentC &accum,
                     IteratorA &iterator_A, IteratorB &iterator_B,
                     int &gemm_k_iterations) {
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;
      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      if (warp_mma_k > 0) {
        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }

      if (Detail::kStagedAccumulation) {
        warp_mma_(pipe_state.tmp_accum_,
                  pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                  pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                  pipe_state.tmp_accum_);
        if (warp_mma_k == 0) {
          cutlass::plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(accum,
                  pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                  pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                  accum);
      }

      if (warp_mma_k < Base::kWarpGemmIterations - 1) {
        int group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;
        copy_tiles_and_advance(iterator_A, iterator_B,
                               group_start_iteration_A, group_start_iteration_B);
      }

      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
        int group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
        copy_tiles_and_advance(iterator_A, iterator_B,
                               group_start_iteration_A, group_start_iteration_B);
        cutlass::arch::cp_async_fence();
        gmem_wait();
        advance_smem_write_stage(iterator_A, iterator_B);
        advance_smem_read_stage();
        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
      }

      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }
    }
  }

  /// THE ONLY FUNCTIONAL CHANGE vs CUTLASS: fold(accum, chunk) fires every
  /// `fold_every` mainloop iterations, with the pipeline left untouched.
  /// mac_loop_iter #i consumes TB-K slice #i in order, so at the fold point
  /// the accumulators contain exactly chunks 0..chunk — cumulative, as the
  /// jackpot spec requires. fold must be warp-local (registers/shfl/lane-0
  /// stores): no barriers, or it would deadlock vs gmem_wait's syncthreads.
  template <typename FoldFn>
  CUTLASS_DEVICE
  void gemm_iters_fold(int gemm_k_iterations, FragmentC &accum,
                       IteratorA &iterator_A, IteratorB &iterator_B,
                       int fold_every, FoldFn fold) {
    PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;

    warp_mma_.transform(pipe_state.warp_transformed_frag_A_[0],
                        pipe_state.warp_transformed_frag_B_[0],
                        pipe_state.warp_loaded_frag_A_[0],
                        pipe_state.warp_loaded_frag_B_[0]);

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }

    int iter = 0;
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      mac_loop_iter(pipe_state, accum, iterator_A, iterator_B, gemm_k_iterations);
      ++iter;
      if (iter % fold_every == 0) {
        if (Detail::kStagedAccumulation) {   // never taken for s8 (compile-time)
          cutlass::plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
        fold(accum, iter / fold_every - 1);
      }
    }

    if (Detail::kStagedAccumulation) {
      cutlass::plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }

  template <typename FoldFn>
  CUTLASS_DEVICE
  void operator()(int gemm_k_iterations, FragmentC &accum,
                  IteratorA iterator_A, IteratorB iterator_B,
                  FragmentC const &src_accum, int fold_every, FoldFn fold) {
    prologue(iterator_A, iterator_B, gemm_k_iterations);
    gmem_wait();
    accum = src_accum;
    gemm_iters_fold(gemm_k_iterations, accum, iterator_A, iterator_B, fold_every, fold);
  }
};

// Rebind DefaultMma's fully-deduced MmaMultistage instantiation onto our fork.
template <typename T> struct RebindFold;
template <typename S, typename IA, typename SIA, cutlass::arch::CacheOperation::Kind CA,
          typename IB, typename SIB, cutlass::arch::CacheOperation::Kind CB,
          typename EC, typename LC, typename P, int St,
          cutlass::gemm::SharedMemoryClearOption Cl, typename En>
struct RebindFold<cutlass::gemm::threadblock::MmaMultistage<S, IA, SIA, CA, IB, SIB, CB, EC, LC, P, St, Cl, En>> {
  using type = FoldMmaMultistage<S, IA, SIA, CA, IB, SIB, CB, EC, LC, P, St, Cl, En>;
};

using Mma     = typename DefaultMmaT::ThreadblockMma;   // for types/SharedStorage
using FoldMma = typename RebindFold<Mma>::type;
static_assert(sizeof(typename FoldMma::Base::SharedStorage) ==
              sizeof(typename Mma::Base::SharedStorage), "smem layout must match");

// fold geometry is specialized to the real config jackpot tile (h=8, w=16)
constexpr int BM    = TBShape::kM;     // 128 or 64
constexpr int BN    = TBShape::kN;     // 256 or 128
constexpr int H     = 8,  W = 16;
constexpr int RTOFF = BM / H;          // jackpot-tile rows per block
constexpr int CTOFF = BN / W;          // jackpot-tile cols per block
constexpr int NJT   = RTOFF * CTOFF;
constexpr int TPB   = 32 * Mma::WarpCount::kM * Mma::WarpCount::kN * Mma::WarpCount::kK;

// Compile-time fold parameters — derived from the warp shape, not hardcoded.
// A warp covers WM x WN accumulators; each m16n8k32 mma produces a 16x8 sub-tile.
constexpr int WM = WarpShape::kM;           // 64 or 32
constexpr int WN = WarpShape::kN;           // 64 or 64
constexpr int ROW_ITERS = WM / 16;          // 4 or 2  (mma sub-tile rows per warp)
constexpr int JR = WM / H;                  // 8 or 4  (jackpot-tile row-bands per warp)
constexpr int JC = WN / W;                  // 4 or 4  (jackpot-tile col-bands per warp)
static_assert(JR * JC == 32 || JR * JC == 16,
              "warp must cover 32 or 16 jackpot tiles (64x64 or 32x64)");

// ---- blake3 jackpot hash + bound check (unchanged) --------------------------
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

// ---- the kernel: ONE fused mainloop over full K, fold hooked in -------------
__global__ void __launch_bounds__(TPB, 1) tc_cutlass_jackpot(
    const int8_t* __restrict__ Ap, const int8_t* __restrict__ Btp,
    int k, int rank, int nrow_off, int ncol_off,
    const uint32_t* __restrict__ key, const uint32_t* __restrict__ bound,
    int* __restrict__ win_flag, int* __restrict__ win_rt, int* __restrict__ win_ct)
{
  extern __shared__ __align__(16) char smem_raw[];
  auto* mma_smem = reinterpret_cast<typename FoldMma::Base::SharedStorage*>(smem_raw);
  uint32_t* jp_sh = reinterpret_cast<uint32_t*>(smem_raw + sizeof(typename FoldMma::Base::SharedStorage));
// stride 17 (not 16): the lane-distributed fold store writes 32 tiles' word c
// in ONE instruction; tile stride 16 words would land on 2 banks (16-way
// conflict), 17 spreads it to 8 banks (4-way). +1 KB smem, still under cap.
#define JPS(t,q) jp_sh[(t)*17+(q)]

  const int thread_idx = threadIdx.x;
  const int warp_idx   = threadIdx.x >> 5;
  const int lane       = threadIdx.x & 31;
  // CUTLASS MmaBase warp-raking convention (mma_base.h): m fastest, then n.
  const int warp_m = warp_idx % Mma::WarpCount::kM;
  const int warp_n = warp_idx / Mma::WarpCount::kM;

  // GROUPED RASTER (the L2 swizzle device::Gemm gets from its
  // ThreadblockSwizzle and a naive 2D launch loses): with row-major raster a
  // wave of ~82 concurrent TBs sits on ONE row strip and streams 82 distinct
  // 1MB B panels (~83MB/wave, B re-read once per strip = ~525GB DRAM/draw =
  // the measured 880ms wall). Remap consecutive block ids column-major inside
  // bands of GROUPM row strips -> a wave covers a ~GROUPMx(82/GROUPM)
  // rectangle, panel working set ~13MB -> ~6x less DRAM traffic.
  // GROUPM=1 degenerates to the plain row-major raster (identity mapping).
#ifndef GROUPM
#define GROUPM 8
#endif
  const int nbm = gridDim.y, nbn = gridDim.x;
  int bm_, bn_;
  {
    int pid    = (int)blockIdx.y * nbn + (int)blockIdx.x;
    int band   = pid / (GROUPM * nbn);          // which band of row strips
    int first  = band * GROUPM;
    int gsz    = (nbm - first < GROUPM) ? (nbm - first) : GROUPM;
    int rem    = pid - first * nbn;
    bm_ = first + rem % gsz;                    // row strip (column-major walk)
    bn_ = rem / gsz;                            // column strip
  }
  const int bi = bm_ * RTOFF;             // first jackpot-tile row of this block
  const int bj = bn_ * CTOFF;             // first jackpot-tile col of this block
  const int M  = nrow_off * H;            // gathered A' rows
  const int N  = ncol_off * W;            // gathered Bt' rows (= B cols)

  for (int t = thread_idx; t < NJT*17; t += blockDim.x) jp_sh[t] = 0;
  __syncthreads();

  typename FoldMma::FragmentC accum;
  accum.clear();

  // ONE iterator pair over the FULL K extent — no per-chunk re-construction
  typename FoldMma::IteratorA::Params paramsA{cutlass::layout::RowMajor(k)};
  typename FoldMma::IteratorB::Params paramsB{cutlass::layout::ColumnMajor(k)};
  const int kfold = k - (k % rank);
  typename FoldMma::IteratorA itA(paramsA, const_cast<int8_t*>(Ap), {M, kfold}, thread_idx,
                                  cutlass::MatrixCoord(bm_ * BM, 0));
  typename FoldMma::IteratorB itB(paramsB, const_cast<int8_t*>(Btp), {kfold, N}, thread_idx,
                                  cutlass::MatrixCoord(0, bn_ * BN));

  // warp-local fold, LANE-DISTRIBUTED (ncu: the previous lane-0 serial RMW
  // chain was the top stall source — 55.9% of issue gaps were "execution pipe
  // oversubscribed", 31 lanes predicated off while lane 0 did 32 serial smem
  // RMWs). A warp covers exactly JR*JC jackpot tiles and a warp has exactly 32
  // lanes. For baseline (64×64 warp, JR=8, JC=4) each lane owns 1 tile; for
  // SMALL_TILE (32×64 warp, JR=4, JC=4) the first 16 lanes own 1 tile each and
  // lanes 16-31 are idle. Redux result of tile #it is claimed by lane #it into
  // a register; after the loop ALL active lanes do the rotl13-RMW of their own
  // tile's transcript word in ONE parallel instruction.
  auto fold = [&](typename FoldMma::FragmentC const& acc, int c) {
    const int32_t* f = reinterpret_cast<const int32_t*>(acc.data());
    uint32_t myx = 0;                           // lane's claimed tile XOR
    #pragma unroll
    for (int jr = 0; jr < JR; ++jr) {           // JR row-bands within warp's WM rows
      const int m = jr >> 1, half = jr & 1;
      const int e0 = half*2, e1 = half*2 + 1;
      #pragma unroll
      for (int jc = 0; jc < JC; ++jc) {         // JC col-bands within warp's WN cols
        const int n0 = jc*2, n1 = n0 + 1;
        uint32_t x = (uint32_t)f[(m + n0*ROW_ITERS)*4 + e0]
                   ^ (uint32_t)f[(m + n0*ROW_ITERS)*4 + e1]
                   ^ (uint32_t)f[(m + n1*ROW_ITERS)*4 + e0]
                   ^ (uint32_t)f[(m + n1*ROW_ITERS)*4 + e1];
        x = __reduce_xor_sync(0xffffffffu, x);  // broadcast to all lanes
        if (lane == jr*JC + jc) myx = x;        // tile #it claimed by lane #it
      }
    }
    // lane L owns tile (jr=L/JC, jc=L%JC); only active lanes write.
    const int my_jr = lane / JC, my_jc = lane % JC;
    if (my_jr >= JR) return;                    // idle lane (SMALL_TILE: lanes 16-31)
    const int jtrib = warp_m*JR + my_jr;
    const int jtcib = warp_n*JC + my_jc;
    if (bi + jtrib < nrow_off && bj + jtcib < ncol_off) {
      const int local_jt = jtrib*CTOFF + jtcib;
      JPS(local_jt, c % 16) = rotl32d(JPS(local_jt, c % 16), 13) ^ myx;
    }
  };

  FoldMma mma(*mma_smem, thread_idx, warp_idx, lane);
  const int gemm_k_iterations = kfold / FoldMma::Shape::kK;
  const int fold_every        = rank / FoldMma::Shape::kK;
  mma(gemm_k_iterations, accum, itA, itB, accum, fold_every, fold);
  // gemm_iters_fold tail already drained cp.async + syncthreads — JPS is coherent

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

// ---- host wrapper: identical ABI/flow to v1 ---------------------------------
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
    if (g_miner_verbose)
        fprintf(stderr,"tc_cutlass: persistent device buffers allocated (%zu MB)\n",
                ((size_t)m*k+(size_t)n*k+apk+bpk)/1024/1024);
  }
  return true;
}

// Expose the persistent device buffers so gpu_prep.cu can generate the draw
// directly in them (GPU-resident pipeline: no host matrices, no 1GB H2D).
extern "C" int tc_alloc_bufs(int m,int n,int k,int h,int w,int nrow_off,int ncol_off,
                             signed char** dA, signed char** dBt)
{
  if (!ensure_dev_bufs(m,n,k,h,w,nrow_off,ncol_off)) return -1;
  *dA = g_bufs.dA; *dBt = g_bufs.dBt;
  return 0;
}

// ---- async launch/wait split -------------------------------------------------
// The search kernel only reads the GATHERED panels dAp/dBtp; gpu_prep writes
// dA/dBt. So once the gather kernels of draw N are done, prep for draw N+1 can
// overwrite dA/dBt UNDER the in-flight search — that's the whole overlap. The
// host contract: tc_search_launch() returns immediately after queueing
// gather+search on a private non-blocking stream; tc_search_wait() blocks for
// the result. g_gather_evt (exposed via tc_gather_done_event) is recorded
// after the gathers so gpu_prep's stream can order itself behind them.
static cudaStream_t g_search_stream = nullptr;
static cudaEvent_t  g_gather_evt = nullptr, g_se0 = nullptr, g_se1 = nullptr;
static size_t g_inflight_tiles = 0;
static double g_inflight_work = 0;

static bool ensure_search_stream() {
  if (g_search_stream) return true;
  if (cudaStreamCreateWithFlags(&g_search_stream, cudaStreamNonBlocking) != cudaSuccess) return false;
  cudaEventCreateWithFlags(&g_gather_evt, cudaEventDisableTiming);
  cudaEventCreate(&g_se0); cudaEventCreate(&g_se1);
  return true;
}

// gpu_prep.cu weak-links this; non-NULL only after the first launch.
extern "C" void* tc_gather_done_event() { return (void*)g_gather_evt; }

extern "C" int tc_search_launch(
    int m,int n,int k,int rank,
    const int* pat_rows, const int* pat_cols, int h,int w,
    const int* row_off, int nrow_off, const int* col_off, int ncol_off,
    const unsigned char* a_noise_seed32, const unsigned char* bound_le32)
{
  if (rank % FoldMma::Shape::kK) { fprintf(stderr,"tc_cutlass: rank %d not a multiple of %d\n", rank, (int)FoldMma::Shape::kK); return -3; }
  if (h != H || w != W) {
    fprintf(stderr,"tc_cutlass: this geometry needs h=%d w=%d (got h=%d w=%d)\n",H,W,h,w); return -2; }
  if (!ensure_dev_bufs(m,n,k,h,w,nrow_off,ncol_off)) return -1;
  if (!ensure_search_stream()) return -1;
  DevBufs& B = g_bufs;
  cudaStream_t s = g_search_stream;

  g_inflight_tiles = (size_t)nrow_off * ncol_off;
  g_inflight_work  = (double)g_inflight_tiles * h * w * (k - (k % rank));

  // pageable-source async copies stage synchronously -> stack/temp sources OK
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
  cudaEventRecord(g_gather_evt, s);   // prep(N+1) may write dA/dBt after this

  dim3 grid((ncol_off + CTOFF - 1)/CTOFF, (nrow_off + RTOFF - 1)/RTOFF);
  size_t smem_bytes = sizeof(typename FoldMma::Base::SharedStorage) + (size_t)NJT*17*4;
  static bool attr_set=false;
  if (!attr_set) {
    cudaError_t ae = cudaFuncSetAttribute(tc_cutlass_jackpot,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    if (ae!=cudaSuccess) fprintf(stderr,"tc_cutlass: smem attr (%zu B) err %s\n", smem_bytes, cudaGetErrorString(ae));
    attr_set=true;
  }
  cudaEventRecord(g_se0, s);
  tc_cutlass_jackpot<<<grid, TPB, smem_bytes, s>>>(
      (const int8_t*)B.dAp,(const int8_t*)B.dBtp,k,rank,nrow_off,ncol_off,B.dk,B.db,B.df,B.dr,B.dc);
  cudaEventRecord(g_se1, s);
  cudaError_t le = cudaGetLastError();
  if (le!=cudaSuccess) {
    fprintf(stderr,"tc_cutlass: LAUNCH err %s (tpb=%d, grid=%dx%d, smem=%zuB)\n",
            cudaGetErrorString(le), TPB, grid.x, grid.y, smem_bytes);
    return -1;
  }
  return 0;
}

extern "C" int tc_search_wait(int* out_rt, int* out_ct)
{
  DevBufs& B = g_bufs;
  cudaError_t err = cudaStreamSynchronize(g_search_stream);
  if (g_miner_verbose) {
    float ms=0; cudaEventElapsedTime(&ms,g_se0,g_se1);
    double ths = g_inflight_work / (ms * 1e-3) / 1e12;
    fprintf(stderr,
            "tc(cutlass2): TB=%dx%dx%d W=%dx%d s%d T%d FUSED %zu tiles, %.3f ms, %.2f TH/s\n",
            BM, BN, (int)TBShape::kK, WM, WN, kStages, TPB, g_inflight_tiles, ms, ths);
  }
  if (err!=cudaSuccess){ fprintf(stderr,"tc_cutlass: err %s\n",cudaGetErrorString(err)); return -1; }

  int wf=0;
  cudaMemcpy(&wf,B.df,4,cudaMemcpyDeviceToHost);
  if (wf){ cudaMemcpy(out_rt,B.dr,4,cudaMemcpyDeviceToHost); cudaMemcpy(out_ct,B.dc,4,cudaMemcpyDeviceToHost); }
  return wf;
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
  if (!ensure_dev_bufs(m,n,k,h,w,nrow_off,ncol_off)) return -1;
  DevBufs& B = g_bufs;
  // a_noised==NULL => the noised matrices were generated GPU-side directly in
  // dA/dBt (gpu_prep.cu); skip the two 512MB H2D copies.
  if (a_noised)    cudaMemcpy(B.dA,a_noised,(size_t)m*k,cudaMemcpyHostToDevice);
  if (b_noised_t)  cudaMemcpy(B.dBt,b_noised_t,(size_t)n*k,cudaMemcpyHostToDevice);
  int rc = tc_search_launch(m,n,k,rank,pat_rows,pat_cols,h,w,
                            row_off,nrow_off,col_off,ncol_off,
                            a_noise_seed32,bound_le32);
  if (rc) return rc;
  return tc_search_wait(out_rt, out_ct);
}
