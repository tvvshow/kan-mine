# Week 1 Day 1: CUTLASS sm80_mma_multistage 架构笔记

## 文件定位
- 主文件：`cutlass/include/cutlass/gemm/collective/sm80_mma_multistage.hpp`
- Kernel entry：`cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp`

## 核心结构：CollectiveMma

### 模板参数
```cpp
template <
  int Stages,                    // Pipeline 深度（我们用 4）
  class TileShape_,              // 例如 Shape<128,256,64> (M,N,K)
  class TiledMma_,               // MMA atom（mma.sync.m16n8k32）
  class GmemTiledCopyA/B_,       // cp.async 组织
  class SmemLayoutAtomA/B_,      // Smem 布局
  class SmemCopyAtomA/B_         // ldmatrix
>
```

### SharedStorage 布局
```cpp
struct SharedStorage {
  cute::array_aligned<ElementA, ...> smem_a;  // (BLK_M, BLK_K, PIPE)
  cute::array_aligned<ElementB, ...> smem_b;  // (BLK_N, BLK_K, PIPE)
};
```
- 3D tensor：最后一维是 pipeline stage
- 我们的配置：(128, 64, 4) 和 (256, 64, 4)

### 关键流程（operator()）

#### 1. Prefetch（L221-228）
```cpp
for (int k_pipe = 0; k_pipe < Stages-1; ++k_pipe) {
  copy(gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));
  copy(gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));
  cp_async_fence();  // <- 异步屏障
  ++k_tile_iter;
}
```
- 预加载 Stages-1 个 tile（例如 3 个）
- cp.async：gmem → smem 异步传输

#### 2. MMA Atom 分区（L235-238）
```cpp
TiledMma tiled_mma;
auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));  // (MMA,MMA_M,MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));  // (MMA,MMA_N,MMA_K)
```
- `thr_mma`：每个线程的 MMA 片段
- `tCrA/tCrB`：寄存器片段（将被 ldmatrix 填充）

## 核心发现

### 1. **没有显式的 producer/consumer warp**
- sm80 用 cp.async（所有 warp 都是 consumer）
- sm90 才有 warp-specialized（producer warp 专门 copy）

### 2. **Pipeline 机制**
- Stages=4 → 4 个 smem buffer 轮转
- 当前读 buffer N，同时异步加载 buffer (N+3)%4

### 3. **我们需要的插入点**
- Mainloop K-loop 在 L250+ （需要继续往下读）
- Fold 需要在每 256 K-steps 后插入（读 accum）

## 下一步
1. 继续读 L250+ 的 mainloop K-loop
2. 找到 MMA 计算位置（gemm() 调用）
3. 确定 accum 访问方式（用于 fold）
4. 看 examples/ 中的 int8 example

## 关键差异：我们 vs CUTLASS
| 项 | CUTLASS sm80 | 我们 tc_imma2 |
|---|---|---|
| Pipeline | cp.async 多级 | cp.async 2 级 |
| Smem 布局 | 3D (M,K,PIPE) | 2D (M,K) × STAGES |
| 线程分区 | cute::Tensor 抽象 | 手写 warp/lane 映射 |
| MMA | TiledMma abstraction | 直接 asm volatile |

**结论**：CUTLASS 的抽象层很厚，直接移植困难。需要先理解 cute::Tensor 语义，或者只抄 mainloop 的**执行逻辑**（预取 + K-loop + barrier），用我们自己的 ldmatrix/mma。
