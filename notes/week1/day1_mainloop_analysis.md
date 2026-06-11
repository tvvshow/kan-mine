# Week 1 Day 1: CUTLASS Mainloop 核心逻辑（续）

## Mainloop 完整流程（L270-339）

### 1. Pipeline 状态
```cpp
int smem_pipe_read  = 0;               // 当前读哪个 buffer
int smem_pipe_write = Stages-1;        // 当前写哪个 buffer（例如 3）
```

### 2. K-loop 结构（L293-339）
```cpp
while (k_tile_count > -(Stages-1)) {      // 继续到所有 tile 处理完
  for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
    
    // === 关键时机：k_block == K_BLOCK_MAX-1 时同步 ===
    if (k_block == K_BLOCK_MAX - 1) {
      cp_async_wait<Stages-2>();          // 等待倒数第2个 tile 加载完
      __syncthreads();                     // <- BARRIER
    }
    
    // === Prefetch 下一个 k_block 的寄存器 ===
    auto k_block_next = (k_block + 1) % K_BLOCK_MAX;
    copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
    copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
    
    // === 启动下一个 K-tile 的 gmem→smem copy ===
    if (k_block == 0) {
      copy(gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
      copy(gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
      cp_async_fence();
      ++k_tile_iter;
      // 推进 pipe 指针
      smem_pipe_write = smem_pipe_read;
      ++smem_pipe_read;
      smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
    }
    
    // === 核心计算：MMA ===
    cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  });
}
```

## 关键发现

### 1. **K_BLOCK_MAX = 寄存器 pipeline 深度**
- `K_BLOCK_MAX = size<2>(tCrA)`（L279）
- 例如 K=64 per tile，如果 mma.sync.m16n8k32，则 K_BLOCK_MAX = 64/32 = 2
- **寄存器级流水**：当前算 k_block，同时 prefetch k_block+1

### 2. **双层 Pipeline**
- **外层（smem）**：Stages 个 buffer（例如 4），cp.async 轮转
- **内层（reg）**：K_BLOCK_MAX 个 k_block，ldmatrix 轮转

### 3. **Barrier 位置**
- 只在 `k_block == K_BLOCK_MAX-1` 时同步（L307）
- **我们的 RSUB=64 优化就是这个**：增大 K_BLOCK_MAX，减少 barrier

### 4. **MMA 调用（L336）**
```cpp
cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
```
- `accum`：累加器（输出）
- `tCrA/tCrB`：当前 k_block 的寄存器片段
- `src_accum`：初始值（通常是 accum 自己，做累加）

## 我们需要插入的 Fold Hook

### 插入点：每 256 K-steps（rank）后
```cpp
// 在 L337 后面加：
if ((current_k_tile * K_per_tile + k_block * 32) % 256 == 0) {
  // === FOLD HOOK ===
  fold_rank_chunk(accum, jp_sh, chunk_idx);
  // accum 在这里可访问（寄存器片段）
}
```

### 问题
- `accum` 是 `cute::Tensor`，不是我们的 `int32_t acc[WSUBM][WSUBN][4]`
- 需要理解 cute::Tensor 的内存布局，或者**放弃 CUTLASS，只抄逻辑**

## 决策点：两条路

### 路径 A：深度集成 CUTLASS
- 学习 cute::Tensor API（数天）
- 修改 CollectiveMma 加 fold hook
- 链接 libcutlass（编译困难）
- **风险**：cute 抽象太厚，可能性能不如手写

### 路径 B：抄逻辑，手写实现（推荐）
- **只抄执行流程**：
  - 预取 Stages-1 个 tile
  - K-loop：ldmatrix → mma → 每 K_BLOCK_MAX 同步
  - Pipeline 轮转逻辑
- **用我们的原语**：
  - 直接 asm volatile mma.sync.m16n8k32
  - 直接 asm volatile ldmatrix
  - `int32_t acc[WSUBM][WSUBN][4]`（清晰的寄存器布局）
- **优势**：
  - 完全控制，易调试
  - 可直接插 fold（访问 acc 数组）
  - 编译简单（单个 .cu 文件）

## 下一步（明天 Day 2）
1. 看 CUTLASS examples/ 中的 int8 GEMM example（理解完整调用链）
2. 决策：A 还是 B？
3. 如果选 B：开始写 `tc_cutlass_base.cu`（基于 tc_imma2，改进 pipeline）

## 当前判断
**倾向路径 B**：CUTLASS 的 cute 抽象对我们的 fold 需求是**负担**，不是帮助。我们需要的是：
- 更深的 pipeline（Stages=4）
- 更大的 K_BLOCK_MAX（RSUB=128？）
- 清晰的累加器访问（fold）
- 可能的 persistent scheduler

这些都能在 tc_imma2 基础上手写实现，比啃 cute API 快。
