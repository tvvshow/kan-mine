# Phase 8: Persistent Scheduler 设计文档

**日期**：2026-06-12  
**目标**：在 v2.0.0 基础上额外获得 5-15% 性能提升  
**难度**：高（需要重写主循环）  
**前置条件**：v2.0.0 已测试并达到 300+ TH/s

---

## 原理

### 当前调度（Standard Grid-Stride）

```cpp
// 启动：每个 tile 一个 threadblock
dim3 grid((ncol_off + CTOFF - 1)/CTOFF, (nrow_off + RTOFF - 1)/RTOFF);
tc_cutlass_jackpot<<<grid, TPB, ...>>>(...)

// 内核：处理一个固定的 tile
__global__ void tc_cutlass_jackpot(...) {
  const int bm_ = blockIdx.y;  // 固定行 strip
  const int bn_ = blockIdx.x;  // 固定列 strip
  
  // 处理 tile (bm_, bn_)
  FoldMma mma(...);
  mma(gemm_k_iterations, accum, itA, itB, accum, fold_every, fold);
  
  // jackpot 检查
  ...
}
```

**Real config 规模**：
- M = N = 131072, BM = 128, BN = 256
- grid = (512, 1024) = **524,288 个 threadblock**
- 4090: 128 SM × 1 TB/SM = 128 个并发 TB
- 执行 524288 / 128 = **4096 wave**

### Persistent Scheduler

```cpp
// 启动：固定数量的 threadblock（等于硬件并发能力）
const int persistent_blocks = num_sm * occupancy;  // 128 或 256
tc_cutlass_persistent<<<persistent_blocks, TPB, ...>>>(...)

// 内核：每个 TB 循环处理多个 tile
__global__ void tc_cutlass_persistent(...) {
  const int total_tiles = nbm * nbn;  // 524288
  
  for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
    // 将 tile_idx 映射到 (bm_, bn_)
    int bm_, bn_;
    tile_idx_to_block_coord(tile_idx, nbm, nbn, bm_, bn_);
    
    // 处理 tile (bm_, bn_)
    accum.clear();
    IteratorA itA(..., MatrixCoord(bm_ * BM, 0));
    IteratorB itB(..., MatrixCoord(0, bn_ * BN));
    mma(gemm_k_iterations, accum, itA, itB, accum, fold_every, fold);
    
    // jackpot 检查
    ...
    
    __syncthreads();  // 准备下一个 tile
  }
}
```

**每个 TB 处理**：524288 / 128 = **4096 个 tile**

---

## 收益分析

### 1. 消除 Kernel Launch Overhead

当前：每个 draw 启动 1 个 kernel，但调度 524k 个 TB 有隐式开销。  
Persistent：只启动 128 个 TB，循环内部是 **纯设备端调度**，无 host-device 交互。

**预期收益**：~2-5%（launch overhead 通常 <10μs，但 524k 个 TB 的调度累积有影响）

### 2. 消除尾部不平衡

**当前问题**：
- 最后一个 wave 可能只有部分 TB 有工作
- 例如：524288 = 4096 × 128，正好整除 → 无尾部浪费
- 但如果 grid 不是 128 的倍数（其他配置），最后几个 SM 空闲

**Persistent 优势**：
- 动态分配：先完成的 TB 立即获取下一个 tile
- 无论 tile 数量，所有 TB 都持续工作到最后

**预期收益**：~1-3%（real config 正好整除，收益有限；但 SMALL_TILE 后 grid = 2097152 = 8192 × 256，收益更明显）

### 3. 更好的 L2 复用

**当前**：
- 每个 TB 处理 1 个 tile，完成后退出
- 下一个 tile 可能被不同的 SM 执行 → L2 cache 冷启动

**Persistent**：
- 同一个 TB 处理多个 **相邻** tile（如果按顺序遍历）
- 连续的 (bm_, bn_) 和 (bm_, bn_+1) 共享 A 的行 panel
- L2 数据留存时间更长

**预期收益**：~5-10%（GROUPM 已经优化了 L2，但 persistent 的时间局部性更强）

### 总预期收益：5-15%

---

## 实现挑战

### 1. Tile ID 映射

需要将线性 `tile_idx` 映射到 2D `(bm_, bn_)`，**同时保持 GROUPM 策略**。

**当前 GROUPM 映射（行优先 band）**：
```cpp
int pid    = blockIdx.y * nbn + blockIdx.x;
int band   = pid / (GROUPM * nbn);
int first  = band * GROUPM;
int gsz    = (nbm - first < GROUPM) ? (nbm - first) : GROUPM;
int rem    = pid - first * nbn;
bm_ = first + rem % gsz;
bn_ = rem / gsz;
```

**Persistent 需要**：
```cpp
__device__ void tile_idx_to_block_coord(
    int tile_idx, int nbm, int nbn, int& bm_, int& bn_) {
  // 保持与 GROUPM 相同的映射逻辑
  int pid = tile_idx;
  int band = pid / (GROUPM * nbn);
  // ... 同上 ...
}
```

### 2. Iterator 重新初始化

每个 tile 都需要新的 `IteratorA` 和 `IteratorB`：

```cpp
for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
  tile_idx_to_block_coord(tile_idx, nbm, nbn, bm_, bn_);
  
  // 每个 tile 都需要重新构造 iterator
  typename FoldMma::IteratorA itA(paramsA, Ap, {M, kfold}, thread_idx,
                                  cutlass::MatrixCoord(bm_ * BM, 0));
  typename FoldMma::IteratorB itB(paramsB, Btp, {kfold, N}, thread_idx,
                                  cutlass::MatrixCoord(0, bn_ * BN));
  
  accum.clear();
  // ... mma ...
}
```

**问题**：Iterator 构造函数可能有非平凡的开销（计算指针偏移、初始化状态）。  
**优化**：如果可能，复用 Params，只更新 ThreadMap。

### 3. Shared Memory 重用

每个 tile 开始前需要：
- 清零 `jp_sh`（jackpot transcript）
- `__syncthreads()` 确保所有线程同步

```cpp
for (int tile_idx = ...) {
  // 清零 jackpot shared memory
  for (int t = thread_idx; t < NJT*17; t += blockDim.x) jp_sh[t] = 0;
  __syncthreads();
  
  // ... mma + jackpot ...
  
  __syncthreads();  // 准备下一个 tile
}
```

### 4. Jackpot 检查分离

**当前**：jackpot 检查在同一个 kernel 内完成。  
**Persistent**：需要在 **每个 tile** 后检查 `win_flag`，如果找到 winner 可以提前退出：

```cpp
for (int tile_idx = ...) {
  // ... mma + fold ...
  
  // jackpot 检查
  for (int t = thread_idx; t < NJT; t += blockDim.x) {
    // ... blake3 + 检查 ...
    if (win) atomicOr(win_flag, 1);
  }
  __syncthreads();
  
  // 提前退出（可选优化）
  if (*win_flag) break;
}
```

---

## 实现计划

### Step 1：创建 `tc_cutlass_persistent.cu`

基于 `tc_cutlass_v2.cu`，修改：
- 将主体逻辑包在 `for (tile_idx = blockIdx.x; ...)` 循环中
- 实现 `tile_idx_to_block_coord()` 函数
- 每个 tile 开始前清零 accum 和 jp_sh
- 每个 tile 后 `__syncthreads()`

### Step 2：修改 `build.sh`

添加编译选项：
```bash
PERSISTENT="${PERSISTENT:-}"
if [ -n "$PERSISTENT" ]; then
  echo "  Using persistent scheduler"
  nvcc ... -c "${ROOT}/src/tc_cutlass_persistent.cu" -o tc_kernel.o
else
  nvcc ... -c "${ROOT}/src/tc_cutlass_v2.cu" -o tc_kernel.o
fi
```

### Step 3：修改 host 调用

在 `tc_cutlass_v2.cu` 的 host 函数中：
```cpp
#ifdef PERSISTENT
  // Persistent scheduler：固定 grid 大小
  int num_sm = 128;  // TODO: query via cudaDeviceGetAttribute
  int occupancy = 1;  // or 2 if SMALL_TILE
  dim3 grid(num_sm * occupancy, 1);
#else
  // Standard scheduler
  dim3 grid((ncol_off + CTOFF - 1)/CTOFF, (nrow_off + RTOFF - 1)/RTOFF);
#endif
```

### Step 4：测试

```bash
# 编译
PERSISTENT=1 ./build.sh

# 测试正确性
./build/plainproof_gen --cfg real --mine --batch 1 2>&1 | grep POSTCHECK
# 必须：POSTCHECK ok=1

# 测试性能
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"
# 预期：如果 v2.0.0 = 200ms/draw，persistent = ~185ms/draw (5-10% 提升)
```

### Step 5：与 v2.0.0 对比

| 配置 | ms/draw | TH/s | 提升 |
|------|---------|------|------|
| v2.0.0 SMALL_TILE=1 GROUPM=16 | ~200 | ~350 | baseline |
| v2.0.0 + PERSISTENT=1 | ~185 | ~380 | +8.6% |

---

## 风险评估

**高风险点**：
1. **Iterator 构造开销**：每个 tile 重新构造可能抵消部分收益
2. **寄存器压力**：循环变量 + 更多状态可能增加寄存器使用 → 降低 occupancy
3. **代码复杂度**：更难调试，回退成本高

**缓解策略**：
1. 保持 `tc_cutlass_v2.cu` 不变，persistent 作为独立文件
2. 编译时选择（`PERSISTENT=1`），默认关闭
3. 充分测试 POSTCHECK（验证正确性）
4. 性能对比：只有确认 >5% 提升才部署

**最坏情况**：无提升或略微下降 → 回退到 v2.0.0，无损失

---

## 优先级判断

**何时实施 Phase 8？**

1. ✅ **前提**：v2.0.0 已测试，达到 300+ TH/s
2. ✅ **条件**：距离 roofline (350 TH/s) 仍有 >10% 差距
3. ✅ **ROI**：5-15% 收益值得 1-2 天开发成本

**何时跳过 Phase 8？**

1. ❌ v2.0.0 已达到 340+ TH/s（97% roofline）→ 边际收益 <5%
2. ❌ v2.0.0 测试失败（性能无提升或下降）→ 需要先解决根本问题
3. ❌ 其他更高 ROI 的优化方向（如 GPU noise，但 v1 已完成）

---

## 下一步

**当前**：等待 v2.0.0 测试结果（`quick_test.sh` on 4090 box）

**如果测试显示**：
- SMALL_TILE = 200ms/draw (~350 TH/s) → **接近 roofline，Phase 8 低优先级**
- SMALL_TILE = 240ms/draw (~290 TH/s) → **仍有 20% 差距，Phase 8 高优先级**
- SMALL_TILE = 280ms/draw (无提升) → **回退，分析 ncu profile，重新评估**

**如果 Phase 8 立即开始**：
1. 创建 `tc_cutlass_persistent.cu`（预计 100-150 行修改）
2. 修改 `build.sh`（+10 行）
3. 测试正确性（POSTCHECK）
4. 对比性能（batch 3）
5. 如果 >5% 提升 → 部署；否则回退

预计时间：**1-2 天**（开发 + 测试）

---

**当前阻塞**：4090 box 网络连接，无法测试 v2.0.0

**临时方案**：Phase 8 设计文档已完成（本文档），代码实现可以并行进行，但 **部署必须等 v2.0.0 测试完成后决策**。
