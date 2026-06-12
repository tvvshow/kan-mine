# Phase 7 调优方向报告（搁置）

> 2026-06-12，基于 4090 (sm_89) 260 TH/s kernel / 190 TH/s wall 基线

## 当前瓶颈分析

ncu profile (3090，架构类似) 显示：
- **Compute 80.9%** — 已经很高，计算不是主要瓶颈
- **DRAM 16.5%** — grouped raster (GROUPM=8) 使 L2 命中 93.9%，DRAM 不是瓶颈
- **Occupancy 16.7%** — **1 TB/SM** ← 最大瓶颈
  - 240 regs + ~100KB smem → 每个 SM 只能放一个 threadblock
  - 4090 有 128 SM → 最多 128 个并发 TB

## 可调参数

### 1. Threadblock 尺寸 vs Occupancy 权衡

| 配置 | Regs | smem | TB/SM | 预估 TH/s |
|------|------|------|-------|-----------|
| BM=128 BN=256 S3 (当前) | 240 | ~100KB | 1 | 260 |
| BM=128 BN=256 S2 | 240 | ~70KB | 1 | ~240? (less overlap) |
| BM=64 BN=128 S3 | ~130 | ~50KB | 2 | ~350? (2× occupancy) |
| BM=64 BN=128 S2 | ~130 | ~35KB | 2 | ~320? |

### 2. GROUPM 调优 (L2 策略)

4090 L2 = 72MB (vs 3090 的 6MB)：
- 当前 GROUPM=8 对 3090 6MB L2 最优
- 72MB L2 可能支持更大的 band → 更好的 B panel 复用
- 建议扫描 GROUPM ∈ {8, 16, 32, 64}

### 3. Persistent Scheduler

lpminer 使用 `StaticPersistentTileScheduler`：
- gridDim = N_SM，每个 TB 循环 `tile_idx += gridDim`
- 消除 launch overhead + 尾部浪费
- 预估收益：5-15%

### 4. 预期收益总结

| 优化 | 难度 | 预估 kernel 提升 | 预估 wall 提升 |
|------|------|-----------------|---------------|
| GROUPM 扫描 | 低 (改编译参数) | +5-10% | +5-10% |
| TB 尺寸减半 → 2 TB/SM | 中 (改 CUTLASS recipe) | +30-50% | +30-50% |
| Persistent scheduler | 高 (重写 host wrapper) | +5-15% | +5-15% |
| 三者叠加 | — | **~350-400 TH/s** | **~280-350 TH/s** |

## Roofline 参考值

- 4090 纯 int8 GEMM roofline ≈ 350 TH/s (cutlass_int8_bench, 128×256 tile)
- 当前 kernel = 260 = 74% roofline
- 优化后目标 = 90%+ roofline ≈ 315+ TH/s
