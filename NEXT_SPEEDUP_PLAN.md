# Kan / Pearl Miner 下一阶段提速推进方案

**日期**：2026-06-13  
**范围**：`peral/` active codebase  
**状态**：分析方案，不含代码改动  
**目标**：在不牺牲正确性的前提下，继续提升 PRL PoUW 矿工的 kernel-only TH/s 与 pool wall-clock TH/s。  

> **2026-06-14 更新**：A2 的**执行就绪细化版**见 `A2_EXECUTION_PLAN.md`。
> 要点：(1) A1 已落地 `e466fde` 并在 3080Ti 正确性验证；(2) 新增 sm_86 实测 ——
> **fold ≈ 47%（NOFOLD 380ms vs full 722ms / 97.5 TH/s），比 5090 的 40% 更大**，
> 即 fold 是**所有架构**的头号瓶颈、3080Ti 是 A2 的最佳开发盒子；(3) A2（移除 JPS
> 共享转录）是架构无关的结构性提速，本文 §A2/§A3 的设计不变，执行步骤与 gate 见新文件。

---

## 0. 背景与当前结论

当前主线已经从早期 `tc_block.cu` / WMMA fallback 的约 **30 TH/s**，推进到：

- `src/tc_cutlass_v2.cu`：CUTLASS fused full-K int8 search kernel；
- `src/gpu_prep.cu`：GPU-resident RNG + BLAKE3 commitments + noise pipeline；
- `plainproof_gen.cpp` / `miner_main.cpp`：统一使用 SRBMiner-MULTI / lpminer / pool 对齐的 **TH/s** 口径；
- 真池结果已有 **accepted share** 验证记录。

最新 5090 实测摘要：

| 项目 | 结果 | 说明 |
|---|---:|---|
| correctness | `POSTCHECK ok=1` | grid-stride / CUTLASS v2 正确性已确认 |
| pool shares | `11 accepted / 0 rejected` | 真池验证通过 |
| full kernel | `199.5-200.8 ms/draw` ≈ `350-353 TH/s` | 当前主线性能 |
| pool wall-clock | `295-333 TH/s` | 包含 prep / gather / driver 开销 |
| pure int8 GEMM same tile | `161.4 ms/draw` ≈ `436 TH/s` | 当前 tile 形状的 GEMM roofline |
| `PROFILE_NOFOLD` | `115.3-123.1 ms/draw` | 说明 fold 是大头；注意可能有 DCE，只能作下界参考 |
| `PROFILE_NOJACKPOT` | `195.2-199.6 ms/draw` | 末尾 jackpot BLAKE3 只占约 2-4ms |
| shfl 替代 redux | `223.0 ms/draw` ≈ `314 TH/s` | `redux.sync` 修复实测约 +11% |

核心判断：

```text
当前继续提速的最大明确瓶颈不是 CPU、H2D、final jackpot hash、
SMALL_TILE、KSTAGES 或 persistent scheduler，而是 tc_cutlass_v2.cu
mainloop 中每个 rank chunk 后的 fold callback。
```

---

## 1. 速度口径与正确性红线

所有面向用户、benchmark、文档、pool 对比的速度必须统一为：

```text
TH/s = tiles * rows_pattern_size * cols_pattern_size * dot_len / seconds / 1e12
dot_len = k - (k % rank)
```

REAL config：

```text
m = n = 131072
k = 4096
rank = 256
rows_pattern_size = 8
cols_pattern_size = 16
tiles = 134,217,728

work_per_draw =
134,217,728 * 8 * 16 * 4096
= 70.37e12 PRL-work
```

必须区分：

| 速度类型 | 含义 |
|---|---|
| kernel-only TH/s | 只计 GPU search kernel，例如 `tc(cutlass2): ... FUSED ... TH/s` |
| end-to-end TH/s | `MINE done` draw loop 总耗时 |
| 60s window TH/s | 对齐 SRBMiner-MULTI UI 的 60 秒窗口 |
| pool wall-clock TH/s | 真池运行实际统计，最接近收益 |

正确性门槛：

```text
任何 kernel / prep / fold 改动：
1. 必须 POSTCHECK ok=1；
2. 上生产前必须尽量跑 official verifier VALID；
3. 真池短测必须 accepted > 0 且 rejected = 0；
4. 速度提升不得以错误 proof / pool rejected 为代价。
```

---

## 2. 已证伪或低 ROI 的方向

### 2.1 不再主攻 `tc_block.cu` / WMMA fallback

`tc_block.cu` 是 CUTLASS 缺失时的 live fallback，约 **30 TH/s**。  
它的价值是稳定、正确、保底，不是继续冲 300+ TH/s 的主线。

### 2.2 不再主攻旧手写 IMMA / WMMA

旧 `tc_imma*` / `tc_deep_pipeline.cu` 已完成大量验证价值，但当前主线已经是 CUTLASS v2。  
继续在旧 IMMA/WMMA 上追 CUTLASS，ROI 低、正确性风险高。

### 2.3 `SMALL_TILE` 在 5090 上已证伪

早期假设：

```text
TB 128×256 -> 64×128
regs/smem 减半
occupancy 提高
```

最新实测结论：

```text
SMALL_TILE / KSTAGES / TC_PERSIST 在 5090 上不应继续盲试。
```

小 tile 的 occupancy 收益不足以抵消 tensor-core efficiency / L2 locality / scheduling 损失。

### 2.4 `KSTAGES=4/5` 当前受 shared memory 限制

当前 baseline：

```text
TB 128×256×64
KSTAGES=3
CUTLASS smem 约 72KB
JPS transcript smem 约 17KB
总计约 90KB
```

若直接 `KSTAGES=4`：

```text
CUTLASS smem 约 96KB
+ JPS 约 17KB
= 约 113KB
```

在 5090 consumer 配置上已超 shared memory cap，launch 失败。  
但如果后续移除或压缩 JPS shared transcript，`KSTAGES=4` 可重新进入测试矩阵。

### 2.5 `TC_PERSIST` 是 per-GPU 小优化，不是主线

当前结果：

| GPU | 结果 |
|---|---|
| L40 | standard `387.4ms` vs `TC_PERSIST=1 375.5ms`，约 `+3.2%` |
| 5090 | 无效，不建议部署 |

保留 A/B 工具，但不要把它当主要提速方向。

### 2.6 final jackpot BLAKE3 不是瓶颈

`PROFILE_NOJACKPOT` 只从约 `200ms` 降到 `195-199ms`。  
末尾每 tile 的 `jackpot_blake3 + bound compare + atomicCAS` 只占约 **2-4ms/draw**，不是当前主攻对象。

---

## 3. 主瓶颈：fold callback

`tc_cutlass_v2.cu` 当前结构：

```text
FULL-K CUTLASS MmaMultistage mainloop
每 rank chunk 边界调用 fold callback
accumulators cumulative across chunks
fold callback 更新 jackpot transcript
mainloop 完成后对 transcript 做 jackpot_blake3 + bound check
```

REAL config：

```text
k = 4096
rank = 256
TBShape::kK = 64
gemm_k_iterations = 4096 / 64 = 64
fold_every = 256 / 64 = 4
fold callbacks per draw = 64 / 4 = 16
```

baseline tile：

```text
TB = 128×256
Warp = 64×64
h = 8
w = 16
JR = 8
JC = 4
NJT per block = 256 jackpot tiles
每 warp = 32 jackpot tiles
每 lane 恰好可对应 1 个 jackpot tile
```

当前 fold 热路径近似：

```text
每 warp 每 fold：
  32 个 jackpot tile
  每个 tile 做 1 次 full-warp XOR reduction
  每个 tile 由一个 lane 写 shared JPS transcript

每 draw：
  16 次 fold callback
```

代价来源：

1. 每 warp 每 fold **32 次 `__reduce_xor_sync`**；
2. fold 读取 accumulator fragment，直接打断 MMA mainloop 依赖链；
3. shared JPS transcript 有 RMW；
4. JPS shared memory 占约 17KB，间接阻止 `KSTAGES=4`。

因此：

```text
下一阶段 kernel 提速主线 = 减少 fold callback 的指令、shared memory 和依赖链成本。
```

---

## 4. 阶段 A：fold 微优化

### A0. 建立 baseline

每次实验前后必须固定记录：

```text
GPU / sm / driver / CUDA
ARCH
CUTLASS_HOME
GROUPM
KSTAGES
TC_PERSIST
kernel FUSED ms/draw
kernel TH/s
MINE done TH/s
pool 60s TH/s
POSTCHECK
official verifier
accepted/rejected
ptxas register count
spill stores / spill loads
```

建议 baseline 命令：

```bash
cd peral
ARCH=sm_120 GROUPM=128 KSTAGES=3 bash build.sh
./build/plainproof_gen --cfg real --mine 3 --breakdown 2>&1 | tee /tmp/baseline.log
```

如果是 4090 / L40 / 3080Ti，替换 `ARCH`：

```bash
ARCH=sm_89   # 4090 / L40
ARCH=sm_86   # 3080 Ti
```

验收：

```text
日志必须是 tc(cutlass2)，不能掉回 tc(block)。
```

---

### A1. real-config fold direct-store

#### 原理

当前 fold：

```cpp
JPS(local_jt, c % 16) = rotl32d(JPS(local_jt, c % 16), 13) ^ myx;
```

REAL config 下：

```text
k / rank = 16
c = 0..15
每个 transcript word 只写一次
JPS 初始为 0
rotl32d(0,13) == 0
```

因此可实验性简化为：

```text
JPS(local_jt, c) = myx
```

#### 预期收益

```text
1% - 3%
```

#### 风险

仅对：

```text
chunk_count == 16
```

成立。若未来 config 出现 `k/rank > 16`，同一 transcript word 可能被多次更新，此简化不成立。

#### 验收

```text
POSTCHECK ok=1
kernel ms 稳定下降
pool accepted / 0 rejected
```

---

### A2. lane-owned transcript，移除 JPS shared transcript

#### 原理

当前 baseline 下：

```text
每 warp 负责 32 jackpot tiles
每 lane 可以天然负责 1 个 jackpot tile
每 tile transcript 是 16 × u32
```

因此可考虑：

```cpp
uint32_t jp_reg[16];  // 每 lane 自己 tile 的 transcript

每次 fold:
  jp_reg[c] = myx;

mainloop 结束:
  lane 自己对 jp_reg[0..15] 做 jackpot_blake3
  le_u256 <= bound 则 atomicCAS
```

这可以去掉：

```text
JPS shared array
JPS clear
JPS RMW
JPS final read
部分 block-end sync 压力
```

更重要的是，去掉 JPS 后 dynamic smem 可能从：

```text
约 90KB = CUTLASS 72KB + JPS 17KB
```

降到：

```text
约 72KB
```

从而重新允许测试：

```text
KSTAGES=4
```

#### 预期收益

单独 lane-owned transcript：

```text
2% - 6%
```

如果配合解锁 `KSTAGES=4`：

```text
综合 5% - 12%
```

#### 最大风险

每 lane 多保存：

```text
16 个 uint32 transcript word
```

可能导致：

```text
register count 逼近/超过 255
spill 到 local memory
性能反降
```

#### 必须记录

编译时需要看：

```text
ptxas register count
spill stores
spill loads
```

如果出现明显 spill，应立即回退或改用压缩/分阶段 transcript。

---

### A3. lane-owned transcript + `KSTAGES=4`

只有在 A2 后 shared memory 降下来时再测。

#### 原理

之前 `KSTAGES=4` 失败原因：

```text
CUTLASS smem 约 96KB
+ JPS 约 17KB
= 约 113KB，超过上限
```

若移除 JPS：

```text
KSTAGES=4 可能接近 96KB，可 launch
```

#### 预期收益

```text
2% - 8%
```

它能否有效取决于当前是否仍有 DRAM / cp.async latency 隐藏不足。

#### 验收

```text
launch 不报 smem attr error
POSTCHECK ok=1
kernel ms 下降
ptxas 无不可接受 spill
```

---

### A4. fold reduction 重排：shared transpose / parallel reduce

如果 A1/A2 收益不足，可试。

#### 当前方式

```text
for tile in 32:
  all lanes reduce one tile
```

#### 替代思路

```text
每个 lane 先算自己对 32 个 tile 的 partial
写 shared partial[lane][tile]
每个 lane 负责一个 tile，读取 32 lane partial 并 XOR
```

#### 优点

可能减少：

```text
32 次 warp redux 的指令压力
```

#### 缺点

增加：

```text
shared store/load
syncwarp
bank conflict 风险
shared memory 使用
```

#### 优先级

低于 A1/A2/A3。  
只有在 A2 腾出 JPS smem 后才值得试。

---

## 5. 阶段 B：wall-clock 收敛，去掉 gather 开销

当前 `gpu_prep.cu` 已经把：

```text
RNG fill
BLAKE3 tree commitments
noise add
```

搬上 GPU，且 search(N) / prep(N+1) 有 overlap。

但当前 search 前仍然有：

```cpp
gather_rows(dA  -> dAp)
gather_rows(dBt -> dBtp)
```

`tc(cutlass2): FUSED ... ms` 只计 search kernel，不计 gather。  
pool wall-clock 低于 kernel-only 的差值，部分来自 gather / prep / stream sync。

### B1. phase2 直接生成 gathered noised layout

#### 当前 layout

```text
gpu_prep_phase2:
  dA / dBt in-place add noise

tc_search_launch:
  gather dA/dBt -> dAp/dBtp
  search reads dAp/dBtp
```

#### 目标 layout

```text
gpu_prep_phase2:
  根据 row_off + pat_rows、col_off + pat_cols
  直接生成 dAp/dBtp 的 noised gathered layout

tc_search_launch:
  跳过 gather_rows
  search 直接读 dAp/dBtp
```

#### 必要配套

为了保持 search(N) 与 prep(N+1) overlap，必须双缓冲：

```text
dAp0 / dBtp0
dAp1 / dBtp1

draw N search 读 cur buffer
draw N+1 prep 写 next buffer
```

否则 prep(N+1) 会覆盖 search(N) 正在读的 gathered panels。

#### 显存成本

约额外：

```text
~1GB device memory
```

对 4090 / 5090 / L40 通常可以接受。

#### 预期收益

```text
pool wall-clock +5% 到 +15%
kernel-only 不变
```

这是提升真实收益的关键方向。

---

### B2. 不优先做 CUTLASS iterator 直接 gather

理论上可让 CUTLASS iterator 根据 offsets/pattern 直接加载原始 dA/dBt。  
但这会破坏：

```text
contiguous load
coalescing
cp.async / ldmatrix feed
```

实现复杂、性能风险高。  
优先级低于 **direct gathered layout + double buffer**。

---

## 6. 阶段 C：WGMMA/TMA 重写，冲击 500+ TH/s

如果目标是：

```text
350 TH/s -> 400 TH/s
```

优先做 fold。

如果目标是：

```text
500+ TH/s
```

当前 `mma.sync` / Sm80-style CUTLASS path 大概率不够，需要：

```text
CuTe WGMMA + TMA
```

### C1. 为什么需要 WGMMA/TMA

当前 `tc_cutlass_v2.cu` 使用：

```text
cutlass::arch::Sm80
mma.sync.m16n8k32
cp.async multistage
```

即使在 5090 上，也是传统 warp-level tensor op 路线。

Blackwell / Hopper 更高上限来自：

```text
WGMMA
TMA
warpgroup producer/consumer pipeline
更大 tile / 更适合硬件的异步搬运
```

### C2. 风险

```text
WGMMA accumulator layout 更复杂
rank chunk 边界读取 accumulator 做 fold 更难
TMA layout / swizzle 需要重新验证
正确性风险高
只对 Hopper/Blackwell 价值最大
```

### C3. 推荐里程碑

| 里程碑 | 目标 |
|---|---|
| W0 | CuTe WGMMA pure int8 GEMM roofline，必须显著超过当前 436 TH/s |
| W1 | WGMMA full-K mainloop，无 fold，输出 checksum 可对拍 |
| W2 | WGMMA + one rank chunk fold，对 CPU / tc_cutlass_v2 对拍 |
| W3 | WGMMA + 16 fold chunks + jackpot，`POSTCHECK ok=1` |
| W4 | 接 GPU prep + pool 短测，accepted / 0 rejected |

如果 W0 无法显著超过当前 pure GEMM roofline，则暂缓 WGMMA 路线。

---

## 7. per-GPU 小优化矩阵

这些不是主攻方向，但应作为每种 GPU 的维护型 benchmark。

### 7.1 GROUPM

当前 `build.sh` 默认：

```bash
GROUPM=128
```

建议每种 GPU 扫：

```text
GROUPM = 32, 64, 128, 256
```

记录：

```text
FUSED ms/draw
kernel TH/s
pool wall TH/s
POSTCHECK
```

### 7.2 TC_PERSIST

运行时开关：

```bash
TC_PERSIST=1
```

建议：

```text
L40 可考虑开启
5090 不默认开启
其它 GPU 必须 A/B
```

### 7.3 native ARCH

必须确保生产机器用 native SASS：

```bash
ARCH=sm_86   # 3080 Ti
ARCH=sm_89   # 4090 / L40
ARCH=sm_120  # 5090 with CUDA 13+
```

避免 5090 长期走 `compute_90 PTX JIT`。

### 7.4 CUDA Graph

当前 draw 是 200ms 级，launch overhead 不是主瓶颈。  
等 kernel 降到 100-150ms 后，可考虑 graph 化：

```text
prep phase1
prep phase2
gather
search
small copies
```

预计收益：

```text
1% - 3%
```

---

## 8. 推荐推进顺序

### 短期：1-2 天

目标：

```text
5090 kernel 200ms/draw -> 170-180ms/draw
350 TH/s -> 390-415 TH/s
```

任务：

1. 固定 baseline；
2. fold direct-store 实验；
3. lane-owned transcript 实验；
4. lane-owned transcript 后复测 `KSTAGES=4`；
5. 若无收益，再尝试 fold shared transpose；
6. 每步都跑 `POSTCHECK ok=1`。

成功标准：

```text
kernel ms 下降 >= 5%
POSTCHECK ok=1
无明显 spill
真池短测 accepted / 0 rejected
```

---

### 中期：2-4 天

目标：

```text
pool wall-clock 逼近 kernel-only
295-333 TH/s -> 350+ TH/s
```

任务：

1. 设计 dAp/dBtp 双缓冲；
2. `gpu_prep_phase2` 直接生成 gathered noised layout；
3. `tc_search_launch` 支持跳过 gather；
4. 保持 search(N) / prep(N+1) overlap；
5. pool 30-60 分钟测试。

成功标准：

```text
kernel-only 不退化
MINE done / pool 60s TH/s 上升
accepted > 0
rejected = 0
```

---

### 长期：1-2 周

目标：

```text
500+ TH/s kernel
```

任务：

1. CuTe WGMMA pure GEMM roofline；
2. WGMMA mainloop；
3. WGMMA fold；
4. POSTCHECK；
5. GPU prep integration；
6. pool validation。

成功标准：

```text
pure WGMMA GEMM 显著超过 436 TH/s
full WGMMA fold kernel POSTCHECK ok=1
pool accepted / 0 rejected
```

---

## 9. 实验记录模板

每次实验记录建议追加到独立日志或 `PROGRESS.md`：

```text
Date:
Commit:
GPU:
Driver:
CUDA:
ARCH:
CUTLASS:
GROUPM:
KSTAGES:
TC_PERSIST:
Variant:

ptxas:
  regs:
  spill stores:
  spill loads:
  smem:

Correctness:
  POSTCHECK:
  official verifier:
  pool accepted:
  pool rejected:

Performance:
  FUSED ms/draw:
  kernel TH/s:
  MINE done TH/s:
  pool 60s TH/s:
  power:
  efficiency TH/s/W:

Conclusion:
  deploy / reject / keep testing
```

---

## 10. 决策规则

### 可以进入生产的条件

```text
1. POSTCHECK ok=1；
2. hard target 多 draw 稳定；
3. official verifier VALID，或至少有明确计划补验；
4. 真池短测 accepted > 0；
5. rejected = 0；
6. 速度提升 >= 3%，否则不值得增加风险；
7. 不破坏 build.sh 单入口；
8. 不依赖 archive / dead kernel。
```

### 立即回退的条件

```text
POSTCHECK 失败
pool rejected share
kernel launch err
ptxas 出现大量 spill 且速度下降
wall-clock 下降超过 3%
batch / mining loop 行为异常
```

---

## 11. 最终建议

下一阶段不要再盲目扫：

```text
SMALL_TILE
旧 IMMA / WMMA
tc_block 参数
final jackpot BLAKE3
stratum 微优化
```

应该按以下主线推进：

```text
第一主线：fold callback 优化
  目标：350 TH/s -> 390-415 TH/s kernel

第二主线：direct gathered layout + 双缓冲
  目标：pool wall-clock 逼近 kernel-only，350+ TH/s wall

第三主线：CuTe WGMMA + TMA
  目标：500+ TH/s kernel
```

一句话结论：

```text
当前 350 -> 400 TH/s 的关键，是让 fold 不再拖慢 CUTLASS full-K mainloop；
当前 400 -> 500+ TH/s 的关键，是切到 Blackwell/Hopper 原生 WGMMA/TMA 架构。
```

