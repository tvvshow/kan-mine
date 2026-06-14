# A2 寄存器转录 — 全 GPU 提速执行方案

**日期**：2026-06-14
**范围**：`peral/src/tc_cutlass_v2.cu` 的 fold callback + jackpot epilogue
**状态**：执行就绪（A1 已落地并在 3080Ti 正确性验证，A2 为下一步结构性改动）
**伴随文档**：本文是 `NEXT_SPEEDUP_PLAN.md`（2026-06-13，总体设计 931 行）的**执行细化 + 新实测数据 + 全 GPU 重定向**。Phase B/C 仍以那份为准。

---

## ⚑ 已实测结论（2026-06-14，3080Ti / sm_86 / CUDA 12.8）—— 本计划的前提被推翻

A2 已**完整实现 + 在真 GPU 验证**（代码在 `src/tc_cutlass_v2.cu`，`-DA2_REG_TRANSCRIPT` 门控；
A/B 脚本 `bench/ablate_a2.sh`）。结果如下：

| 构建 (sm_86, REAL config) | ptxas | POSTCHECK | ms/draw | TH/s |
|---|---|---|---:|---:|
| A1（smem 转录，已发布） | 255 reg, spill **16B** | ok=1 | 707.9 | 99.4 |
| **A2（寄存器转录）KSTAGES=3** | 255 reg, spill **0B** | ok=1 | 704.1 | 99.9 |
| A2 + KSTAGES=4（A2 腾出 smem 才能 launch） | 255 reg | ok=1 | 702.3 | 100.2 |
| A2 + KSTAGES=5 | — | launch err (122KB>100KB cap) | — | — |

**gate-1/2 通过**：寄存器残留成功（编译期 predicated-select 让 `mytrans[]` 留在寄存器，
spill=0，甚至消掉了 A1 的 16B spill），POSTCHECK ok=1。**但 gate-3 失败**：
A1→A2-K4 仅 **+0.85%**，在 run-to-run 噪声（697–718ms，~2%）以内 = **perf 中性**。

**fold 成本分解（A2 基线，PROFILE_NOREDUX / NOFOLD）：**
```
FULL      697.9 ms   全部
NOREDUX   651.7 ms   跳过 32× __reduce_xor_sync
NOFOLD    371.7 ms   完全无 fold（含 MMA DCE，不是真 floor）

redux 净成本 = FULL - NOREDUX = 46 ms = 内核的 6.6%
真 GEMM floor（同形状纯 int8 GEMM）= 537 ms / 131 TH/s
fold 总开销（vs 真 floor）= 698 - 537 = 161 ms = 内核的 23%
  其中 redux 46ms + (累加器读+XOR+写) ~115ms
```

**结论（本计划前提被推翻）：**
1. fold 的瓶颈**从来不是** smem 写（A2 移除它 = +0.5%），**也不主要是** redux（6.6%），
   而是**每 rank-chunk 对累加器做 XOR-fold 这件事本身**（~115ms / 16% of kernel）。
2. fold 微优化（A0→A1→A2→redux）已**触及收益递减**：相对真 GEMM floor，全部 fold 开销
   23%，其中可干净优化的 redux 上限只有 +6.6%；累加器读-XOR（16%）无法消除，只能**与
   下一 chunk 的 MMA 重叠**（5090 memory 里 "snapshot accum + 在 MMA 阴影里做 reduction"
   那条，结构复杂、高风险）。
3. **A2 的处置**：正确、零 spill、解锁 KSTAGES=4，但 perf 中性 → **保留为 `-DA2_REG_TRANSCRIPT`
   门控的基础**（任何"寄存器转录 + fold/MMA 重叠"的后续都建立在它之上），**暂不设为默认**
   （未达 ≥3% 部署门槛，且增加代码复杂度）。

**下一步的真正岔路（已不在 fold 微优化里）：**
- (A) 收尾式：redux 优化（≤+6.6%，低风险）+ 保留 A2，然后转 Phase B（wall-clock）。
- (B) 结构式：fold 累加器读与下一 chunk MMA 重叠（搏 ~+13%，高风险，需重写 mainloop fold 点）。
- (C) 阶跃式：Phase C CuTe WGMMA + TMA —— 真 GEMM roofline 是 2× 当前（dense 838 TOPS），
  这才是从 100→200+ TH/s 的唯一阶跃；fold 也要随之重写为 WGMMA-native。
- 建议：fold 这条线已基本榨干，**把精力投到 (C)**；(A) 仅作为 (C) 之前的低成本收尾。

---

## 0. 核心结论（这次的重定向）

> **fold 不是 5090 专属问题，是所有架构的头号瓶颈。3080Ti 不只是正确性盒子，
> 而是 A2 的最佳开发 + 测量盒子（信号比 5090 更强）。提速目标 = 所有 GPU。**

两条证据：

1. **A1 在 sm_86 上 perf 中性（Δ≈ −0.46%），不是因为 fold 在 sm_86 不重要，
   而是因为 A1 改的不是 fold 的主成本。** A1（direct-store）只是把转录字"首次触碰"
   的一次 `JPS = rotl32d(JPS,13)^myx`（读 + rotl + xor + 写）换成 `JPS = myx`（只写），
   省掉的是 smem **读** + ALU（rotl/xor）。但 smem **写** 与 JPS 的 smem **占用** 都还在。
   fold 的 47% 主要来自后两者 → A1 动不到，A2 才动得到。

2. **fold 占比实测（PROFILE_NOFOLD 下界）：**

   | 架构 | full kernel | NOFOLD | fold 占比 |
   |---|---:|---:|---:|
   | sm_120 (5090) | 199.5–200.8 ms | 115.3–123.1 ms | **≈ 40%** |
   | sm_86 (3080Ti) | 722 ms (97.5 TH/s) | 380 ms | **≈ 47%** |

   3080Ti 的 fold 占比**更大**，所以 A2 的 Δ% 在 3080Ti 上更容易测准。
   （NOFOLD 受 DCE 影响是下界，真实 fold 成本略低，但 40–47% 的量级稳健。）

---

## 1. 当前状态快照

- **A1（fold direct-store）已提交 `e466fde`**，3080Ti 正确性验证：RMW 与 A1 两路 `POSTCHECK ok=1`。
  代码在 `tc_cutlass_v2.cu:643-648`，`-DFOLD_RMW_ALWAYS` 可一键 A/B 回到 pre-A1。
  perf：sm_86 中性（预期，见 §0）；sm_120 增益**尚未实测**（5090 盒子已释放）。
- **便携包已落地**：`PORTABLE=1 build.sh` + `package_portable.sh` → 自包含 tarball，
  仅依赖 NVIDIA 驱动；预编译产物在 cnb `dist` 分支。详见 `reference_portable_build`。
- **便携 kan 真池验证通过**（3080Ti，kryptex 7048）：`GPU #0 share accepted`，
  75 TH/s wall（96 TH/s 10s 窗口），worker pmport。
- **GEMM roofline（同 tile 形状，纯 int8）**：sm_86 ≈ **131 TH/s**。
  当前 97.5 TH/s = roofline 的 74%，即 fold 优化的理论头顶约 **+34%**。

---

## 2. A2 是什么：lane-owned register transcript

**现状（smem 转录）：**
```
jp_sh[NJT*17]  共享数组（约 17KB dynamic smem，stride 17 防 bank conflict）
  fold:      每 rank chunk 后，拥有该 tile 的 lane 把 redux 结果 RMW 写进 JPS(tile,q)
  epilogue:  每个线程 for t: 读 JPS(t, 0..15) → jackpot_blake3 → 比 bound
```

**A2（寄存器转录）：** 每个 lane 把**它所拥有的那一个 tile** 的 16 个转录字
（`uint32_t mytrans[16]`）保存在**寄存器**里：
```
fold:      lane 把 redux 结果写进自己的 mytrans[q]（无 smem）
epilogue:  每个 active lane 直接对自己的 mytrans 做 jackpot_blake3 → 比 bound
```

baseline tile（64×64 warp，JR=8、JC=4）下 **一个 warp 恰好覆盖 JR*JC=32 个 tile，
一个 warp 恰好 32 lane → 每 lane 拥 1 个 tile**，所以 16 个转录字全部落在该 lane 的寄存器，
天然可行。（SMALL_TILE 时 lane 16-31 空闲，但 SMALL_TILE 已在 5090 证伪，不予考虑。）

---

## 3. A2 的三项结构性收益（架构无关）

1. **移除 fold 的 smem 写**：fold 47% 成本的主来源（A1 动不到的那部分）。
2. **移除 JPS 的 ~17KB dynamic smem**：直接解锁 `KSTAGES=4`。
   之前 `KSTAGES=4` launch err 正是 `72KB CUTLASS + 17KB JPS > 100KB`(sm_120/sm_86 cap)；
   去掉 JPS 后约 `90KB → 73KB`，`KSTAGES=4`（深一级 pipeline）重新进入测试矩阵。
3. **移除 epilogue 的 JPS clear + final read + 末尾 `__syncthreads`**：省一轮 smem 往返。

**诚实的预期边界**：A2 **不**动 XOR redux 本身（`__reduce_xor_sync` 每 chunk 每 tile 一次仍在）。
所以 A2 的增益 = `smem-write 消除 + occupancy/KSTAGES 提升`，**不是全部 47%**。
若 A2 后 fold 仍显著，下一手是 §6 的 redux 优化（segmented butterfly），而非加深 KSTAGES。

**主风险 = 寄存器压力**：当前 5090 build 已 244 regs。每 lane +16 regs（mytrans）。
若逼近/超过 255 → ptxas spill → 可能反而变慢。**这正是必须在盒子上实测、不能纸面判定的点。**

---

## 4. A2 实施步骤（具体到 `tc_cutlass_v2.cu`）

> 改动前先 `cp tc_cutlass_v2.cu tc_cutlass_v2.cu.preA2` 备份，便于一键回退。

1. **lane 私有寄存器数组**：在 kernel 体内（fold lambda 捕获范围）声明
   `uint32_t mytrans[16];` 并在 mainloop 前 `#pragma unroll` 清零（替代 `:572` 的 `jp_sh` 清零循环）。

2. **fold lambda（`:632-649`）改写**：保留 A1 的 `c<16` 分支语义，但写寄存器：
   ```cpp
   // A2: 写 lane 私有寄存器而非 JPS smem
   if (bi + jtrib < nrow_off && bj + jtcib < ncol_off) {
     const int q = c & 15;
   #if defined(FOLD_RMW_ALWAYS)
     mytrans[q] = rotl32d(mytrans[q], 13) ^ myx;
   #else
     mytrans[q] = (c < 16) ? myx : (rotl32d(mytrans[q], 13) ^ myx);
   #endif
   }
   ```
   （`mytrans` 的索引 `q` 是编译期可展开的常量循环变量吗？不是——`c` 是运行期。
   寄存器数组按变量下标访问，ptxas 需把它放进寄存器且下标可解析；若 ptxas 退化为 local memory
   就失去意义。**编译后必须确认 `mytrans` 没有掉进 local memory**，见 §5 gate-1。）

3. **jackpot epilogue（`:656-669`）改写**：从"每线程遍历所有 tile 读 JPS"
   改为"每个 active lane 处理自己的 tile"：
   ```cpp
   #ifndef PROFILE_NOJACKPOT
   const int my_jr = lane / JC, my_jc = lane % JC;
   if (my_jr < JR) {
     const int jt_i = bi + warp_m*JR + my_jr;
     const int jt_j = bj + warp_n*JC + my_jc;
     if (jt_i < nrow_off && jt_j < ncol_off) {
       uint32_t out[8];
       jackpot_blake3(key, mytrans, out);      // 直接用寄存器转录
       if (le_u256(out, bound))
         if (atomicCAS(win_flag, 0, 1) == 0) { *win_rt = jt_i; *win_ct = jt_j; }
     }
   }
   #endif
   ```

4. **移除末尾 `__syncthreads`（`:673`）** 与 persistent 复用 JPS 的注释——寄存器不跨迭代共享，
   每次 mainloop 自带 barrier。**但要确认 persistent grid-stride 下 `mytrans` 在下一 pid 迭代起点被重新清零**
   （把 §4.1 的清零放进 `pid` 循环体内，而非循环外）。

5. **host wrapper 的 dynamic smem**：从 launch 配置里减去 JPS 的 `NJT*17*sizeof(uint32_t)`
   （搜索 wrapper 里 `cudaFuncSetAttribute` / shared size 计算处），让 occupancy 与 KSTAGES=4 真正受益。

---

## 5. 验证门槛（全部可在 3080Ti 完成，不需要 5090）

```
gate-1  编译/寄存器:  grep -E "registers|spill|local" 编译日志
        - mytrans 必须在寄存器，不得掉 local memory（否则 A2 失去意义）
        - regs 不得逼近 255 导致 occupancy 崩；spill 显著增加且变慢 → 回退或调 budget

gate-2  正确性:       ./build/plainproof_gen --cfg real --mine 1 --target <EASY> 777
        - POSTCHECK ok=1（CPU 独立重算 GPU 中标 tile）

gate-3  A/B Δ%:       复用 bench/ablate_fold_a1.sh 的结构（RMW vs A2 默认）
        - hard target 多 draw，比 FUSED ms/draw，Δ 稳定为正才算赢

gate-4  KSTAGES=4:    A2 过后 → KSTAGES=4 ARCH=sm_86 ./build.sh
        - 确认 smem < cap 且能 launch；再测一次 Δ%（可能再 +x%，也可能无效）

gate-5  上生产:        official verifier VALID（cnb）+ 真池短测 accepted>0 / rejected=0
```

**3080Ti 盒子操作配方**（CUDA 已就位，`~/cutlass` 已 clone）：
```bash
# A/B（A2 默认 vs pre-A1 RMW 基线）
ARCH=sm_86 bash bench/ablate_fold_a1.sh        # FOLD_RMW_ALWAYS 一键 A/B 仍有效

# 单独看 A2 编译后的寄存器/spill
ARCH=sm_86 ./build.sh 2>&1 | grep -E "registers|spill|local|tc_cutlass_v2"

# KSTAGES=4 解锁验证
KSTAGES=4 ARCH=sm_86 ./build.sh && \
  ./build/plainproof_gen --cfg real --mine 3 --target 0000...0001 --breakdown 777
```
盒子连接见 memory `reference_portable_build` / `reference_5090_ablation_2026_06_13`。

---

## 6. A2 之后的全 GPU 路线

```
A2   (现在, 3080Ti)   移除 JPS smem → fold 结构性提速 + 解锁 KSTAGES=4
                      目标: sm_86 97.5 → 105-115 TH/s; sm_120 350 → 390-415 TH/s
                      （两档都验证，因为收益是架构无关的）

A2.1 (若 fold 仍显著)  redux 优化: 32 个 tile 的分段蝶形归约替代 32 次 full-warp redux
                      （A2 不动 redux，这是补刀）

Phase B (wall-clock)  direct gathered layout + 双缓冲，让 pool wall-clock 逼近 kernel-only
                      （详见 NEXT_SPEEDUP_PLAN.md §B）

Phase C (架构原生)     CuTe WGMMA + TMA（Blackwell/Hopper），目标 500+ TH/s kernel
                      （详见 NEXT_SPEEDUP_PLAN.md §C；这一档才真正只惠及新架构）
```

**一句话**：A0→A1 是代数削减（小、架构无关但中性），**A2 是结构削减（移除 smem 转录，
所有 GPU 受益，3080Ti 就能开发和量化）**，Phase C 才是只服务新架构的重写。

---

## 7. 回退与红线

- 任一 gate 失败立即回退到 `tc_cutlass_v2.cu.preA2`（或 `git checkout` A1 版本）。
- 速度提升 < 3% 不值得增加正确性风险（沿用 `NEXT_SPEEDUP_PLAN.md §10`）。
- 不破坏 `build.sh` 单入口；`-DFOLD_RMW_ALWAYS` 的一键 A/B 必须继续可用。
- pool rejected > 0 或 POSTCHECK 失败 = 硬回退。
