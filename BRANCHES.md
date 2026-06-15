# BRANCHES.md — 版本与分支总览（权威地图）

> 本文件是仓库所有版本的**唯一权威说明**。落到任何分支先读这里。
> 最后更新：2026-06-15（5090 实测收尾后）。
>
> 一句话现状：**算法本身已贴近各卡的 int8 张量核硬件天花板，dense int8 没有 2× 的提速空间了。**
> 每张卡选对应分支编译即可；调优阶段暂告一段落（"暂时这样吧"）。

---

## 0. TL;DR — 我该用哪个分支？

| 你的显卡 | 用哪个分支 | 编译命令 | 实测算力 |
|---|---|---|---|
| **RTX 5090**（sm_120） | `arch/5090` | `./build.sh`（CUDA≥12.8 自动原生 sm_120） | **354 TH/s** 内核 |
| **RTX 4090**（sm_89） | `arch/4090` | `./build.sh`（自动 sm_89 原生） | **260 TH/s** 内核 / 190-220 端到端 |
| **RTX 3090 / 3080Ti**（sm_86） | `main` | `ARCH=sm_86 ./build.sh` | **106-112 TH/s** 内核 |
| 其它 / 不确定 / 打包分发 | `main` | `PORTABLE=1 ./build.sh` | 多架构 fatbin，自动 JIT |

> 所有分支的**矿工代码完全相同**（同一个 `tc_cutlass_v2.cu` + `gpu_prep.cu`）。
> 分支之间**只差 `build.sh` 的默认架构 gencode**——为每张卡选最优 SASS。
> 正确性红线在所有分支一致：任何 result-VALID 构建必须 `POSTCHECK ok=1` 且通过官方 verifier。

---

## 1. 生产分支（按卡编译，直接挂矿）

### `main` — 基线 / 通用 / 可分发
- **用途**：默认入口分支。便携多架构 fatbin（`PORTABLE=1` → sm_75..sm_90 SASS + PTX），
  或 `ARCH=sm_XX` 手动指定。Ampere（3090/3080Ti，sm_86）走这个分支。
- **build.sh**：`ARCH` 覆盖 > nvidia-smi 自动检测 > `PORTABLE` 多架构。
- **状态**：稳定。线上 earner（4090 4/4 share、3090 153/0）就是这条线的内核。

### `arch/4090` — Ada Lovelace（sm_89）原生
- **相对 main 的差异**：1 个 commit（`5261ac7`），`build.sh` 默认 `-arch=sm_89` 原生 SASS + KSTAGES=3。
- **实测**：260 TH/s 内核 / 190-220 TH/s 端到端 / ~450W。4/4 share，日志干净。
- **状态**：完成。Ada 没有 WGMMA，mma.sync 已是原生最快路径。

### `arch/5090` — Blackwell GeForce（sm_120）原生
- **相对 main 的差异**：2 个 commit（`77b19a4`、`c967581`）。
- **build.sh 关键修正（`c967581`）**：CUDA **≥12.8** 即发原生 `sm_120` SASS + `compute_120` PTX
  （此前误卡在 ≥13，导致 12.8 静默回落到 `compute_90` PTX-JIT）。
- **实测（真 5090，2026-06-15）**：
  - 原生 `-arch=sm_120` SASS = **354 TH/s** vs `compute_90` PTX-JIT = 340 TH/s ⇒ **+3.5% 白嫖**。
  - `cuobjdump` 确认 `arch = sm_120`。POSTCHECK ok=1。
- **状态**：完成，**本轮调优的实际收益就在这里**（原生 SASS）。

---

## 2. 研究分支（不挂矿，记录结论与教训）

### `dev/fold` — fold 成本的诚实测量（结论：不重写）
- **目的**：攻"实测 fold = 40% 内核"这个假设。
- **关键产出**：`PROFILE_TRUEFLOOR`（DCE-proof 的诚实 GEMM 地板）、`FOLD_SHFL_REDUX`（redux A/B）、
  `bench/fold_ablation.sh`（一次决定性扫描）、`FOLD_PLAN.md`（分析 + §C 预留的 warp-specialization 设计，**未实现**）。
  全部 flag-gated，走 `build.sh` 的 `NVCC_EXTRA`/`SMALL_TILE` 透传，不改 build.sh。
- **结论（真 5090 实测）**：
  - **true_fold% = 7.3%，不是 40%。** 旧的"40%"是 **DCE 伪影**——fold 置空后 ptxas
    把 60.9 ms 的整条 mma 累加链 DCE 掉了，NOFOLD 假快。TRUEFLOOR(184ms) vs NOFOLD(123ms) 坐实。
  - redux = 1.8%（不可约，符合预测）。
  - **决策：fold 已近最优 → 不重写。** warp-specialization / WGMMA-for-fold 最多抠回 <7%，
    风险却很大，不值得。所有 fold 变体都退步：SHFL_REDUX **−12%**（Blackwell 上 REDUX.U32 更快）、
    SMALL_TILE **−45%**（数据复用 ≫ 占用，哪怕 170 个 SM 也是亏）。
- **状态**：已结案。默认构建行为 == main；价值在它记录的**测量方法学与结论**（别再重蹈 DCE 误判）。

### `dev/wgmma` — ⚠️ 死路（保留作教训，勿投入）
- **为什么死**：它的 `device::Gemm` "initial drop" 会**物化完整 m×n 的 int32 C**
  = 131072² × 4 = **68 GB**（+9 GB transcript）→ **任何卡 OOM**；且硬编码 h=8/w=16 正好是 OOM 那一档。
- **正确做法（已在 main 的 `tc_cutlass_v2.cu` 落地）**：按 threadblock 分块输出（BM×BN），
  每块算完**立刻 fold**，从不物化整个 C。
- **结论**：真正的 WGMMA 路必须是 **fused + tiled**，不能用黑盒 `device::Gemm`。
- **状态**：DEAD。可随时删除（`git push origin --delete dev/wgmma`），保留仅为记录这条教训。

---

## 3. 各卡天花板与"还有没有提速空间"（2026-06-15 收尾结论）

dense int8 是协议**写死**的数据类型，不能用 FP4 / 2:4 稀疏（Blackwell 张量核 2× 增益全在那两者上）。
因此结论是：**没有任何架构对本负载还有 ~2× 的空间**——各卡都已跑在自己 dense int8 硅片峰值的 76-93%。

| 卡 | 架构 | 内核 TH/s | 占 dense-int8 roofline | tcgen05? | 剩余空间 |
|---|---|---:|---|---|---|
| RTX 3090/3080Ti | sm_86 | 106-112 | ~76% | 无（Ampere） | 占用/功耗墙，≈无 |
| RTX 4090 | sm_89 | 260 | ~79% | 无（Ada） | 小，mma.sync 微调 |
| RTX 5090 | sm_120 | 354 | **93%**（roofline 382=TRUEFLOOR） | **无**（消费级 Blackwell 走 mma.sync） | **~7%**，mma.sync 微调 |
| (数据中心 B200) | sm_100 | — | — | 有（tcgen05+TMEM） | 仅 FP4/稀疏有 2×，dense int8 无 |

**关于 tcgen05**（更正一处早先口误）：
- `tcgen05` + tensor memory（TMEM）是**数据中心 Blackwell（sm_100：B100/B200）**专属；
  **消费级 sm_120（RTX 5090）没有**，走的是 warp 级 `mma.sync`——也就是我们已经在用的指令。
- 即便在 sm_100 上，tcgen05 对 **dense int8** 的增益也很小（2× 来自 FP4/稀疏，本协议禁用）。
- 所以"为 5090 写 tcgen05 融合方案"= 为一张你没有的卡、为一种协议禁用的精度做设计 → **不做**。
- 5090 的 354 已是其 dense int8 GEMM roofline(382) 的 93%；要再上只能换更快的 GEMM，
  而那是数据中心张量核项目，不是这个矿工的事。

> 置信度说明：tcgen05/sm_120 这条结论基于架构知识（本地无法联网复核），约 85% 置信；
> 若日后租到 box，5 分钟可验证（尝试为 `sm_120a` 编译 tcgen05 PTX，预期 nvcc 拒绝）。

---

## 4. 文档索引

| 文档 | 在哪个分支 | 内容 |
|---|---|---|
| `README.md` | 全部 | 用户向：安装、编译、运行、参数、日志格式、FAQ |
| `BRANCHES.md` | `main` | **本文件**：版本地图 + 各卡天花板 + tcgen05 结论 |
| `DESIGN_speedup.md` | 全部 | 历史提速路线（sm_86 视角），结论已被 §3 实测收口 |
| `FOLD_PLAN.md` | `dev/fold` | fold 成本分析 + DCE 方法学 + 5090 实测矩阵 + §C 未实现设计 |

---

## 5. 维护约定

- **加新卡支持**：从 `main` 开 `arch/<card>` 分支，只改 `build.sh` 的默认 gencode，别动内核源码。
- **改内核**：先在 `main` 改并通过 POSTCHECK + 官方 verifier，再让各 `arch/*` 分支 rebase。
- **测性能假设**：先想清楚 ptxas 会不会把你"短路"的代码 DCE 掉（见 `dev/fold` 的 TRUEFLOOR 教训），
  否则地板是假的。
- **红线**：提速不得换正确性。任何 result-VALID 构建必须 `POSTCHECK ok=1` 且官方 verifier VALID。
