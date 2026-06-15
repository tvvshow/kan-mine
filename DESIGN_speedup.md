# Pearl(PRL) 矿工提速设计 (DESIGN_speedup.md)

> **【状态：已收口，2026-06-15】** 本文是 sm_86(3080Ti) 视角的历史提速路线。其中的 Phase 全部
> 已落地并经实测验证——内核从 30 → 106-112 TH/s（3090），4090=260，5090=354。后续的
> fold/占用/Blackwell 假设已在真 5090 上测完，结论是**算法近硬件天花板、无 2× 空间**。
> **当前权威结论看 [`BRANCHES.md`](BRANCHES.md)；fold 实测矩阵看 `dev/fold` 分支的 `FOLD_PLAN.md`。**
> 本文保留作设计与诊断方法学的历史记录。

---

> 目标卡：RTX 3080 Ti (sm_86)。同卡同配置基线：
> **SRBMiner 106 TH/s / lpminer ~91 TH/s（share 640–774ms）**，本实现 kernel-only ~30 TH/s。
> 配置 REAL：m=n=131072, k=4096, rank=256, h=8, w=16；每 draw 工作量 70.37e12 PRL-work。
> 3080Ti int8 张量峰值 ≈ 273 TOPS ⇒ 计算 roofline ≈ **0.5s/draw**。lpminer 的 0.66s 已贴近 roofline，本实现 2.3s 离 roofline 约 4.6×。

---

## 0. 现状澄清 + 一条已被实测推翻的假设（动手前必读）

### 0.1 历史 session 的硬证据（来自 83d946ba transcript）
- **`tc_imma.cu` 实测 30.27 TMAC/s（2324ms/draw，kernel-only）= WMMA 版 `tc_block.cu` 的 ~30 完全相同。指令从 WMMA 16×16×16 换成 IMMA 16×8×32，零提速。**
- 当时 tc_imma 的 fragment-layout 正确性**一直没干净确认**：全 1 测试（A=B=all-ones ⇒ C=32）无法区分布局对错，调试卡在这里，过程中出现过 `POSTCHECK ok=0`。
- 还没有任何一份 ncu profiling 定位真正的 stall 原因（perf-counter 在 cnb 被封；3080Ti box 上可做）。

### 0.2 由此推翻的假设
> **"WMMA→IMMA 指令替换是最大收益" 是错的。** 两者都撞到同一个 30 TMAC/s 天花板 ⇒ **瓶颈不是 MMA 指令吞吐，而是访存/占用/fold**。单纯把 tc_block 换成"正确的 IMMA"不会更快。lpminer 的优势不在 IMMA 这一条指令，而在 **tile 128×256 的数据复用 + persistent 调度 + 全 GPU 侧生成（无 1GB H2D、无 CPU 噪声）** 的组合。

### 0.3 当前 build 的风险
`build.sh` 现在只链接 `tc_imma.o`（`tc_block.cu` 未编译）。而记忆里"153/0 线上 earner、官方 verifier ALL-VALID"是 **WMMA 版 `tc_block.cu`**。tc_imma 既**不更快**、又**正确性存疑**（ldmatrix per-lane 寻址可能违反 PTX：`ldmatrix.x4` 要求每 lane 提供自己那行的 shared 地址，当前所有 lane 用同一个 `a_base`）⇒ **链接它是纯下行风险**。

> **M0（先做）**：
> 1. **立即把 `build.sh` 链接回已验证的 `tc_block.cu`**，保住线上收益（tc_imma 无速度收益，不值得冒正确性风险）。
> 2. 在 3080Ti box 上对 `tc_block` 跑 `ncu --set full`，拿到**真正的 stall 归因**（memory-bound? occupancy? fold 串行?）——这是把后续投入对准瓶颈的唯一依据，别再凭猜。
> 3. tc_imma 留作分支实验，但只有在它**既正确又更快**时才有意义；按 0.2，它在"更快"上已被证伪，除非配合 tile/占用一起改。

**任何 kernel 改动的不可逾越红线**：同一 (header,target,seed) 下，win tile 与 proof 必须通过 CPU `POSTCHECK ok=1` 且官方 verifier VALID。提速不得以正确性换取。

### 0.4 Box 实测验证（2026-06-10，RTX 3080 Ti / sm_86 / CUDA 12.8 / driver 595.80）
在新测试机 `ssh -p 23 root@117.50.194.150` 上实测，命令 `plainproof_gen --mine N --cfg real --target <bound=2^249>`（每 draw 期望 ~2^20 命中）：

| 测试 | 结果 | 推论 |
|---|---|---|
| **tc_imma（当前 build）** | **0 命中**，proof 空 | **tc_imma 是坏的**：挂线上永远挖不到 share |
| **tc_block（已验证）** | **MINE WIN draw=1 rt=12 ct=544, POSTCHECK ok=1, 136985B proof** | 正确 |
| tc_imma vs tc_block 速度 | 都 **30.3 TH/s（2.31s/draw）** | **指令替换零提速** |
| STAGES 2→3→4 | 2315→2269→2233ms（仅 **3.5%**） | **非延迟瓶颈**（加深预取无用） |
| fold 短路（纯 GEMM 计时） | 2308ms（仅省 **0.3%**） | **非 fold 瓶颈** |
| 计算 roofline | 0.52s（70.4e12 MAC ÷ 136 TMAC/s int8 峰值） | 实测 2.31s = **离 roofline 4.4×** |
| 端到端 | 2.97s/draw（23.7 TH/s），CPU prep 被双缓冲藏住 | 一旦 kernel 提速，1.4s CPU 翻成墙 |
| **GPU 运行态实测** | **349W（=SRBMiner 同功耗墙）、sm 1800MHz、gpu_util 100%、mem_util 仅 34%** | **不是显存带宽瓶颈！** |

> **⚠️ 诊断纠正（2026-06-10 实测）**：先前"DRAM 带宽瓶颈 ~1TB/draw"是**错的**——`mem_util 仅 34%`，显存远未跑满。真正事实：**同样 349W/满频，SRBMiner 做 106 TH/s（~78% int8 张量峰值），我们只做 30 TH/s（~22%）**。芯片在忙但**张量核没喂饱**——周期耗在非 MMA 开销上。根因：① tile 太小（128×128）→ 每次共享加载喂的 MMA 太少；② 占用率仅 ~50%（每 block 32KB 共享 → sm_86 上 3 block/SM = 768/1536 线程）→ 盖不住共享加载延迟，张量管线空转；③ WMMA opaque 片段加载 vs lpminer 的 ldmatrix(LDSM)；④ grid launch vs persistent。STAGES 2→4 无效（全局延迟早被 cp.async 藏住，故 DRAM 才 34%）、fold 短路无效——都印证瓶颈在 **SM 内部喂张量核的效率**，不在显存、不在延迟、不在 fold、不在指令类型。
> **提速方向（不变，但理由更准）**：128×256 大 tile + ldmatrix + persistent + 提占用 → 提高张量核利用率（不是降显存流量）。手写现实可达 50–80 TH/s；稳上 100+ 基本需 CUTLASS（§5）。
> **已执行**：build.sh 已改回 tc_block 并在 box 全新构建验证（POSTCHECK ok=1）——线上风险已消除。

---

## 1. 瓶颈分解（按对端到端 TH/s 的影响排序）

| # | 瓶颈 | 现状成本/draw | 根因 | 解法所在 Phase |
|---|---|---|---|---|
| B1 | 张量核利用率仅 ~22%（**已实测确认**） | 2.31s（roofline 0.5s；349W/满频但 mem_util 仅 34%） | 张量核没喂饱：tile 小(128×128)、占用 ~50%、WMMA 加载开销、非 persistent | Phase 1.2/3 + Phase 5 |
| B2 | 每 draw ~1GB H2D | ~0.65s 非重叠开销（端到端 2.97 vs kernel 2.31） | `cudaMemcpy` 整个 a_noised+b_noised_t（各 512MB） | Phase 2 |
| B3 | CPU 噪声生成 ~1.4s | 现被双缓冲藏在 2.31s kernel 后 | 噪声/RNG/blake3 在 CPU | Phase 2（**B1 一旦见效，B3 立即翻成瓶颈，Amdahl**） |
| B4 | 启动/调度开销、占用不足 | 数十 ms × 多次 launch | 非 persistent、grid launch、gather 独立 kernel | Phase 3 |

**关键依赖**：B1 与 B3 必须一起解决才能真正逼近 lpminer。只优化 kernel（B1）会让 1.4s 的 CPU 噪声（B3）成为新天花板——端到端最多到 ~1.4s/draw（≈50 TH/s），到不了 106。

---

## 2. Phase 1 — GEMM 计算打到 roofline

> ⚠️ 据 §0.2：WMMA 和 IMMA 都卡在 30 TMAC/s ⇒ **瓶颈不是指令类型**。Phase 1 的重点因此**不是** 1.1 的"换/修 IMMA"，而是 **1.2 tile 复用 + 1.3 流水 + 1.4 fold + 占用**。先做 §0 的 ncu profiling 再决定 1.2/1.3/1.4 的先后。把内核形态对齐 lpminer 的 SASS（512× IMMA.16832 + LDSM + tile 128×256 + 253 寄存器 + 无 cp.async 用 LDG + persistent），但要清楚**收益来自组合，不来自单换指令**。

### 1.1 正确的 IMMA m16n8k32 + ldmatrix（核心）
- 指令：`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`（A 4 regs / B 2 regs / C 4 regs，K=32/inst，4096 MAC/inst）。
- smem→reg 用 `ldmatrix.sync.aligned.m8n8.x4.b16`（int8 视作 b16 打包，2×s8/元素）。**务必按 PTX 规范让 lane t 提供它负责那一行的 shared 地址**——这是当前 `tc_imma.cu` 最可疑处，重写时以官方 PTX ISA 表为准并用单元测试逐 fragment 比对 CPU 参考。
- 验证手段：写一个 16×8×32 的微基准，与 `compute_tile_jackpot_hash_cpu` 的逐元素结果对拍，再扩到整 tile。

### 1.2 tile 放大到 128×256（对齐 lpminer）
- BM=128, BN=256；每 block 输出 128×256，提高算术强度、减少 A 的重复加载（A 在 N 方向复用翻倍）。
- 注意 smem 预算与寄存器：lpminer 用满 253 寄存器、靠 `__launch_bounds__` 控占用。

### 1.3 深化 K 流水
- lpminer **不用 cp.async、直接 LDG + LDSM** 也能贴 roofline（说明它靠寄存器/调度隐藏延迟）。两条路二选一并实测：
  - (a) 保留 cp.async 但 **ISTAGES 提到 3–4**（当前 2 太浅）；
  - (b) 仿 lpminer：双缓冲 LDG→smem→ldmatrix，手动软件流水。
- 以 ncu 的 `sm__throughput`、`smsp__inst_executed_pipe_tensor` 占用率为准择优。

### 1.4 降低 fold 串行成本
- 现状：每个 rank-chunk（16×/draw）把累加器 `store_matrix_sync→o_sh`，再 `__syncwarp` + warp-XOR-shuffle 折叠，shared 往返穿插在 MMA 之间，强制串行。
- 改法：**直接从累加器寄存器做 warp shuffle 折叠**，去掉 `o_sh` 往返；jackpot transcript（每 jackpot-tile 16×u32）尽量留寄存器/最小 shared。数学不变（仍是每 chunk `rotl(·,13)^xored`，tid=chunk%16）。
- XOR-reduction over h*w=128 个 cell 用寄存器内 + `__shfl_xor_sync`，避免转置式 shared 访问。

**Phase 1 验收**：`tc(imma): … TH/s` 从 ~30 → **≥70**（接近 lpminer kernel-only），且 `POSTCHECK ok=1` + 官方 verifier VALID。ncu 张量管线利用率 >60%。

---

## 3. Phase 2 — 干掉 1GB H2D 与 1.4s CPU 噪声（GPU 侧生成）

lpminer 的 fatbin 含 `NoiseGenerationKernel<r,128>`、`fill_xorshift_kernel`、`extract_sparse_pairs_kernel`、`fold_noise_a/b_kernel`、`pearl_merkle_chunk_groups_kernel<128>`——**A/B/噪声/merkle 全在 GPU**。本实现把这些放在 CPU，导致 B2(1GB H2D) + B3(1.4s CPU)。

### 2.1 把 A/B 的 RNG 填充搬上 GPU
- 现状 `produce_draw` 用 splitmix64 在 CPU 填 A、Bt，再整体 H2D。
- 改：GPU kernel 直接用同一 splitmix64 公式（`(seed^…)+d*…+row*…`，逐行流）把 A、Bt **直接生成进 device 的 gather 后布局**（甚至 fuse 进 gather，连 `dA/dBt` 全量都不必驻留）。
- 收益：消除每 draw ~1GB H2D（仅传 seed/d/config 几十字节）。RNG 公式与 CPU 完全一致 ⇒ win 后 CPU 端 `produce_draw(draw)` 重算仍 byte-identical，Merkle/postcheck 不变。

### 2.2 把噪声生成搬上 GPU
- `compute_noise_for_indices` 的四块（E_AL/E_AR/E_BL/E_BR）+ `matvec_sparse_perm` 全部并行度极高：
  - keyed-blake3 `get_random_hash`（已有 device blake3，可复用 `jackpot_blake3` 同款 compress）；
  - uniform：`byte&63-32`；perm：`first=ru&(r-1); second=first^(1+mulhi(r-1,ru))`；
  - noise_a[row]=matvec(E_AR,E_AL[row])，noise_bt[col]=matvec(E_BL,E_BR[col])。
- 直接把 `a_noised = A + noise_a` 融进生成 kernel，省掉中间缓冲与第二趟。
- 收益：消除 B3 的 ~1.4s CPU；blake3 承诺链（hash_a/hash_b/seeds）也可上 GPU（或保留在 CPU，因为只需 4 次整矩阵 blake3，可与 GPU 重叠）。

### 2.3 承诺/seed 链的最小 CPU 残留
- `job_key`、`bound` 是 draw 不变量（已在循环外）。
- `hash_a/hash_b → b_noise_seed → a_noise_seed` 依赖 A/B 全量。两条路：
  - (a) 在 GPU 上做 A/Bt 的 padded-blake3（lpminer 的 `pearl_merkle_chunk_groups_kernel` 思路），全程不回 host；
  - (b) 过渡期保留 CPU blake3，但用 GPU 生成的 A/Bt（经一次 D2H 仅传 hash 输入或在 device 算）——优先 (a)。

**Phase 2 验收**：端到端 `MINE done … TH/s` 与 kernel-only 收敛（差距 <15%），即 CPU/H2D 不再是瓶颈；win→produce_draw 重算→POSTCHECK ok=1 仍成立。

---

## 4. Phase 3 — persistent 调度 + 占用收尾

- **Persistent tile scheduler**：一个 block 处理多个输出 tile（lpminer `StaticPersistentTileScheduler`），grid = SM 数×每 SM block 数，消除大 grid 的尾部不均与多次 launch。
- **gather 融合**：把行/列 gather 融进生成 kernel 或 GEMM 的 prologue，去掉两个独立 `gather_rows` launch。
- **占用/寄存器**：用 `__launch_bounds__` + `-maxrregcount` 逼近 lpminer 的 253 寄存器/合理 occupancy；ncu 看 `achieved_occupancy`、`stall_*` 调 ISTAGES 与每 warp 子块数。
- **sm_86 SASS**：用 `nvcc -gencode arch=compute_86,code=sm_86` 出原生 SASS（不要只靠 PTX JIT），`cuobjdump -sass` 对照 lpminer 确认出现 `IMMA.16832` + `LDSM`，没有意外的 `LDG` 回退。

**Phase 3 验收**：kernel-only ≥ **90 TH/s**，端到端 ≥ **80 TH/s**，share 时间进入 lpminer 量级（<1s）。

---

## 4.5 ✅ CUTLASS roofline 实测（2026-06-10，本卡决定性数据）

在 box（RTX 3080 Ti / sm_86 / CUDA 12.8）上用 CUTLASS 3.5.1 跑纯 int8 GEMM（`bench/cutlass_int8_bench.cu`，shape m=n=16384 k=4096，C int32 落地，20 次取平均）：

| tile (TB) / warp / stages | ms/gemm | **TMAC/s (=Pearl TH/s)** | int8 TOPS | 占 int8 峰值 |
|---|---:|---:|---:|---:|
| **128×256×64 / 64×64 / s4** | **8.35** | **131.7** | 263 | **~96%** |
| 128×256×64 / 64×64 / s3 | 8.47 | 129.8 | 260 | 95% |
| 128×128×64 / 64×64 / s3 | 11.70 | 94.0 | 188 | 69% |
| 256×128×64 / 64×64 / s3 | 11.91 | 92.3 | 185 | 67% |
| 128×128×128 / 64×64 / s3 | 11.93 | 92.1 | 184 | 67% |

**结论（直接回答"为什么达不到 106、能不能达到"）：**
1. **这张卡纯 GEMM 能到 131.7 TMAC/s**（96% 峰值）——**远高于 SRBMiner 的 106 TH/s**。SRBMiner 106 < 131.7 的差额（~19%）正是 jackpot fold/epilogue + 噪声开销，完全自洽：**CUTLASS GEMM 130 → 叠加我们的 jackpot epilogue ≈ 106**。所以 **106 TH/s 是确凿可达的**，不是玄学。
2. **我们当前 30 TMAC/s = 这张卡 GEMM roofline 的 23%**。差距 100% 在 GEMM 工程，不在算法、不在带宽、不在指令类型。
3. **仅 tile 从 128×128→128×256 就把吞吐从 94→130（+38%）**——精确证实"张量核被小 tile 饿着"的诊断；这正是 lpminer 用 128×256 的原因。`256×128×128` CUTLASS 报 `Error Internal`（smem 超限/不支持组合），忽略。
4. **可达目标分层据此锁定**：CUTLASS 128×256 + 自定义 jackpot epilogue + Phase 2（GPU 侧生成）→ **端到端 ~100–106 TH/s**（≈SRBMiner）。手写 WMMA 无论怎么调都到不了，因为 WMMA opaque fragment + 我们的 fold 往返就是把利用率压在 ~22% 的根因。

> **⇒ 路线决断：不再投入手写 IMMA/WMMA 调优（§1.1、原 Phase 1 已废）。直接走 §5 的 CUTLASS 路线**——它在本卡已实测拿到 130 GEMM roofline，是唯一能逼近 106 的工程路径。下一步：在 CUTLASS GEMM 上挂自定义 epilogue 实现 per-rank-chunk XOR-fold + rotl13 transcript + jackpot blake3 + ≤bound 比较，POSTCHECK ok=1 后再叠 Phase 2。

---

## 5. 主路线：手写 CUTLASS-shaped mainloop（mma.sync+ldmatrix）+ 融合 fold

> device::Gemm 不能当黑盒：jackpot fold 必须在**每个 rank-chunk 边界读累加器**（tc_block.cu:174-202），要 hook mainloop 内部，CUTLASS 的 device GEMM 只在整 K 结束后进 epilogue。故**自己写 mainloop**，但用 CUTLASS 同款 `mma.sync.m16n8k32`+`ldmatrix`+多级 cp.async + 128×256 tile（roofline 已证这套到 130）。

### 5.1 ✅ 已验证的原语（2026-06-10 box，随机数据 CPU 对拍，tc_imma 当年正缺这步）
全部在 `peral/bench/`，`nvcc -O3 -arch=sm_86` 秒级编译；精确布局公式见记忆 `reference_imma_layout_validated`：
1. **`mma_microtest.cu`** — `mma.sync.m16n8k32.row.col.s32.s8.s8.s32` 寄存器布局（手工填充 fragment）：**128/128 cell 全对**。
2. **`mma_ldmatrix_test.cu`** — `ldmatrix.x4`（A）/`.x2`（B）寻址喂出与手工版**完全相同**的寄存器：**128/128 全对**，无需 `.trans`。
3. **`mma_warpgemm_test.cu`** — 单 warp 整 K=4096 累加 + 双 N-tile + ldmatrix 沿 K 流式（RK=256 chunk）：**256/256 全对**。
⇒ **warp 级内循环完全打通，无未知原语**。mma 算 C[i][j]=A[i]·Bt[j]（i=行,j=列），与我们 A'·Bt'ᵀ 朝向一致。

### 5.2 剩余 = 纯组装（无新风险，但需仔细）
全部复用 tc_block.cu 已验证部分（gather、persistent buffer、`jackpot_blake3`、`le_u256`、host wrapper 原样照搬），只换内循环：
- **threadblock 铺瓦**：BM×BN 先 128×128 求正确（POSTCHECK ok=1），再升 128×256 取 roofline。8 warp，WGM×WGN=4×2，每 warp WSUBM×WSUBN 个 16×8 mma 子瓦，acc 全程驻寄存器。
- **cp.async 多级流水**：照搬 tc_block 的 `LOAD_SLICE`/STAGES 结构（RSUB=32），smem 布局 `a_sh[STAGES][BM][RSUB]`/`b_sh[STAGES][BN][RSUB]` 行主，ldmatrix 直接从中取。
- **⚠️ fold 朝向坑（关键）**：jackpot tile = h×w = **8 行 × 16 列**，但 mma 子瓦 = **16 行(M) × 8 列(N)**。故一个 jackpot tile 跨 **2 个 N-子瓦**（16 列 = 2×8），且只占 M-子瓦的半高（8/16）。fold 时不能像 tc_block 那样按单个 WMMA 16×16 子瓦直接折——要把一个 M-子瓦行条（16 行 × WSUBN*8 列）存进 o_sh，再在其中按 8×16 划分 jackpot tile 做 XOR-reduce + rotl13。o_sh 预算要按"每 warp 一条 16×(WSUBN*8) 行条"算，注意别和 a_sh/b_sh/jp_sh 加起来超 sm_86 的 ~100KB/SM（否则占用掉到 1 block/SM）。
- **fold 数学不变**：每 rank-chunk 后，对每个 8×16 jackpot tile 把 128 个 cell 当 u32 XOR 得 `x`，`jp[tile][c%16]=rotl(jp,13)^x`；整 K 后 `jackpot_blake3(key=a_noise_seed, jp16)`，`≤bound` 则 `atomicCAS` 写 win tile (jt_i,jt_j)。
- **验收红线**：`plainproof_gen --mine N --tc --cfg real` POSTCHECK ok=1 + 官方 verifier VALID，且 A/B 对比 tc_block 同 (header,target,seed) 命中同一 tile。

### 5.2.1 🆕 发现：线性 smem 布局有 4-way bank 冲突（2026-06-10，桌面推导 + CPU 证明）
tc_imma2/tc_deep_pipeline 的 `a_sh[r][o]` 行距 = RSUB=64B ⇒ bank base = 16·r mod 32：**隔行（r,r+2,…）撞同一 4-bank 组**。每条 ldmatrix 一相读 8 行×16B，只落 2/8 个 bank 组 ⇒ **每 128B 读串行化成 4 个 phase**，A 的 .x4 与 B 的 .x2 全中。这就是 CUTLASS `Swizzle<>` 解决的问题——56 TH/s（tc_imma2）是**带着 4 倍 smem 读串行化**测出来的，与 CUTLASS 94（同 128×128）的缺口大概率主要是它。
修复 = `tc_deep_pipeline.cu` 新旋钮 **SWIZZLE=1**（默认开）：`SWZ(r,o) = o ^ ((((r·RSUB)>>7) & (RSUB/16-1))<<4)`，cp.async 写与 ldmatrix 读同宏 ⇒ 数据自洽、mma/fold 数学不动。CPU 穷举证明（`experiments/verify_swizzle.py`）：RSUB∈{32,64,128} 读侧冲突 ways 4→**1**、写侧保持 1。sweep 网格已加 SWIZZLE 消融列 + `anchor+SWIZZLE` POSTCHECK 门。

### 5.3 备选：CUTLASS warp 级原语（cutlass::gemm::warp::MmaTensorOp）
若手写流水离 130 仍远，可改用 CUTLASS 的 warp 级 MMA + smem iterator（已封装正确 ldmatrix+pipeline），只在外面套自定义 fold——但驱动 warp iterator 较繁。当前 5.1 已自证手写 mma+ldmatrix 正确，优先 5.2。

---

## 6. 里程碑与验证命令

| 里程碑 | kernel TH/s | 端到端 TH/s | 门槛 |
|---|---:|---:|---|
| M0 现状澄清 | 记录 tc_imma 实测 | — | POSTCHECK ok=1 / verifier VALID（否则回退 tc_block） |
| M1 Phase 1 | ≥70 | （受 B3 限）~40+ | 同上 + ncu 张量利用率>60% |
| M2 Phase 2 | ≥70 | ≥65 | 端到端贴近 kernel；win 重算 byte-identical |
| M3 Phase 3 | ≥90 | ≥80 | share<1s；SASS 出现 IMMA.16832+LDSM |

```bash
# 每次 kernel 改动后（3080Ti box）：
cd peral && bash build.sh
cd build && OMP_NUM_THREADS=$(nproc) ./plainproof_gen --mine 5 --tc --cfg real   # 看 tc(imma) TH/s + POSTCHECK ok=1
bash ../verify_run.sh                                                            # 官方 verifier VALID
# 性能剖析（perf-counter 权限够的卡上）：
ncu --set full --kernel-name regex:tc_imma ./plainproof_gen --mine 1 --tc --cfg real
cuobjdump -sass plainproof_gen | grep -E "IMMA|LDSM|LDG" | sort | uniq -c       # 比对 lpminer SASS 形态
```

记录规范沿用 `reference_lpminer_benchmark_3080ti.md`：同表列 GPU/sm/CUDA/driver、kernel TH/s、端到端 TH/s、draw/sec、SRBMiner(106)/lpminer(91) 基线、POSTCHECK、accepted share。

---

## 7. 优先级（按 §0.4 + §4.5 box 实测最终修正）

> ncu 硬件计数器在该容器被 `ERR_NVGPUCTRPERM` 封锁；瓶颈归因用**时序微实验**（STAGES/fold/WMMA-IMMA + nvidia-smi mem_util 34%）+ **CUTLASS roofline 实测（§4.5）**完成，结论确凿：**张量核被小 tile/低占用饿着，不是带宽、不是指令、不是延迟、不是 fold**。

0. **✅ CUTLASS roofline 已实测（§4.5）**：本卡纯 int8 GEMM 128×256/s4 = **131.7 TMAC/s（96% 峰值）** > SRBMiner 106 ⇒ **106 TH/s 确凿可达**，差额即 jackpot epilogue 开销。我们当前 30 = roofline 的 23%。
1. **M0 ✅ 已完成**：build.sh 改回 tc_block 并在 box 全新构建验证（POSTCHECK ok=1）；tc_imma 证伪（0 命中且零提速）。
2. **🎯 主路线 = CUTLASS GEMM + 自定义 jackpot epilogue（§5）**：放弃手写 WMMA/IMMA 调优（实测 22% 利用率封顶，opaque fragment + fold 往返是根因）。在 CUTLASS 128×256 int8 GEMM 的 epilogue 里实现 per-rank-chunk XOR-fold + rotl13 transcript + jackpot blake3 + ≤bound 原子写 win。本卡已实测这条路 GEMM 段拿到 130，是唯一能到 100+ 的工程路径。
3. **Phase 2（GPU 侧 RNG+噪声，与 §5 并行、可控）**：A/B 生成 + 噪声 + a_noised 组装搬上 GPU，砍掉每 draw ~1GB H2D 和 ~1.4s CPU；否则 kernel 提速后 CPU 1.4s 翻成新墙。
4. **手写 IMMA/WMMA 调优已废**：实测零提速、利用率封顶 22%、tc_imma 还坏掉。仅作理解 CUTLASS 的参考件，不再投入。
