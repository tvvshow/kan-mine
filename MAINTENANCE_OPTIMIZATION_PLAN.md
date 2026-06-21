# Kan / Pearl Miner 未来维护、推进、优化方案

**日期**：2026-06-20  
**范围**：`peral/` active codebase  
**性质**：维护治理 + 性能推进 + 生产稳定性方案  
**原则**：不以牺牲 correctness / verifier / pool accepted rate 换速度。  

---

## 0. 当前阶段判断

当前项目已经从早期“追大幅 kernel 提速”的阶段，进入：

```text
近硬件天花板 + 生产稳定性 + 端到端收益收敛 + 技术债治理
```

最新权威性能结论以：

```text
BRANCHES.md
README.md
DESIGN_speedup.md 顶部状态说明
tc_cutlass_v2.cu 当前实现
build.sh 当前产物
```

为准。

当前关键性能水平：

| GPU | Kernel TH/s | Wall TH/s | 状态 |
|---|---:|---:|---|
| RTX 5090 / sm_120 | 约 354 TH/s | 待持续统计 | 已达 dense-int8 roofline 约 93%，大幅 kernel 提速空间很小 |
| RTX 4090 / sm_89 | 约 260 TH/s | 约 190-220 TH/s | kernel 强，wall gap 仍有优化价值 |
| RTX 3090 / 3080Ti / sm_86 | 约 106-112 TH/s | 约 106 TH/s | 已达到闭源同级基线 |

重要修正：

```text
旧的 “fold 占 40%” 判断已经被 dev/fold 的 TRUEFLOOR 方法修正。
当前真实 fold 成本约 7.3%，不是第一主瓶颈。
SMALL_TILE、SHFL_REDUX、黑盒 WGMMA device::Gemm 等路线已经实测或分析证伪。
```

因此，未来推进不能再按“盲目重写 kernel 追 2x”路线走，而应按：

```text
生产稳定优先
端到端收益优先
能效优先
技术债逐步收敛
可测性优先
```

推进。

---

## 1. 总目标

### 1.1 生产目标

```text
1. build / start / deploy / package 指向同一个实际二进制；
2. 不 silent fallback 到 30 TH/s WMMA；
3. pool 挖矿长期 accepted > 0 且 rejected = 0；
4. 遇到断线/job 更新可自动恢复，不破坏 batch loop；
5. 任意新部署可一眼确认：GPU、arch、kernel、TH/s、accepted/rejected。
```

### 1.2 性能目标

短期不再设定不现实的 2x kernel 目标，而是：

```text
RTX 5090:
  保持 350+ TH/s kernel；
  重点提升稳定性、native sm_120、能效 TH/W。

RTX 4090:
  保持 260 TH/s kernel；
  优先收敛 190-220 TH/s wall gap，目标 wall 230+ TH/s。

RTX 3090/3080Ti:
  保持 106+ TH/s wall；
  确保不 fallback、不因部署/依赖问题退化。
```

### 1.3 维护目标

```text
1. 降低 tc_block.cu / tc_cutlass_v2.cu 重复；
2. 降低 plainproof_gen.cpp 巨型函数风险；
3. 降低 miner_main.cpp pool/solo/stats 混杂风险；
4. 把性能假设、A/B、profile 结果固化为可重复 benchmark；
5. 所有文档明确“当前权威结论”和“历史假设”。
```

---

## 2. 不可违反的红线

任何改动，无论是性能、重构还是部署脚本，都必须遵守：

```text
1. build.sh 仍是唯一生产 build 入口；
2. vendored 目录只读；
3. 不从 archive / notes / experiments 链接生产代码；
4. kernel / prep / proof 改动必须 POSTCHECK ok=1；
5. 上生产前必须尽量跑 official verifier VALID；
6. 真池短测必须 accepted > 0 且 rejected = 0；
7. user-facing speed 统一使用 TH/s；
8. TH/s 公式必须与 SRBMiner-MULTI / lpminer / pool 口径一致；
9. batch 保持小批量，不再改成 1000000 这类破坏 loop 的值；
10. 任何 profiling-only 宏必须明确标注“结果不正确，仅计时”。
```

统一速度公式：

```text
TH/s = tiles * rows_pattern_size * cols_pattern_size * dot_len / seconds / 1e12
dot_len = k - (k % rank)
```

REAL config：

```text
work_per_draw =
134,217,728 * 8 * 16 * 4096
= 70.37e12 PRL-work
```

---

## 3. 推进总路线

未来推进分四条主线：

```text
主线 A：生产一致性与部署安全
主线 B：端到端 wall-clock 收益收敛
主线 C：代码重复与结构债治理
主线 D：能效、benchmark、文档治理
```

优先级：

```text
P0：会导致跑错 binary、跑旧 binary、fallback 到 30 TH/s、无法启动的债务；
P1：影响 wall-clock 收益、正确性维护、长期迭代安全的债务；
P2：结构清理、文档治理、benchmark 自动化；
P3：高风险低收益的算法实验，仅在有明确数据支持时做。
```

---

# 主线 A：生产一致性与部署安全

## A1. 统一二进制命名

### 当前问题

当前 `build.sh` 输出：

```text
build/kan
build/plainproof_gen
build/zkprove
```

但仍有脚本/注释使用：

```text
build/pearl-miner
pearl-miner driver
```

尤其：

```text
start_pool.sh 仍检查并启动 build/pearl-miner
```

这是高优先级生产风险：

```text
1. 一键启动可能失败；
2. 远端可能跑旧残留 binary；
3. deploy/check 脚本可能误判 build 成功但启动失败。
```

### 短期方案

在 build 阶段保持 `kan` 为主产物，同时提供兼容 alias：

```text
build/kan
build/pearl-miner -> kan
```

如果不想用 symlink，可用 copy：

```text
cp kan pearl-miner
```

### 中期方案

统一所有脚本和文档：

```text
start_pool.sh
package_portable.sh
README.md
AGENTS.md
prover.h 注释
root-level deploy scripts
```

统一命名策略：

```text
主名：kan
兼容名：pearl-miner
```

### 验收

```bash
cd peral
bash build.sh
test -x build/kan
test -x build/pearl-miner || test -L build/pearl-miner
bash start_pool.sh --dry-run   # 若未来加 dry-run
```

优先级：**P0**

---

## A2. 防止 silent fallback 到 `tc_block.cu`

### 当前风险

如果 `CUTLASS_HOME` 不存在：

```text
build.sh 会 fallback 到 tc_block.cu
性能约 30 TH/s
```

这对生产是灾难级退化。

### 方案

生产启动/部署脚本必须检查 build 输出包含：

```text
CUTLASS at ... -> tc_cutlass_v2
```

如果出现：

```text
falling back to tc_block
```

默认应失败，除非显式：

```bash
ALLOW_FALLBACK=1
```

### 推荐策略

```text
开发/CI：允许 fallback，用于无 CUTLASS 环境编译保底；
生产/部署：不允许 fallback，必须 CUTLASS kernel。
```

### 验收

```text
生产日志必须包含：
tc(cutlass2)

不能只看到：
tc(block)
```

优先级：**P0**

---

## A3. 原生架构构建策略

### 当前原则

每张卡应尽量使用 native SASS：

```text
RTX 3080Ti / 3090: ARCH=sm_86
RTX 4090 / L40:   ARCH=sm_89
RTX 5090:         ARCH=sm_120 with CUDA 13+
```

5090 已实测：

```text
native sm_120 SASS 比 compute_90 PTX-JIT 快约 3.5%
```

### 方案

部署脚本输出并记录：

```text
GPU model
compute capability
ARCH
CUDA version
是否 native SASS
```

如果是 portable 包：

```text
明确提示 Blackwell 可能走 PTX-JIT；
需要最高性能时在 CUDA 13 host 上 ARCH=sm_120 重建。
```

优先级：**P1**

---

## A4. start/restart 安全

### 当前风险点

```text
start_pool.sh 直接生成 _run_loop.sh；
旧 PID / 旧 binary / 旧日志可能干扰；
没有明确 dry-run；
没有明确输出当前 kernel 类型；
```

### 建议

未来维护版 `start_pool.sh` 应加入：

```text
1. dry-run 模式；
2. build 产物检查；
3. CUTLASS kernel 检查；
4. binary hash / mtime 输出；
5. 启动前打印 ARCH/GROUPM/KSTAGES/TC_PERSIST；
6. 启动后 30 秒检查日志中是否出现 tc(cutlass2) 或 stats；
7. 如果出现 fallback 或 launch err，自动停止并报错。
```

优先级：**P1**

---

# 主线 B：端到端 wall-clock 收益收敛

## B1. 先建立分段性能观测

当前 kernel-only 已经很强，继续只看：

```text
tc(cutlass2): FUSED ... TH/s
```

不够。必须明确 wall-clock gap 来自哪里。

### 需要分段计时

建议在 verbose / breakdown 模式下记录：

```text
gpu_prep_phase1:
  rng
  tree_hash_A
  tree_hash_B

host:
  seed derive
  permutation generation

gpu_prep_phase2:
  perm H2D
  noise A
  noise B

tc_search_launch:
  small H2D config copies
  gather A
  gather B
  search FUSED
  D2H win flag/result

mine loop:
  total draw elapsed
  overlap efficiency
```

### 指标

```text
kernel TH/s
MINE done TH/s
pool 10s / 60s / 15m TH/s
wall/kernel ratio
```

重点关注：

```text
4090 wall/kernel ratio = 190-220 / 260 = 73%-85%
```

这说明 4090 端到端仍有实际收益空间。

优先级：**P1**

---

## B2. gather_rows 优化评估

当前 search 前仍做：

```cpp
gather_rows(dA  -> dAp)
gather_rows(dBt -> dBtp)
```

而 `FUSED ms` 不包含 gather。

### 风险

如果 gather 占 draw 的显著比例，那么：

```text
kernel-only 看起来很强；
pool wall-clock 仍被 gather/prep 拖低。
```

### 推进方式

第一步：

```text
只测，不改。
```

记录：

```text
gather A ms
gather B ms
gather bytes
effective GB/s
是否被 prep/search overlap 隐藏
```

第二步，如果确认为瓶颈：

```text
gpu_prep_phase2 直接生成 gathered noised layout
dAp/dBtp 双缓冲
search 直接读 gathered buffer
```

### 双缓冲设计

需要：

```text
dAp0 / dBtp0
dAp1 / dBtp1

draw N search 读 cur
draw N+1 prep 写 next
```

否则 prep 会覆盖 search 正在读的数据。

### 预期收益

```text
kernel-only 不变；
wall-clock 可能提升 5%-15%，尤其 4090。
```

优先级：**P1**

---

## B3. GPU prep 小优化

当前 `gpu_prep.cu` 已解决大问题：

```text
CPU prep 约 1490ms + 1GB H2D
```

但仍可优化：

```text
1. host permutation generation；
2. perm H2D；
3. tree hash 多层 kernel launch；
4. stream sync；
5. phase1/phase2 与 search overlap 的真实程度。
```

建议：

```text
优先 Nsight Systems / cudaEvent 量化；
不凭猜测重写。
```

可选优化：

```text
1. permutation 生成上 GPU；
2. phase1/phase2 合并部分 kernel；
3. CUDA Graph 降 launch overhead；
4. prep buffer 生命周期清晰化。
```

优先级：**P2**

---

## B4. pool 侧真实收益评估

每次性能改动都必须最终看：

```text
pool 60s TH/s
accepted/rejected
power
efficiency
```

建议最小真池测试：

```text
10 分钟：smoke
30 分钟：性能确认
2 小时：稳定性确认
```

记录：

```text
accepted
rejected
stale / invalid
job 更新频率
矿池显示 TH/s
本地 60s TH/s
```

优先级：**P1**

---

# 主线 C：代码重复与结构债治理

## C1. 抽 `tc_common.cuh`

### 当前重复

`tc_block.cu` 与 `tc_cutlass_v2.cu` 重复：

```text
rotr32 / rotl32d
BLAKE3 IV / message schedule
jackpot_blake3
le_u256
gather_rows
words_from_le32
DevBufs / ensure_dev_bufs
tc_jackpot_search ABI wrapper 部分流程
```

### 风险

```text
1. BLAKE3 constants 漂移；
2. bound compare 漂移；
3. gather semantics 漂移；
4. fallback 与 primary ABI 漂移；
5. 修改一个 kernel 漏另一个。
```

### 方案

新增：

```text
src/tc_common.cuh
```

初期只抽低风险纯函数：

```cpp
rotr32
rotl32d
le_u256
words_from_le32
gather_rows
jackpot_blake3 constants + function
```

第二阶段再考虑：

```text
DevBufs / ensure_dev_bufs
```

### 验收

必须同周期跑：

```bash
bash build.sh
bash cpu_test.sh
./build/plainproof_gen --cfg real --mine 3
bash verify_run.sh
pool short run
```

优先级：**P1**

---

## C2. 拆 `plainproof_gen.cpp`

### 当前问题

```text
mine_plain_proof() 约 696 行。
```

混合职责：

```text
config
draw generation
CPU path
GPU path
async overlap
POSTCHECK
Merkle
bincode
MineResult
CLI
```

### 风险

```text
1. draw N / draw N+1 状态错位；
2. win 后 CPU re-derive 缺变量；
3. GPU path 与 CPU path 隐式共享状态；
4. proof assembly 修改容易影响矿池 share。
```

### 拆分目标

建议未来拆成：

```text
plain_config.h/.cpp
draw_cpu.h/.cpp
draw_gpu.h/.cu 或 gpu_prep 接口
search_runner.h/.cpp
postcheck.h/.cpp
merkle_proof.h/.cpp
bincode_plainproof.h/.cpp
plainproof_gen.cpp 只保留 orchestration / CLI
```

### 拆分顺序

先拆低风险纯 CPU 辅助：

```text
1. bincode writer
2. base64 / hex helpers
3. Merkle proof writer
4. config builder
```

再拆高风险路径：

```text
5. CPU draw producer
6. GPU prep coordinator
7. search loop
8. postcheck
```

原则：

```text
不在同一个 commit 里同时做重构和性能改动。
```

优先级：**P1/P2**

---

## C3. 拆 `miner_main.cpp`

当前职责混合：

```text
CLI
NVML
stats
stratum
solo RPC
pool reader
submit
driver loop
```

建议拆成：

```text
miner_options.h/.cpp
miner_stats.h/.cpp
nvml_monitor.h/.cpp
stratum_client.h/.cpp
solo_client.h/.cpp
miner_driver.h/.cpp
miner_main.cpp
```

优先级低于 `plainproof_gen.cpp`，因为它不是当前 correctness 热点。

优先级：**P2**

---

## C4. profiling / experiment 宏治理

当前 `tc_cutlass_v2.cu` 内存在多种宏：

```text
SMALL_TILE
KSTAGES
GROUPM
A2_REG_TRANSCRIPT
PROFILE_NOFOLD
PROFILE_NOREDUX
PROFILE_NOJACKPOT
FOLD_RMW_ALWAYS
```

建议治理：

```text
1. production 默认宏列表写入文档；
2. profiling-only 宏集中到 bench scripts；
3. 每个 profiling 宏都输出 “WRONG result / timing-only”；
4. CI 禁止 profiling 宏进入 release build；
5. benchmark 结果自动记录 ptxas regs/spills。
```

优先级：**P2**

---

# 主线 D：能效、benchmark、文档治理

## D1. 能效曲线

当前 kernel 已贴近 hardware roofline。下一步应关注：

```text
TH/s/W
```

而不只是最高 TH/s。

### 建议 power sweep

RTX 5090：

```text
450W / 500W / 550W / 575W
```

RTX 4090：

```text
350W / 400W / 450W
```

RTX 3090：

```text
280W / 320W / 350W
```

记录：

```text
kernel TH/s
wall TH/s
power W
GH/W
temperature
clock
accepted/rejected
```

目标：

```text
找到长期运行收益最优点，而不是峰值跑分点。
```

优先级：**P2**

---

## D2. benchmark 规范化

建立统一 benchmark 输出：

```text
bench/results/YYYYMMDD_GPU_ARCH.csv
bench/results/YYYYMMDD_GPU_ARCH.md
```

字段：

```text
date
commit
branch
GPU
driver
CUDA
ARCH
GROUPM
KSTAGES
TC_PERSIST
NVCC_EXTRA
kernel ms/draw
kernel TH/s
MINE done TH/s
pool 60s TH/s
power
GH/W
POSTCHECK
official verifier
accepted
rejected
notes
```

优先级：**P2**

---

## D3. DCE-proof profiling 规范

`dev/fold` 的经验必须固化：

```text
NOFOLD 这类宏可能被 ptxas DCE，产生假地板。
```

未来所有性能假设必须有：

```text
1. DCE-proof TRUEFLOOR；
2. ptxas regs/spills；
3. SASS spot-check；
4. correctness run；
5. release build 对照。
```

优先级：**P1/P2**

---

## D4. 文档权威层级

建议明确文档层级：

```text
BRANCHES.md
  当前各架构权威状态、性能结论、分支策略。

README.md
  用户安装、运行、性能表。

DESIGN_speedup.md
  历史设计、方法学、已收口路线。

MAINTENANCE_OPTIMIZATION_PLAN.md
  后续维护和推进计划。

NEXT_SPEEDUP_PLAN.md
  若存在，应标注为历史方案或更新为最新结论。
```

尤其要避免：

```text
历史 fold 40% 判断
WGMMA 500+ 短期路线
SMALL_TILE 高收益假设
```

继续误导后续开发。

优先级：**P1**

---

# 4. 未来 30 天推进计划

## 第 1 周：生产一致性与可观测性

目标：

```text
不跑错、不 fallback、可准确看到 wall gap。
```

任务：

```text
1. 修复 kan / pearl-miner 命名兼容；
2. start_pool.sh 支持 kan；
3. 生产启动拒绝 CUTLASS fallback；
4. 增加 build/start 输出 kernel 类型；
5. 增加 gather/prep/search 分段计时；
6. 清理 temp_check_release.ps1；
7. 更新文档权威层级。
```

验收：

```text
bash build.sh
bash cpu_test.sh
POSTCHECK ok=1
start_pool.sh 能启动当前 build 产物
日志明确 tc(cutlass2)
```

---

## 第 2 周：4090 wall gap 分析与优化决策

目标：

```text
确认 4090 wall 190-220 vs kernel 260 的差距来自哪里。
```

任务：

```text
1. 在 4090 上跑 30 分钟 pool；
2. 记录 prep/gather/search/ms；
3. 记录 power / temp / clock；
4. 判断 gather 是否值得重构；
5. 若 gather 明显，设计 dAp/dBtp 双缓冲方案。
```

验收：

```text
有一张完整 breakdown 表：
prep phase1 / phase2 / gather A/B / search / total / pool 60s
```

---

## 第 3 周：代码重复治理

目标：

```text
降低 tc_block / tc_cutlass_v2 漂移风险。
```

任务：

```text
1. 抽 tc_common.cuh 第一阶段；
2. 只抽纯 device/common；
3. 不改变算法；
4. 完整 correctness 验证；
5. pool 短测。
```

验收：

```text
diff 中不应出现算法逻辑变化；
POSTCHECK ok=1；
official verifier VALID；
pool accepted / 0 rejected。
```

---

## 第 4 周：能效和 benchmark 自动化

目标：

```text
形成可重复 benchmark 和长期收益最优配置。
```

任务：

```text
1. power sweep；
2. TC_PERSIST per GPU A/B；
3. GROUPM per GPU 低频 sweep；
4. benchmark CSV/MD 输出；
5. 更新 README 性能表。
```

验收：

```text
每张主力 GPU 至少一份 TH/s/W 曲线；
README 中性能数据可追溯到 benchmark result。
```

---

# 5. 未来 90 天推进计划

## 5.1 稳定分支策略

建议维持：

```text
main:
  通用 / sm_86 / portable 稳定主线。

arch/4090:
  sm_89 原生 SASS 和 4090 配置。

arch/5090:
  sm_120 原生 SASS 和 5090 配置。

dev/*:
  所有实验分支，不能直接生产。
```

合并规则：

```text
dev -> main:
  必须 POSTCHECK + verifier + benchmark。

main -> arch/*:
  rebase/merge 后必须 on-card smoke。
```

---

## 5.2 plainproof_gen 结构拆分

按低风险到高风险：

```text
1. bincode writer；
2. Merkle proof builder；
3. config/context；
4. CPU draw producer；
5. GPU prep coordinator；
6. search loop；
7. postcheck。
```

每步单独验证，不混性能改动。

---

## 5.3 direct gathered layout 是否实施

只有当 Week 2 profiling 证明 gather 是显著 wall gap 时实施。

实施前必须设计：

```text
dAp/dBtp double buffer
draw id ownership
stream/event dependency
win 后 CPU rederive 不变
fallback path 不破
```

---

## 5.4 WGMMA/TMA 保留为研究项

当前不作为主线。只有满足：

```text
1. 有明确硬件；
2. pure WGMMA dense-int8 roofline 明显高于现有；
3. 可以 fused tiled fold；
4. 不物化 full C；
5. 有完整验证预算；
```

才启动。

---

# 6. 每次改动的标准验收流程

## 6.1 文档/脚本改动

```bash
git diff
bash -n build.sh
bash -n start_pool.sh
bash -n package_portable.sh
```

如果涉及启动：

```text
dry-run 或 staging box 启动验证。
```

## 6.2 CPU/proof 改动

```bash
cd peral
bash build.sh
bash cpu_test.sh
bash verify_run.sh
```

## 6.3 GPU/prep/kernel 改动

```bash
cd peral
bash build.sh
./build/plainproof_gen --cfg real --mine 3 --breakdown
bash verify_run.sh
```

必须看到：

```text
POSTCHECK ok=1
official verifier VALID
```

## 6.4 生产候选

```text
pool short run 10-30 分钟
accepted > 0
rejected = 0
60s TH/s 正常
power/temp 正常
```

---

# 7. 风险登记表

| 风险 | 等级 | 表现 | 预防 |
|---|---:|---|---|
| binary 命名不一致 | P0 | start_pool 启动失败或跑旧版本 | `kan` + `pearl-miner` alias |
| CUTLASS fallback | P0 | 退回 30 TH/s | 生产禁止 fallback |
| profiling DCE 误判 | P1 | 假地板、错误优化方向 | TRUEFLOOR + ptxas/SASS |
| giant function 修改破 proof | P1 | POSTCHECK fail / rejected | 拆分 + 小步验证 |
| tc_common 抽取引入漂移 | P1 | fallback/primary 不一致 | 同周期完整验证 |
| gather 优化破 overlap | P1 | wall 下降或数据竞争 | double buffer + events |
| power 过高不增收益 | P2 | TH/W 低、温度高 | power sweep |
| WGMMA 黑盒 OOM | P3 | 68GB C / OOM | 只做 fused tiled，不用 device::Gemm 黑盒 |

---

# 8. 推荐优先级总表

| 优先级 | 项目 | 类型 | 预期收益 |
|---:|---|---|---|
| P0 | 修 `kan` / `pearl-miner` 命名兼容 | 生产稳定 | 避免启动失败/跑旧版本 |
| P0 | 生产禁止 CUTLASS fallback | 生产稳定/性能 | 避免 350/260/112 TH/s 退回 30 TH/s |
| P1 | prep/gather/search 分段计时 | 可观测性 | 找准 wall gap |
| P1 | 4090 wall gap profiling | 性能收益 | 可能 +5%-15% wall |
| P1 | 文档权威层级更新 | 维护 | 避免旧假设误导 |
| P1 | 抽 `tc_common.cuh` 第一阶段 | 技术债 | 降低 correctness 漂移 |
| P1/P2 | 拆 `plainproof_gen.cpp` | 技术债 | 降低 proof/prep 修改风险 |
| P2 | 能效 power sweep | 收益优化 | 提升 TH/W |
| P2 | benchmark CSV/MD 自动化 | 工程化 | 数据可追溯 |
| P3 | WGMMA/TMA fused tiled 研究 | 长期实验 | 仅在新硬件/新 roofline 证明后做 |

---

# 9. 最终执行建议

未来推进不要再以：

```text
“还能不能 kernel 2x”
```

作为主问题。

应该改成：

```text
1. 生产是否一定跑的是最快正确 binary？
2. wall-clock 是否接近 kernel-only？
3. TH/s/W 是否最优？
4. 每个性能结论是否可复现？
5. 每次重构是否不影响 POSTCHECK/verifier/pool accepted？
```

一句话方案：

```text
先补生产一致性和可观测性；
再用数据决定是否做 gather/double-buffer；
随后治理 tc_common 和 plainproof_gen 结构债；
最后做能效曲线和 benchmark 自动化。
```

这比继续盲目重写 kernel 更符合当前项目阶段，也更能提升真实收益和长期可维护性。

