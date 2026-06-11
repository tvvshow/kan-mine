# Pearl Miner — 项目结构与模块化纪律

更新：2026-06-10（Week1 Day2）。本文件是结构的唯一权威地图；改动结构必须同步更新这里。

## 模块总览

```
peral/
├── build.sh                 构建入口（唯一）：blake3 + prover + miner + tc_block + zkprove
├── README.md                项目说明
├── STRUCTURE.md             本文件：结构地图 + 模块化纪律
├── ROADMAP_106TH.md         4 周推进计划（Week1=mainloop 重写）
├── DESIGN_speedup.md        [权威] 性能分析 + 重写计划（箱上实测数据）
├── DESIGN_vs_official.md    正确性对照（官方源码 ↔ 我们的实现，全部吻合）
│
├── src/                     ★ 生产源码（build.sh 只链接这里）
│   ├── plainproof_gen.cpp       核心 prover：A/B 生成、噪声、fold、Merkle、bincode
│   │                            （-DPROVER_LIB 去 main() → 同源两用）
│   ├── miner_main.cpp           统一矿工驱动（--pool stratum / --solo RPC）
│   ├── prover.h                 prover ↔ miner 接口
│   ├── tc_block.cu              [LIVE] WMMA 30 TH/s — 直播挣钱内核（153/0 shares）
│   │                            build.sh 当前唯一链接的 GPU 内核
│   ├── tc_imma2.cu              [REFERENCE] IMMA 56 TH/s @BN=128 — 已验证、未部署
│   │                            禁止再就地改参数（教训：BN=256 污染事故）
│   ├── tc_deep_pipeline.cu      [EXPERIMENT] 参数化 CUTLASS 形 mainloop
│   │                            -D 旋钮：BN/STAGES/RSUB/REG_PIPE/MINBLOCKS/SWIZZLE
│   │                            （SWIZZLE=smem XOR 置换，修 4-way bank 冲突，
│   │                            CPU 证明见 experiments/verify_swizzle.py）
│   │                            sweep 的对象；赢家将取代 tc_block
│   ├── gpu_draw.cu / .h         GPU 端 RNG+噪声（Week4 接入）
│   └── （已知债：blake3/gather/DevBufs/wrapper ~200 行在 3 个 tc_*.cu 中
│             重复 ×3 — 见下方「模块化纪律」第 3 条，sweep 定论后合并）
│
├── bench/                   ★ 独立微基准（自带 main，随机数据 CPU 对照）
│   ├── mma_microtest.cu         mma.sync.m16n8k32 片段布局验证（128/128 cells）
│   ├── mma_ldmatrix_test.cu     ldmatrix.x4/.x2 寻址验证
│   ├── mma_warpgemm_test.cu     单 warp 全 K=4096 GEMM 验证
│   └── cutlass_int8_bench.cu    CUTLASS roofline：131.7 TMAC/s @128×256（箱上实测）
│
├── notes/                   ★ 过程笔记 + 实验 harness（不进构建）
│   └── week1/
│       ├── day1_*.md            CUTLASS mainloop / sm_80 架构分析
│       └── sweep_pipeline.sh    [当前关键] tc_deep_pipeline 配置扫描 + 正确性门
│
├── experiments/             一次性实验（test_gpu_rng.cu、verify_swizzle.py）
├── archive/                 已废弃内核（tc_imma BROKEN、imma_* 早期实验）— 只读
│
├── blake3/                  [VENDORED] 官方 BLAKE3 C + SIMD asm
├── cutlass/                 [VENDORED] CUTLASS v3.5.1（只读参考 + bench 依赖）
├── zk-pow/                  [VENDORED] 官方 Rust verifier + zkprove（--solo 用）
├── plonky2/ pearl-blake3/   [VENDORED] zk-pow 的依赖
└── oracle/                  lpminer share-dump 离线测试数据（真实网络 config）
```

仓库根（D:\mybitcoin\3）：`sshrun.py`（箱驱动，密码只从 env BOXPW），`_archive/`（旧驱动/死内核隔离区）。

## 模块化纪律（违反 = 返工）

1. **一个职责一个地方**：生产代码只在 `src/`；验证只在 `bench/`；实验 harness
   只在 `notes/weekN/`；废弃只进 `archive/`（不删除，但绝不 include/链接）。
2. **参考实现冻结**：标 [LIVE]/[REFERENCE] 的文件禁止就地实验。要试参数 →
   `tc_deep_pipeline.cu` 用 `-D` 旋钮。每个实验配置必须可由命令行一行复现。
3. **重复代码的处理时机**：3 个 tc_*.cu 重复 blake3/gather/DevBufs/wrapper 是
   已知债。合并方案 = 抽 `src/tc_common.cuh`（device 函数 + DevBufs）。但合并
   触碰 [LIVE] 内核 = 必须过完整验证（POSTCHECK ok=1 + 官方 verifier VALID），
   所以定在 sweep 决出赢家、整合进 build.sh 的同一个验证周期里一次完成 —
   不为纯结构改动单独消耗一次箱上验证。
4. **vendored 目录只读**：blake3/cutlass/zk-pow/plonky2/pearl-blake3 不改一行。
   需要包装 → 在 src/ 写 adapter。
5. **构建入口唯一**：所有产物经 build.sh。sweep 类临时构建放 build/ 临时文件，
   不污染脚本化构建。

## 当前状态一句话

tc_block 30 TH/s 直播挣钱；tc_imma2 56 TH/s 已验证待超越；tc_deep_pipeline
（BN256/s3/RSUB64/REG_PIPE1 = CUTLASS 配方）等箱上 sweep
（`bash notes/week1/sweep_pipeline.sh`）决定：手写 mainloop 能否逼近 131.7
TMAC/s roofline 且零 spill —— 能 → 整合+验证；不能 → 切 CUTLASS warp 原语
（DESIGN_speedup.md §5.3）。
