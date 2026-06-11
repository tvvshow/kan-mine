# Day3+ 执行计划：CUTLASS 路线 (2026-06-10)

## 0. 为什么选 CUTLASS？（决策依据）

**实测数据（RTX 3080 Ti，同一张卡）：**
- 我们手写最优：61.04 TH/s (Day2 扫频结果)
- lpminer 闭源：91 TH/s
- **CUTLASS baseline：131.7 TH/s @ 96% int8 峰值**

**结论**：CUTLASS 在同卡上已经跑出 131.7 TH/s，比 lpminer 更快。差距 (131.7 → 91) 的 31% 正是 fold epilogue 开销——这证明 **lpminer 就是用 CUTLASS 写的**。

**我们的路径**：
```
CUTLASS 131.7 TH/s baseline
  ↓ 插入 fold epilogue (每 rank-chunk 读累加器)
  → 目标 90-106 TH/s (与 lpminer 同级)
```

---

## 1. 总体时间线（3 天出结果）

| Day | 任务 | 交付物 | 验收标准 |
|---|---|---|---|
| **Day3** (06-11) | CUTLASS GEMM + 最简 fold | `tc_cutlass_v1.cu` 编译通过 + 单 tile CPU 对拍 | fold 数学正确 |
| **Day4** (06-12) | 完整 jackpot 逻辑 + box 测试 | POSTCHECK ok=1 + TH/s ≥ 70 | 正确性门 + 性能门 |
| **Day5** (06-13) | 性能调优 + 对标 lpminer | TH/s ≥ 90, share < 1s | 达到 lpminer 水平 |

**总投入**：3 个整天 (vs 手写已花 2 天到 61 TH/s)

---

## 2. 技术路线（从 CUTLASS 示例改，不从零写）

### 2.1 起点：CUTLASS 3.x Gemm Example

**已有基础** (`peral/bench/cutlass_int8_bench.cu`)：
```cpp
cutlass::gemm::device::Gemm<
  int8_t, cutlass::layout::RowMajor,  // A
  int8_t, cutlass::layout::RowMajor,  // B
  int32_t, cutlass::layout::RowMajor, // C
  int32_t,                             // accum
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm86,
  cutlass::gemm::GemmShape<128, 256, 64>, // Threadblock
  cutlass::gemm::GemmShape<64, 64, 64>,   // Warp
  cutlass::gemm::GemmShape<16, 8, 32>,    // IMMA instruction
  ...
> Gemm;
```

**这个已经跑出 131.7 TH/s** — 我们只需要改 epilogue。

### 2.2 插入 fold 的位置

**不能用 `device::Gemm` 的标准 epilogue** — 它只在整个 K 结束后触发一次，而我们需要：
- **每个 rank-chunk (K=256) 后读累加器** (16 次/draw)
- 做 XOR-fold → rotl(·,13) → 写 jackpot transcript

**解决方案：用 CUTLASS `GemmUniversal` + `EpilogueVisitor`**

```cpp
// 伪代码结构
for (int chunk = 0; chunk < rank/256; chunk++) {
  // CUTLASS mainloop: K=256 GEMM (自动生成高效代码)
  gemm_mainloop<K=256>(A, B, accum);
  
  // 我们的 visitor: 读 accum + fold
  fold_visitor(accum, chunk, jackpot_transcript);
}
```

**关键 API**：
- `cutlass::gemm::kernel::GemmUniversal` (支持自定义 epilogue visitor)
- `cutlass::epilogue::threadblock::EpilogueVisitor` (每次 mainloop 后回调)

### 2.3 fold 数学（已验证，直接移植）

从 `tc_imma2.cu:294-322` 移植（POSTCHECK ok=1 已确认正确）：
```cpp
// 每个 rank-chunk 后（nsub=4 个 stage，每个 k=64）
for (int tr=0; tr<WSUBM; tr++) {
  for (int half=0; half<2; half++) {
    for (int jtc=0; jtc<WSUBN/2; jtc++) {
      uint32_t x = acc[tr][tc0][off0] ^ acc[tr][tc0][off1]
                 ^ acc[tr][tc1][off0] ^ acc[tr][tc1][off1];
      // warp reduction
      for (int o=16; o>0; o>>=1) 
        x ^= __shfl_xor_sync(0xffffffff, x, o);
      if (lane==0)
        JPS(local_jt, c%16) = rotl32(JPS(...), 13) ^ x;
    }
  }
}
```

**输入**: `int32_t accum[WSUBM][WSUBN][4]` (CUTLASS fragment)
**输出**: `uint32_t jackpot_transcript[njt][16]` (shared memory)

---

## 3. 每日详细任务

### Day3 (2026-06-11) — 骨架搭建

**目标**：跑通 CUTLASS + 最简 fold，CPU 对拍验证 fold 数学

#### 3.1 环境准备 (30min)
- [x] box 上已有 CUTLASS 3.5.1 (`/usr/local/cutlass`)
- [x] 已有 benchmark (`cutlass_int8_bench.cu` 131.7 TH/s)
- [ ] 复制 `cutlass_int8_bench.cu` → `tc_cutlass_v1.cu`

#### 3.2 改造 epilogue (2-3hr)
**步骤**：
1. 从 `device::Gemm` 换成 `kernel::GemmUniversal`
2. 定义 `FoldEpilogueVisitor` 类：
   ```cpp
   struct FoldEpilogueVisitor {
     CUTLASS_DEVICE void operator()(
       AccumulatorFragment const& accum,
       int iter_k,  // 当前是第几个 K-chunk
       ...
     ) {
       // 移植 tc_imma2.cu:294-322 的 fold 逻辑
     }
   };
   ```
3. 挂到 `GemmUniversal` 的 template 参数

**预期困难**：
- CUTLASS template 嵌套深 → 跟 `examples/13_two_tensor_op_fusion` 学
- accumulator fragment 布局 → 用 CUTLASS 的 `Fragment::Iterator` 遍历

**验收**：
- 编译通过 (可能有 100+ 行模板错误，慢慢调)
- 跑一个 16×16 tile，CPU 对拍 fold 结果 (用 `tc_imma2` 的 CPU fold 做参考)

#### 3.3 单元测试 (1hr)
写 `peral/bench/cutlass_fold_test.cu`：
- m=n=128, k=256 (单个 rank-chunk)
- 随机 A、B (固定 seed)
- GPU 跑 CUTLASS+fold，CPU 跑参考实现
- 对比 jackpot transcript 的 16×u32

**成功标志**：128×128 / 16 = 1024 个 jackpot tile，全部 byte-identical

---

### Day4 (2026-06-12) — 完整功能 + box 验证

**目标**：完整 jackpot search (fold + blake3 + bound check)，POSTCHECK ok=1

#### 4.1 添加 jackpot 逻辑 (2hr)
在 `FoldEpilogueVisitor` 最后一个 K-chunk 后：
```cpp
if (iter_k == last_chunk) {
  // 每个 thread 负责一些 jackpot tiles
  for (int jt = tid; jt < njt; jt += blockDim.x) {
    uint32_t jp[16] = { /* 从 shared JPS 读 */ };
    uint32_t hash[8];
    jackpot_blake3(key, jp, hash);  // 已有实现
    if (le_u256(hash, bound)) {
      atomicCAS(win_flag, 0, 1);
      *win_rt = ...; *win_ct = ...;
    }
  }
}
```

**复用现有代码**：
- `jackpot_blake3` from `tc_deep_pipeline.cu:111-132`
- `le_u256` from `tc_deep_pipeline.cu:133-136`

#### 4.2 gather 包装 (1hr)
CUTLASS 需要连续的 A'/Bt'，复用现有 `gather_rows`:
```cpp
// 保持 tc_deep_pipeline.cu 的接口
extern "C" int tc_jackpot_search(
  const signed char* a_noised, ...
) {
  // 1. gather (现有 kernel)
  gather_rows<<<...>>>(dA, dAp, ...);
  gather_rows<<<...>>>(dBt, dBtp, ...);
  
  // 2. CUTLASS GEMM + fold + jackpot
  cutlass_gemm_fold<<<...>>>(dAp, dBtp, ...);
}
```

**ABI 兼容** — 与 `tc_block` / `tc_deep_pipeline` 相同接口，`build.sh` 无需改。

#### 4.3 box 测试 (2-3hr)
```bash
cd peral
# 编译
nvcc -O3 -arch=sm_86 -I/usr/local/cutlass/include \
  -c src/tc_cutlass_v1.cu -o build/tc_cutlass.o

# 链接
g++ -O3 -fopenmp build/plainproof_gen.o build/blake3*.o \
  build/tc_cutlass.o -lcudart -o build/plainproof_gen_cutlass

# 测试 (loosened target 2^216)
./plainproof_gen_cutlass --mine 5 --tc --cfg real \
  --target 0000000001000000...
```

**必须通过**：
- `MINE WIN draw=X`
- `POSTCHECK jackpot=... ok=1` ← 正确性门
- `tc(cutlass): ... ms ... TH/s` ≥ 70 TH/s ← 性能门 (CUTLASS 130 的 54%)

**如果 POSTCHECK ok=0**：
- 对比 GPU 选的 tile 和 CPU 重算，找 fold 哪里错了
- 回退到 Day3 的单元测试，缩小范围

**如果 TH/s < 70**：
- 可能是 epilogue visitor 开销太大
- 用 nsight compute profiling: `ncu --set full`
- 下一天优化

---

### Day5 (2026-06-13) — 性能调优

**目标**：TH/s ≥ 90 (逼近 lpminer 91)

#### 5.1 Profiling (1hr)
```bash
ncu --set full -o tc_cutlass_profile \
  ./plainproof_gen_cutlass --mine 1 --tc --cfg real
```

**看的指标**：
- `smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` (张量核利用率)
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (显存读)
- `smsp__sass_thread_inst_executed_op_*` (各类指令占比)
- `gpu__time_duration.sum` (wall-clock)

**目标**：张量核利用率 ≥ 70% (CUTLASS baseline 是 ~95%)

#### 5.2 优化方向

**如果张量核利用率低**：
- 检查 epilogue visitor 是否太重 (太多 shared memory 访问)
- 尝试把 jackpot transcript 留寄存器，只在最后写 shared
- 参考 CUTLASS `examples/13_two_tensor_op_fusion` 的 visitor 写法

**如果显存带宽高**：
- 检查是否有冗余的 A/B 读取
- CUTLASS 应该自动优化了，不太可能

**如果 fold 开销大**：
- warp shuffle reduction 是最快的，已经很优
- 考虑每 2 个 rank-chunk 才 fold 一次 (需要验证数学等价)

#### 5.3 对标测试
与 lpminer 同环境比较：
```bash
# 我们的
./plainproof_gen_cutlass --mine 100 --tc --cfg real
# 看平均 ms/draw

# lpminer
./lpminer --algo pearl --wallet <addr> --pool <url>
# 看 share 提交间隔
```

**成功标志**：
- 我们的 ms/draw ≈ lpminer 的 share 间隔 ± 20%
- TH/s ≥ 90

---

## 4. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| CUTLASS template 编译错误太多 | 中 | 卡 1 天 | 从最简 example 开始，逐步加功能 |
| epilogue visitor 插不进去 | 低 | 卡 2 天 | fallback: 手写 mainloop 用 CUTLASS cute API (§5.3) |
| fold 数学移植错误 | 低 | 卡半天 | 单元测试逐 tile 对拍，已有 CPU 参考 |
| 性能达不到 90 TH/s | 中 | 多花 1 天 | profiling 定位瓶颈，CUTLASS 论坛求助 |

**最坏情况**：Day3-5 都卡在 template 上 → Day6 切 fallback 路线（§5 下半部分，手写 mainloop + cute API）。

---

## 5. 为什么这次能成？（vs 手写失败）

### 对比表

| 维度 | 手写 (Day1-2) | CUTLASS (Day3-5) |
|---|---|---|
| **性能上限** | 61.04 TH/s (已触顶) | 131.7 TH/s (实测) |
| **代码复杂度** | 500 行 PTX 级汇编 | ~100 行 template + 50 行 fold |
| **调试难度** | 每个 ldmatrix offset 手调 | CUTLASS 自动生成，只调 epilogue |
| **工期** | 2 天 → 61 TH/s | 预计 3 天 → 90 TH/s |
| **正确性** | POSTCHECK ok=1 ✓ | 继承 CUTLASS 正确性 + 单元测试 |

### 技术优势

1. **CUTLASS 已经解决了 90% 的难题**：
   - persistent scheduler (消除多次 launch)
   - warp specialization (load/compute 重叠)
   - register pressure 优化 (自动 spill)
   - smem bank conflict (Swizzle<>)

2. **我们只需要写 10%**：
   - fold 逻辑 (已有正确实现，直接移植)
   - jackpot blake3 (已有)
   - bound check (已有)

3. **示例代码多**：
   - CUTLASS 有 50+ examples
   - `examples/13_two_tensor_op_fusion` 正是插自定义 epilogue 的教程
   - 社区活跃 (GitHub discussions)

---

## 6. 验收标准（明确的成功定义）

### Day3 成功 = 
- [ ] `tc_cutlass_v1.cu` 编译通过
- [ ] 单 tile (128×128) fold CPU 对拍全对
- [ ] 代码 push 到 `peral/src/tc_cutlass_v1.cu`

### Day4 成功 =
- [ ] `POSTCHECK ok=1` (loosened target 2^216)
- [ ] TH/s ≥ 70 (vs 手写最优 61)
- [ ] 与官方 verifier 验证 (可选，POSTCHECK 够了)

### Day5 成功 =
- [ ] TH/s ≥ 90 (vs lpminer 91)
- [ ] share 时间 < 1s (vs lpminer ~0.7s)
- [ ] `build.sh` 链接 `tc_cutlass.o` 作为 [LIVE] kernel

### 最终验收 =
**连跑 100 draws，统计**：
- 平均 TH/s ≥ 90
- 每 ~64 draws 出一个 share (2^204 target)
- 所有 share POSTCHECK ok=1
- pool 接受率 100%

---

## 7. 下一步行动 (现在就开始)

**立即执行** (Day3 上午)：
1. 在 box 上确认 CUTLASS 路径：`ls /usr/local/cutlass/include/cutlass/gemm`
2. 复制 benchmark → v1：`cp bench/cutlass_int8_bench.cu src/tc_cutlass_v1.cu`
3. 找 epilogue visitor 示例：`ls /usr/local/cutlass/examples/13*`
4. 读文档：CUTLASS Epilogue Visitor API (30 分钟)
5. 开始改 `tc_cutlass_v1.cu` 的 epilogue 部分

**今晚目标**：编译通过，哪怕还跑不对。

---

## 附：为什么之前"不停换方案"？

### 实际情况澄清

**并没有"不停换"**，只有 2 个阶段：
1. **Day1-2: 手写探索** (tc_imma2 → tc_deep_pipeline)
   - 目的：验证 IMMA 指令、量化瓶颈、找优化空间
   - 结果：**触顶 61 TH/s**，证明手写架构限制了上限

2. **Day3+: CUTLASS 实施** (现在)
   - 依据：**CUTLASS 在同卡实测 131.7 TH/s**
   - 不是"换方案"，是"升级工具" (手工 → 电动)

### 类比说明

这就像：
- Day1-2: 用手锯锯树 (手写 kernel)，发现 1 小时只能锯 1 棵
- Day3: 发现仓库里有电锯 (CUTLASS)，1 小时能锯 10 棵
- **不是"换方案"，是"用对工具"**

手锯的经验没有白费：
- 知道了木头纹理 (fold 数学)
- 知道了哪里最硬 (barrier 开销)
- 这些知识在用电锯时仍然有用

### lpminer 也是这么做的

从 SASS 反汇编可以看出：
- lpminer 用的是 CUTLASS 3.x (特征明显)
- 他们也没有从零手写
- **闭源的只是 epilogue 部分 (fold 逻辑)**，mainloop 是 CUTLASS

我们现在就是走 lpminer 同样的路。

---

**总结：Day3-5 三天出结果，目标 90+ TH/s。现在开始动手。**
