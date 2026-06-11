# Pearl Miner 齐平 SRBMiner 106 TH/s 路线图

**目标**：重写 CUTLASS 级别的 kernel，从当前 56 TH/s 提升到 106 TH/s（端到端）

**时间**：4 周（2026-06-10 开始）

**当前状态**（2026-06-10）：
- 内核：tc_imma2.cu = 56 TH/s (1256ms/draw, 128×128 RSUB=64)
- 端到端：12.55 TH/s (5.61s/draw, CPU 2.3s 瓶颈)
- 差距：内核 1.9×, 端到端 8.4×

---

## 差距根因（已实测验证）

| 技术 | 我们 | SRBMiner | 差距来源 |
|---|---|---|---|
| GEMM roofline | 56 TH/s (41% 峰值) | ~106 TH/s (78% 峰值) | **架构差距** |
| Tile 大小 | 128×128 | 128×256 | A 复用不足 |
| Warp 设计 | 单一职责 | warp-specialized | 寄存器压力集中 |
| 调度 | grid launch | persistent | launch 开销 |
| 流水深度 | STAGES=2 | deep software pipeline | 延迟隐藏不足 |
| 噪声生成 | CPU 2.3s | GPU | CPU 墙 |

**核心结论**：不是"某个参数"差距，是**整套工程架构**差距。需要重写。

---

## Week 1: CUTLASS GEMM mainloop（目标 100+ TH/s 纯 GEMM）

### Day 1-2: 研究 CUTLASS 源码
- [ ] 下载 CUTLASS 3.5.1：`git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git`
- [ ] 定位关键文件：
  - `include/cutlass/gemm/kernel/gemm_universal.hpp`（kernel entry）
  - `include/cutlass/gemm/collective/sm80_mma_multistage.hpp`（warp-specialized mainloop）
  - `include/cutlass/arch/mma_sm80.h`（mma.sync 封装）
- [ ] 理解架构：
  - Producer warp：async copy gmem→smem
  - Consumer warp：ldmatrix smem→reg + mma.sync
  - Ping-pong smem buffer (multi-stage pipeline)
- [ ] 记录到 `notes/cutlass_architecture.md`

### Day 3-4: 最小可用 GEMM kernel
- [ ] 创建 `src/tc_cutlass_base.cu`
- [ ] 复制 CUTLASS 组件：
  - Tile: 128(M) × 256(N) × 64(K)
  - Warp tile: 64 × 64
  - Stages: 4
  - mma.sync.m16n8k32.row.col.s32.s8.s8.s32
- [ ] 输出：纯 int32 累加器到 global memory（C[m×n]）
- [ ] 编译测试：`nvcc -O3 -arch=sm_86 -I./cutlass/include -c src/tc_cutlass_base.cu`
- [ ] 基准：与 `bench/cutlass_int8_bench.cu` 对比速度
- [ ] **验收**：≥ 125 TMAC/s (接近 CUTLASS 131.7)

### Day 5-7: 集成到 plainproof_gen
- [ ] 保留 tc_imma2 的 gather/blake3/host wrapper
- [ ] 替换 `tc_jackpot_search` 的 GEMM 部分为 tc_cutlass_base
- [ ] 暂时跳过 fold（纯 GEMM 输出 C，CPU 验证结果）
- [ ] 编译链接：`build/plainproof_cutlass_base`
- [ ] **验收**：`--mine 1 --cfg real` POSTCHECK ok=1（C 矩阵正确）

**Week 1 交付物**：`tc_cutlass_base.cu`，纯 GEMM ~125 TH/s，POSTCHECK ok=1

---

## Week 2: 自定义 epilogue（per-rank-chunk fold）

### Day 8-9: 设计 fold 插入点
- [ ] 分析 CUTLASS mainloop 的 K-loop 结构
- [ ] 确定插入点：每处理 rank=256 K-steps 后
- [ ] 设计累加器访问接口（consumer warp 寄存器）
- [ ] 设计 smem transcript 布局：`jp_sh[njt][16]`（每 tile 16×u32）
- [ ] 绘制数据流图到 `notes/fold_design.md`

### Day 10-12: 实现 register-only fold
- [ ] 复用 tc_imma2 的 fold 逻辑：
  - warp-shuffle XOR-reduce（`__shfl_xor_sync`）
  - rotl13: `jp[tile][c%16] = rotl(jp, 13) ^ x`
- [ ] 修改 mainloop：
  ```cpp
  for (int k_chunk = 0; k_chunk < nchunks; k_chunk++) {
    // ... GEMM 256 K-steps ...
    if ((k_chunk + 1) % (rank / K_PER_STAGE) == 0) {
      fold_rank_chunk(acc, jp_sh, k_chunk);  // <- 插入
    }
  }
  ```
- [ ] 验证：与 tc_imma2 逐 tile 对比 jp_sh 内容
- [ ] **验收**：fold 结果 byte-identical

### Day 13-14: 集成 blake3 + bound check
- [ ] 整 K 后，每 tile 读 jp_sh[tile][0..15]
- [ ] 调用 `jackpot_blake3(key, jp, out)`（复用 tc_imma2）
- [ ] `if (le_u256(out, bound)) atomicCAS(win_flag, 0, 1)`
- [ ] 测试非饱和 bound：`--target 0000004000000...`
- [ ] **验收**：`--mine 5 --cfg real` POSTCHECK ok=1，MINE WIN

**Week 2 交付物**：`tc_cutlass_fold.cu`，GEMM+fold ~100 TH/s，POSTCHECK ok=1

---

## Week 3: Persistent scheduler + 优化

### Day 15-16: Persistent tile scheduler
- [ ] 实现 tile 分配器：
  ```cpp
  __shared__ int tile_counter;
  if (threadIdx.x == 0) tile_counter = atomicAdd(&global_tile_idx, 1);
  __syncthreads();
  while (tile_counter < total_tiles) {
    int my_tile = tile_counter;
    // ... GEMM + fold for my_tile ...
    if (threadIdx.x == 0) tile_counter = atomicAdd(&global_tile_idx, 1);
    __syncthreads();
  }
  ```
- [ ] Grid size：`68 SMs × 2 blocks/SM = 136 blocks`（固定）
- [ ] 验证：与非 persistent 版本结果逐 tile 对比
- [ ] **验收**：结果完全一致

### Day 17-19: 性能调优
- [ ] 扫描参数矩阵：
  | STAGES | blocks/SM | RSUB | 预期 TH/s | 实测 |
  |---|---|---|---|---|
  | 3 | 2 | 64 | ~100 | ? |
  | 4 | 2 | 64 | ~105 | ? |
  | 4 | 1 | 128 | ~90 | ? |
  | 5 | 2 | 64 | ~108 | ? |
- [ ] 记录每组配置的 registers/smem/occupancy（`nvcc --ptxas-options=-v`）
- [ ] 选择最优配置
- [ ] **验收**：内核 ≥ 105 TH/s

### Day 20-21: 官方 verifier 测试
- [ ] Box 上编译官方 verifier（Rust 1.96，已有 `zkprove` binary）
- [ ] 跑 `--mine 10`，收集 10 个 proof
- [ ] 每个 proof 过 verifier：`zkprove verify <proof.bin>`
- [ ] 检查 VALID 率
- [ ] 如果有 INVALID，二分查找（与 tc_imma2 对比哪个 field 错）
- [ ] **验收**：VALID 率 100%

**Week 3 交付物**：`tc_cutlass_persistent.cu`，内核 105+ TH/s，官方 verifier VALID

---

## Week 4: GPU 噪声生成（端到端 100+ TH/s）

### Day 22-23: GPU RNG + blake3
- [ ] 已有 `gpu_draw.cu`（RNG 正确，已验证）
- [ ] 添加 GPU blake3 kernel：
  ```cpp
  __global__ void blake3_hash_matrix(const uint8_t* mat, int rows, int cols, 
                                      const uint8_t* key, uint8_t* out);
  ```
- [ ] 流程：GPU 生成 A/Bt → GPU blake3 → hash_a/hash_b → seeds（全在 device）
- [ ] 验证：与 CPU blake3 结果对比
- [ ] **验收**：GPU blake3 byte-identical

### Day 24-25: GPU 噪声生成
- [ ] Permutation matrix kernel（generate_permutation_matrix）
- [ ] Uniform row kernel（uniform_row，keyed hash + mod）
- [ ] Matvec kernel（matvec_sparse_perm，稀疏矩阵乘向量）
- [ ] 组合：noise_a = matvec(e_ar_t, uniform_row(每行))
- [ ] 验证：与 CPU noise 逐元素对比
- [ ] **验收**：GPU noise byte-identical

### Day 26-27: 端到端集成
- [ ] 修改 plainproof_gen.cpp：
  - produce_draw 改为纯 device 调用（无 H2D）
  - 仅传 seed/draw/job_key/bound（<1KB）
  - a_noised/b_noised_t 直接在 device 生成并传给 kernel
- [ ] 测量端到端时间：`--mine 10 --breakdown`
- [ ] 目标分解：
  - GPU RNG+noise: < 100ms
  - Kernel: ~1000ms (105 TH/s)
  - Total: < 1200ms → **58 draws/s → 104 TH/s**
- [ ] **验收**：端到端 ≥ 100 TH/s

### Day 28: 线上测试
- [ ] 部署到 box：`scp build/pearl-miner root@117.50.194.150:/root/`
- [ ] 连接 live pool：`./pearl-miner --pool prl.kryptex.network:7048 --wallet <addr>`
- [ ] 观察 30 分钟：
  - Hashrate 稳定在 100+ TH/s
  - Share accepted（无 reject）
  - Share 间隔 < 60s（vs SRBMiner ~10-20s，我们 diff 可能不同）
- [ ] **验收**：Pool accepted shares，零 reject

**Week 4 交付物**：完整 miner，端到端 100+ TH/s，pool accepted

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| CUTLASS 代码太复杂，移植困难 | 高 | Week 1 延期 | 回退：直接链接 libcutlass.a，只写 epilogue（放弃深度定制） |
| Fold 插入破坏性能 | 中 | Week 2 达不到 100 | 优化：减少 smem 往返，pure register fold |
| 官方 verifier INVALID | 中 | Week 3 卡住 | Debug：二分法定位错误 field，与 tc_imma2 逐步对比 |
| GPU 噪声太慢 | 低 | Week 4 端到端慢 | 已有 GPU blake3，permutation 纯并行，应该 < 100ms |
| 端到端仍达不到 100 | 中 | 整体失败 | 分析瓶颈：如果 kernel 105 但端到端 80，说明还有隐藏墙（profiling） |

---

## 检查点与回滚策略

| 周 | 最低标准 | 理想目标 | 回滚条件 |
|---|---|---|---|
| W1 | 100 TH/s 纯 GEMM | 125 TH/s | < 80 TH/s → 回退用 CUTLASS 库 |
| W2 | 80 TH/s GEMM+fold | 100 TH/s | POSTCHECK ok=0 → 回 tc_imma2 fold 逻辑 |
| W3 | 90 TH/s persistent | 105 TH/s | < 70 TH/s → 放弃 persistent，用 grid launch |
| W4 | 80 TH/s 端到端 | 100 TH/s | < 60 TH/s → 保留 CPU 噪声，只优化 kernel |

---

## 当前 repo 状态（需整理）

### 已验证可用
- `src/tc_block.cu`：WMMA 30 TH/s，153/0 shares accepted（live earner）
- `src/tc_imma2.cu`：IMMA 56 TH/s，POSTCHECK ok=1（faster but not deployed）
- `bench/mma_*.cu`：IMMA primitives 验证（CPU 对拍全通过）
- `bench/cutlass_int8_bench.cu`：CUTLASS roofline 实测 131.7 TH/s

### 实验品（可归档）
- `src/tc_imma.cu`：BROKEN（0 命中，已证伪）
- `src/tc_gemm.cu`：旧 dp4a 版本（已被 tc_block 替代）
- 所有 `*_test.cu` / `*_old.cu`

### 需要创建
- `src/tc_cutlass_base.cu`（Week 1）
- `src/tc_cutlass_fold.cu`（Week 2）
- `src/tc_cutlass_persistent.cu`（Week 3）
- `src/gpu_noise.cu`（Week 4）
- `notes/cutlass_architecture.md`（Week 1）
- `notes/fold_design.md`（Week 2）

---

## 下一步

1. **整理文件夹**（本文档完成后立即执行）
2. **Week 1 Day 1**：下载 CUTLASS 3.5.1，开始研究源码
