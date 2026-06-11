# Week 1 Day 1 总结与决策

## 今日完成
1. ✅ 下载 CUTLASS 3.5.1
2. ✅ 定位关键文件：`sm80_mma_multistage.hpp`
3. ✅ 分析 mainloop 架构（见 day1_sm80_architecture.md + day1_mainloop_analysis.md）

## 核心发现

### CUTLASS 的优势（为什么它能到 130 TH/s）
1. **双层 Pipeline**：
   - Smem：4 个 buffer 轮转（cp.async）
   - Reg：K_BLOCK_MAX 个 k-block 轮转（ldmatrix）
2. **Barrier 优化**：只在 k_block 边界同步，不是每个 mma
3. **软件流水**：当前算 k_block N，同时 prefetch N+1

### 我们 tc_imma2 的差距
- STAGES=2（vs CUTLASS 4）→ pipeline 太浅
- K_BLOCK_MAX 隐含在 RSUB（64/32=2 vs 可能更大）
- 每 RSUB 有 barrier（128 次/K=4096）

## 决策：路径 B（抄逻辑，手写实现）

### 为什么不用 CUTLASS 库？
1. **cute::Tensor 抽象太厚**：
   - `accum` 是 cute::Tensor，不是 `int32_t acc[][]`
   - Fold 需要清晰访问每个累加器寄存器
   - 学习 cute API 成本 > 手写收益
2. **编译复杂**：
   - 需要链接 libcutlass
   - 模板实例化慢
3. **调试困难**：
   - 模板错误信息难懂
   - 无法单步调试 cute 内部

### 路径 B 具体方案
**基于 tc_imma2，改进 pipeline**：
1. 增加 STAGES（2→4）
2. 增大 RSUB（64→128，需要 dynamic smem）
3. 实现**寄存器级流水**（K_BLOCK_MAX=2，即 RSUB/32=2）
4. 抄 CUTLASS 的 pipeline 轮转逻辑

### 预期收益
- tc_imma2 RSUB=64 已经 56 TH/s
- STAGES=4 + 寄存器流水 → 减少 stall → **目标 80-100 TH/s**
- 仍然清晰的 `acc[tr][tc][4]` 布局 → fold 无需改

## 修订后的 Week 1 计划

### Day 2-3: 实现深度 pipeline（目标 80 TH/s）
- [ ] 创建 `src/tc_deep_pipeline.cu`（基于 tc_imma2）
- [ ] 增加 STAGES=4
- [ ] 实现 K_BLOCK_MAX=2 的寄存器流水：
  ```cpp
  for (int ks = 0; ks < RSUB/32; ks++) {
    if (ks < RSUB/32 - 1) {
      // Prefetch ks+1 的 A/B 到 af_next/bf_next
      ldmatrix(..., af_next, ...);
    }
    // MMA 当前 ks
    mma(..., af[ks], bf[ks], acc);
    // Swap buffers
    swap(af, af_next);
  }
  ```
- [ ] 测试：POSTCHECK ok=1 + 速度

### Day 4-5: 优化 smem + 测试
- [ ] 尝试 RSUB=128（需要 dynamic smem 96KB）
- [ ] 如果超限，保持 RSUB=64 但优化其他
- [ ] 与 CUTLASS benchmark 对比速度

### Day 6-7: 集成到 plainproof_gen
- [ ] 保留 gather/blake3/host wrapper
- [ ] 验证：POSTCHECK ok=1
- [ ] 性能测试：`--mine 10 --cfg real`

## Week 1 修订目标
- **最低**：70 TH/s（比 tc_imma2 的 56 快 25%）
- **理想**：80-90 TH/s（接近 CUTLASS 纯 GEMM 的 70%）
- **如果达不到**：Day 6 重新评估，考虑更激进方案（persistent scheduler）

## 明天开始
Day 2：创建 `tc_deep_pipeline.cu`，实现 STAGES=4 + 寄存器流水。
