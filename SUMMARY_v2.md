# Kan v2.0.0 优化工作总结

**日期**：2026-06-12  
**目标**：4090 从 260 TH/s → 315+ TH/s (90% roofline)  
**状态**：代码就绪，等待测试

---

## 已完成工作

### 1. 性能瓶颈分析 ✅

从 TUNING_PHASE7.md 和 4090 benchmark 确认：
- **当前**：260 TH/s kernel (270ms/draw), 190-220 TH/s wall
- **瓶颈**：Occupancy 16.7% (1 TB/SM)
  - 240 regs + ~100KB smem → 每 SM 只能放 1 个 threadblock
  - 4090 有 128 SM，但只有 128 个并发 TB
- **Roofline**：350 TH/s (cutlass_int8_bench 实测)
- **潜力**：260 = 74% roofline → 还有 26% 空间

### 2. 优化方案设计 ✅

**路径 1：GROUPM 调优**（低成本，5-10% 收益）
- 原理：4090 L2 = 72MB (vs 3090 的 6MB)，更大的 GROUPM 可能提升 B panel 复用
- 实施：测试 GROUPM ∈ {16, 32, 64}
- 预期：如果有效 → 260 → 280 TH/s

**路径 2：SMALL_TILE**（中等成本，30-50% 收益）
- 原理：TB 128×256 → 64×128，regs 减半，smem 减半 → 2 TB/SM
- 实施：编译时定义 SMALL_TILE
- 预期：260 → 340-390 TH/s
- 风险：小 tile 可能降低 L2 命中率

**路径 3：组合优化**
- SMALL_TILE + 最优 GROUPM
- 预期：260 → 350-400 TH/s (达到 90% roofline)

### 3. 代码实现 ✅

**src/tc_cutlass_v2.cu**：
```cpp
#ifdef SMALL_TILE
using TBShape   = cutlass::gemm::GemmShape<64, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
#else
using TBShape   = cutlass::gemm::GemmShape<128, 256, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
#endif
```

**build.sh**：
```bash
GROUPM="${GROUPM:-8}"
SMALL_TILE="${SMALL_TILE:-}"
EXTRA_FLAGS=""
[ -n "$SMALL_TILE" ] && EXTRA_FLAGS="-DSMALL_TILE"
nvcc -DGROUPM=${GROUPM} ${EXTRA_FLAGS} ...
```

### 4. 测试工具 ✅

**bench/quick_test.sh**：测试 3 个最有希望的配置
- GROUPM=16
- SMALL_TILE=1
- SMALL_TILE=1 GROUPM=16

**bench/sweep_4090.sh**：完整扫描所有 GROUPM 值

### 5. 文档 ✅

- **OPTIMIZATION_PLAN.md**：详细优化方案和 ROI 分析
- **DEPLOY_4090.md**：部署指南
- **STATUS_v2.md**：完整状态和时间线
- **README.md**：已有用户文档（v1.0.0）

### 6. 版本控制 ✅

- Commit: `87fe1e3` - "perf: add GROUPM/SMALL_TILE optimization paths for v2.0.0"
- 已推送到 cnb.cool/wuyueyi/peral

---

## 待执行操作

### 测试（在 4090 box 上）

```bash
# 方式 A：自动测试
ssh ubuntu@117.50.47.40
cd kan
git pull
bash bench/quick_test.sh

# 方式 B：手动单项测试
# GROUPM=16
GROUPM=16 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"

# SMALL_TILE
SMALL_TILE=1 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"

# 组合
SMALL_TILE=1 GROUPM=16 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"
```

### 判断标准

| ms/draw | 对应 TH/s | 行动 |
|---------|----------|------|
| <250 | >280 | ✅ 立即部署到生产 |
| 250-270 | 260-280 | 🤔 小幅改善，考虑部署 |
| >270 | <260 | ❌ 保持当前配置 |

### 部署到生产（如果测试有效）

```bash
# 假设 SMALL_TILE=1 GROUPM=16 最优
cd kan
SMALL_TILE=1 GROUPM=16 ./build.sh

# 验证
./build/plainproof_gen --cfg real --mine --batch 1 2>&1 | grep POSTCHECK
# 必须：POSTCHECK ok=1

# 重启矿工
pkill -9 kan
nohup ./build/kan --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet prl1patz2m...apmv.pm \
  > kan_v2.log 2>&1 &

# 监控
tail -f kan_v2.log
# 15秒后应看到 >280 TH/s
```

---

## 技术细节

### GROUPM 原理

**当前 GROUPM=8**（3090 优化）：
- 3090 L2 = 6MB
- band = 8 个连续行条带
- 列优先栅格化：block 0-7 处理同一 B panel
- L2 命中率 93.9%

**GROUPM=16 假设**（4090 优化）：
- 4090 L2 = 72MB (12× 3090)
- 更大的 band → 更多 block 共享 B panel
- 如果 B panel working set 仍能放入 L2 → 更少 DRAM 流量

### SMALL_TILE 原理

**当前 TB 128×256**：
- 每个 warp 处理 64×64 tile
- 2 个 warp 横向拼接（128 行）
- 4 个 warp 纵向拼接（256 列）
- 总 8 warp × 32 thread = 256 thread/block
- Regs：~240 → 限制 occupancy

**SMALL_TILE TB 64×128**：
- 每个 warp 处理 32×64 tile（减半）
- 2×2 warp = 4 warp × 32 thread = 128 thread/block
- Regs：~130 → 允许 2 TB/SM
- Smem：~50KB（vs 100KB）→ 不再是瓶颈

### 风险评估

**GROUPM 调优**：
- 风险：低（只是重新映射 blockIdx）
- 最坏情况：无改善，保持 GROUPM=8

**SMALL_TILE**：
- 风险：中（可能降低 L2 命中率）
- 原因：更多 block → 更分散的内存访问
- 预期：2× occupancy 的收益 > L2 命中率下降的损失
- 最坏情况：速度不增反降 → 回退

---

## 下一阶段（如果 v2.0.0 不够）

### Phase 8：Persistent Scheduler ✅ 已实现（2026-06-12）

**原理**：lpminer 风格
- gridDim = N_SM（4090 = 128，或 256 if SMALL_TILE）
- 每个 TB 循环处理多个 tile：`for (int t = blockIdx.x; t < total_tiles; t += gridDim.x)`
- 消除 kernel launch overhead
- 消除尾部不平衡（最后几个 tile 等待最慢 TB）
- 更好的 L2 复用（同一 TB 处理相邻 tile）

**预期**：+5-15%

**代码状态**：✅ 完成
- `src/tc_cutlass_persistent.cu` — 完整实现，128 行
- `build.sh` 支持 `PERSISTENT=1` 环境变量
- `bench/test_persistent.sh` — 自动对比标准 vs persistent

**测试方式**：
```bash
cd kan
git pull
BASELINE="SMALL_TILE=1 GROUPM=16" bash bench/test_persistent.sh
```

**部署决策**：
- 如果 v2.0.0 = 200ms/draw (~350 TH/s) → **接近 roofline，Phase 8 低优先级**
- 如果 v2.0.0 = 240ms/draw (~290 TH/s) → **仍有 20% 差距，测试 Phase 8**
- 如果 Phase 8 提升 >5% → 部署：`PERSISTENT=1 SMALL_TILE=1 GROUPM=16 ./build.sh`
- 如果 Phase 8 提升 <5% → 保持标准版本

---

## 时间估算

- **测试**：5 分钟（3 个配置 × 3 draw × ~30 秒）
- **分析**：1 分钟（对比 baseline）
- **部署**：2 分钟（重新编译 + 重启矿工）
- **验证**：15 秒（等第一个统计表）
- **总计**：<10 分钟从测试到生产运行

---

## 当前阻塞

- **4090 box 网络不稳定**：SSH 连接超时
- **解决方案**：用户手动登录测试
  ```bash
  ssh ubuntu@117.50.47.40
  cd kan && git pull && bash bench/quick_test.sh
  ```

---

## 文件位置

**本地**：
- `D:\mybitcoin\3\peral\optimization_kit.tar.gz` — 完整部署包

**仓库**：
- https://cnb.cool/wuyueyi/peral (commit 87fe1e3)

**4090 box**：
- `ssh ubuntu@117.50.47.40` (密码 `P8NG257Wv13OT9c6`)
- 路径：`/home/ubuntu/kan/`
- 需要 `git pull` 获取最新代码

---

## 成功指标

**v2.0.0 成功**：4090 wall-clock ≥ 270 TH/s (+42% over 190 baseline)

**对应 kernel**：≥ 360 TH/s (假设 75% wall/kernel 比例)

**测试验证**：`ms/draw ≤ 200` (vs 270 baseline)

---

**下一步**：等待用户在 4090 box 上运行 `bash bench/quick_test.sh`
