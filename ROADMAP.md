# Kan v2.0+ 完整优化路线图

**日期**：2026-06-12  
**目标**：4090 从 260 TH/s → 315+ TH/s (90% roofline)  
**状态**：v2.0.0 代码就绪，Phase 8 已实现，等待测试

---

## 性能基线（v1.0.0）

| GPU | Kernel TH/s | Wall TH/s | Roofline | 利用率 |
|-----|------------|-----------|----------|--------|
| 4090 | 260 | 190-220 | 350 | 74% |
| 3090 | 108 | 102-106 | ~140 | 77% |

**瓶颈**：Occupancy 16.7% (1 TB/SM)  
- 240 regs + ~100KB smem → 每个 SM 只能放 1 个 threadblock
- 4090 有 128 SM → 最多 128 个并发 TB，但 grid = 524,288 个 TB

---

## v2.0.0 优化路径（✅ 代码完成，等待测试）

### 路径 1：GROUPM 调优

**目标**：利用 4090 的 72MB L2（vs 3090 的 6MB）

**实施**：
```bash
GROUPM=16 ./build.sh
```

**原理**：
- 当前 GROUPM=8 对 3090 6MB L2 最优
- 更大的 GROUPM → 更多 block 共享 B panel → 更少 DRAM 流量
- 4090 L2 = 72MB 可能支持更大的 band

**预期收益**：+5-10%（260 → 270-280 TH/s）

**风险**：低（只是重新映射 blockIdx）

---

### 路径 2：SMALL_TILE

**目标**：降低寄存器压力，提升 occupancy 到 2 TB/SM

**实施**：
```bash
SMALL_TILE=1 ./build.sh
```

**原理**：
- TB 尺寸减半：128×256 → 64×128
- 寄存器减半：240 → ~130
- Smem 减半：100KB → 50KB
- Occupancy：1 TB/SM → 2 TB/SM

**预期收益**：+30-50%（260 → 340-390 TH/s）

**风险**：中
- 小 tile → 更多 block → 更分散的内存访问 → 可能降低 L2 命中率
- 但 2× occupancy 的收益预期大于 L2 损失

---

### 路径 3：组合优化

**实施**：
```bash
SMALL_TILE=1 GROUPM=16 ./build.sh
```

**预期收益**：+35-54%（260 → 350-400 TH/s）

**目标**：达到 90% roofline

---

## Phase 8：Persistent Scheduler（✅ 代码完成，条件触发）

### 何时测试 Phase 8？

**前提条件**（任一满足即可测试）：
1. v2.0.0 达到 300+ TH/s 但距 350 TH/s 仍有 >10% 差距
2. v2.0.0 的 SMALL_TILE 收益低于预期（<20%）

**何时跳过 Phase 8？**
1. v2.0.0 已达到 340+ TH/s（97% roofline）→ 边际收益 <5%
2. v2.0.0 测试失败（无提升或下降）→ 需要先解决根本问题

### Persistent Scheduler 原理

**当前调度**（Standard Grid-Stride）：
- Grid = (512, 1024) = 524,288 个 threadblock
- 每个 TB 处理 1 个 tile
- 4090: 128 SM × 1 TB/SM = 128 个并发 TB
- 执行 524,288 / 128 = 4096 wave

**Persistent 调度**：
- Grid = 128 或 256（= num_SM × occupancy）
- 每个 TB 循环处理多个 tile：`for (tile_idx = blockIdx.x; tile_idx < 524288; tile_idx += 128)`
- 每个 TB 处理 ~4096 个 tile

**收益来源**：
1. **消除 launch overhead**：1 次启动 vs 524k 个 tile 调度
2. **消除尾部不平衡**：动态分配，先完成的 TB 立即获取下一个 tile
3. **更好的 L2 复用**：同一 TB 处理相邻 tile，L2 数据留存时间更长

**预期收益**：+5-15%

### 实施方式

```bash
# 测试（在 v2.0.0 最优配置基础上）
cd kan
git pull
BASELINE="SMALL_TILE=1 GROUPM=16" bash bench/test_persistent.sh
```

测试脚本会：
1. 编译标准版本，运行 3 draw
2. 编译 persistent 版本，运行 3 draw
3. 对比性能，计算 speedup
4. 给出部署建议（>5% → 部署，<5% → 保持标准）

### 代码文件

- `src/tc_cutlass_persistent.cu` — 完整实现（128 行）
- `build.sh` 支持 `PERSISTENT=1` 环境变量
- `bench/test_persistent.sh` — 自动对比脚本
- `DESIGN_persistent.md` — 详细技术文档

---

## 测试决策树

```
开始：测试 v2.0.0
│
├─ quick_test.sh 结果
│  │
│  ├─ GROUPM=16: <260ms? → 部署 GROUPM=16
│  │
│  ├─ SMALL_TILE=1: ~200ms (350 TH/s)?
│  │  ├─ YES → ✅ 接近 roofline，部署 v2.0.0
│  │  │        可选：测试 Phase 8（预期额外 +5-10%）
│  │  │
│  │  └─ NO: ~240ms (290 TH/s)?
│  │     └─ 仍有 20% 差距 → 必须测试 Phase 8
│  │
│  └─ SMALL_TILE=1 无提升或下降?
│     └─ 回退，分析 ncu profile，重新评估
│
└─ 如果测试 Phase 8：
   ├─ test_persistent.sh 显示 >5% 提升?
   │  └─ YES → 部署 PERSISTENT=1 SMALL_TILE=1 GROUPM=16
   │
   └─ NO (<5% 或负提升)
      └─ 保持 v2.0.0，roofline 已达到或其他瓶颈
```

---

## 快速参考

### 测试命令

```bash
# 方式 A：自动测试所有配置（推荐）
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

# Phase 8 对比测试
BASELINE="SMALL_TILE=1 GROUPM=16" bash bench/test_persistent.sh
```

### 判断标准

| ms/draw | TH/s | roofline % | 行动 |
|---------|------|-----------|------|
| <200 | >350 | >100% | 🎯 超出预期！立即部署 |
| ~200 | ~350 | ~100% | ✅ 达到 roofline，部署 |
| 210-240 | 290-330 | 83-94% | 🤔 接近目标，可选测试 Phase 8 |
| 240-260 | 270-290 | 77-83% | 📊 小幅改善，必须测试 Phase 8 |
| >260 | <270 | <77% | ❌ 无改善，回退并分析 |

### 部署命令（找到最优配置后）

```bash
# 1. 重新编译最优配置（例如 SMALL_TILE=1 GROUPM=16）
cd kan
SMALL_TILE=1 GROUPM=16 ./build.sh

# 2. 验证 POSTCHECK
./build/plainproof_gen --cfg real --mine --batch 1 2>&1 | grep POSTCHECK
# 必须：POSTCHECK ok=1

# 3. 停止旧矿工
pkill -9 kan

# 4. 启动新版本
nohup ./build/kan --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet prl1patz2m...apmv.pm \
  > kan_v2.log 2>&1 &

# 5. 监控
tail -f kan_v2.log
# 等待第一个统计表（15秒后）
# 确认 HASHRATE 列显示 >280 TH/s
```

---

## 文件清单

### 核心代码（v2.0.0）
- `src/tc_cutlass_v2.cu` — 标准版本，支持 GROUPM 和 SMALL_TILE
- `src/tc_cutlass_persistent.cu` — Phase 8 persistent scheduler
- `src/gpu_prep.cu` — GPU-resident pipeline（v1.0.0 已完成）
- `build.sh` — 统一构建脚本

### 测试工具
- `bench/quick_test.sh` — 快速测试 3 个 v2.0.0 配置
- `bench/test_persistent.sh` — 对比标准 vs persistent
- `bench/sweep_4090.sh` — 完整扫描（可选）

### 文档
- `OPTIMIZATION_PLAN.md` — v2.0.0 详细方案
- `DESIGN_persistent.md` — Phase 8 技术文档
- `DEPLOY_4090.md` — 部署指南
- `STATUS_v2.md` — 完整状态跟踪
- `SUMMARY_v2.md` — 工作总结
- `ROADMAP.md` — 本文档

---

## 预期时间线

| 阶段 | 时间 | 操作 |
|------|------|------|
| v2.0.0 测试 | 5-10 分钟 | `bash bench/quick_test.sh` |
| 分析结果 | 1 分钟 | 对比 ms/draw |
| Phase 8 测试（可选） | 5-10 分钟 | `bash bench/test_persistent.sh` |
| 部署 | 2 分钟 | 重新编译 + 重启矿工 |
| 验证 | 15 秒 | 等第一个统计表 |
| **总计** | **<20 分钟** | 从测试到生产 |

---

## 成功指标

**v2.0.0 成功**：
- 4090 wall-clock ≥ 270 TH/s (+42% over 190 baseline)
- 对应 kernel ≥ 360 TH/s（假设 75% wall/kernel 比例）
- 测试验证：ms/draw ≤ 200

**Phase 8 成功**（如果测试）：
- 额外提升 ≥5% over v2.0.0
- 例如：v2.0.0 = 200ms → Phase 8 = 190ms（+5.3%）

**最终目标**：
- 4090 kernel ≥ 315 TH/s (90% roofline)
- 4090 wall ≥ 240 TH/s（考虑 CPU overhead）

---

## 当前阻塞

**4090 box 网络连接不稳定**（117.50.47.40）

**解决方案**：用户手动登录测试
```bash
ssh ubuntu@117.50.47.40
cd kan && git pull && bash bench/quick_test.sh
```

所有代码已推送到仓库（commit 87fe1e3 + Phase 8 新增），pull 后即可测试。

---

**下一步**：等待 4090 box 测试结果，根据决策树选择部署方案。
