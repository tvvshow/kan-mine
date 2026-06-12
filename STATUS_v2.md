# Kan 性能优化 v2 (2026-06-12)

## 当前状态

**4090 基线**：
- Kernel：260 TH/s (270ms/draw)
- Wall-clock：190-220 TH/s
- Roofline：350 TH/s (74% 利用率)
- 瓶颈：**Occupancy 16.7%** (1 TB/SM)

**3090 基线**：
- Kernel：108 TH/s (650ms/draw)  
- Wall-clock：106.5 TH/s

---

## 已实施优化（v1.0.0 → v2.0.0）

### ✅ 已完成（Week 1）

1. **GPU-resident pipeline** (Day 5)
   - RNG + blake3 + noise 全部移到 GPU
   - CPU 准备时间：1490ms → 10ms
   - 3090 wall-clock：57 → 76.6 TH/s

2. **CUTLASS FoldMmaMultistage** (Day 4-5)
   - 融合 3-stage cp.async pipeline
   - 消除 16× 跨 chunk 的 barrier drain
   - 3090：56 → 108 TH/s kernel

3. **Lane-distributed fold** (Day 5)
   - 32-lane 并行 RMW 替代 lane-0 串行
   - 消除 "execution pipe oversubscribed" 瓶颈
   - 无明显速度提升（已经不是瓶颈）

4. **Grouped raster GROUPM=8**
   - 列优先栅格化，band = 8 行条带
   - L2 命中率：93.9%（3090 6MB L2）
   - 3090：避免了 ~525GB DRAM 流量

5. **异步 search/prep overlap**
   - search(N) 与 prep(N+1) 并行
   - 理论 2× CPU 准备，实际贡献有限（GPU prep 只需 10ms）

---

## 待测试优化（v2.0.0 候选）

### 🔄 路径 1：GROUPM 调优（ROI 高，成本低）

**原理**：4090 L2 = 72MB（12× 3090），更大的 GROUPM 可能提升 B panel 复用

**实施**：
```bash
GROUPM=16 ./build.sh && ./build/plainproof_gen --cfg real --mine --batch 3
```

**预期**：
- 如果有效：260 → 280 TH/s (+7%)
- 如果无效：说明 GROUPM=8 对 4090 仍是最优

**代码状态**：✅ 已就绪
- `build.sh` 支持 `GROUPM=` 环境变量
- 默认值保持 8（向后兼容）

---

### 🔄 路径 2：SMALL_TILE（ROI 最高，成本中）

**原理**：TB 128×256 → 64×128，寄存器 240 → ~130，smem 100KB → 50KB → 2 TB/SM

**实施**：
```bash
SMALL_TILE=1 ./build.sh && ./build/plainproof_gen --cfg real --mine --batch 3
```

**预期**：
- 保守：260 → 340 TH/s (+30%)
- 乐观：260 → 390 TH/s (+50%)
- 风险：L2 命中率可能下降（小 tile = 更多 block = 更分散的访问）

**代码状态**：✅ 已就绪
- `src/tc_cutlass_v2.cu` 已添加 `#ifdef SMALL_TILE` 分支
- WarpShape 同步调整：64×64 → 32×64（必须整除 TBShape）

---

### 🔄 路径 3：组合优化

**实施**：
```bash
SMALL_TILE=1 GROUPM=16 ./build.sh
```

**预期**：260 → 350-400 TH/s（达到 90% roofline）

---

## 未实施优化（Phase 8）

### ⏸️ Persistent Scheduler（ROI 中，成本高）

**原理**：lpminer 使用 `StaticPersistentTileScheduler`
- gridDim = N_SM，每个 TB 循环 `tile_idx += gridDim`
- 消除 launch overhead + 尾部浪费

**预期**：+5-15%

**成本**：需要重写 CUTLASS host wrapper

**优先级**：仅当路径 1-3 达到 300+ TH/s 但仍不够时考虑

---

## 测试指南

### 方式 A：快速测试（推荐）

```bash
# 1. 上传 optimization_kit.tar.gz 到 4090 box
scp optimization_kit.tar.gz ubuntu@117.50.47.40:/home/ubuntu/

# 2. SSH 并测试
ssh ubuntu@117.50.47.40
cd kan
tar xzf ../optimization_kit.tar.gz
chmod +x bench/*.sh
bash bench/quick_test.sh
```

### 方式 B：手动单项测试

```bash
# GROUPM=16
cd kan
GROUPM=16 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"

# SMALL_TILE
SMALL_TILE=1 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"
```

### 判断标准

| 配置 | ms/draw | TH/s | 结论 |
|------|---------|------|------|
| Baseline (8, 128×256) | 270 | 260 | 当前生产版本 |
| GROUPM=16 | <260 | >270 | ✅ 部署 |
| GROUPM=16 | ≥270 | ≤260 | ❌ 保持 GROUPM=8 |
| SMALL_TILE | ~200 | ~350 | ✅✅ 立即部署！ |
| SMALL_TILE | >270 | <260 | ❌ 保持大 tile |
| 组合 | ~180 | ~400 | 🎯 最优 |

---

## 部署流程

找到最优配置后：

```bash
# 1. 重新编译（例如 SMALL_TILE=1 GROUPM=16）
cd kan
SMALL_TILE=1 GROUPM=16 ./build.sh

# 2. 验证 POSTCHECK
./build/plainproof_gen --cfg real --mine --batch 1 2>&1 | grep POSTCHECK
# 必须输出：POSTCHECK ok=1

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

## Roofline 目标

| GPU | 理论峰值 | 当前 v1.0.0 | 目标 v2.0.0 | 利用率 |
|-----|---------|------------|------------|--------|
| 4090 | 350 TH/s | 260 (74%) | 315+ (90%) | ✅ 可达 |
| 3090 | ~140 TH/s | 108 (77%) | 126+ (90%) | 🤔 测试 |

---

## 文件清单

### 核心代码
- `src/tc_cutlass_v2.cu` — 内核（已添加 SMALL_TILE 支持）
- `src/gpu_prep.cu` — GPU-resident pipeline
- `src/miner_main.cpp` — 矿工主程序

### 构建系统
- `build.sh` — 支持 `GROUPM=` 和 `SMALL_TILE=` 环境变量

### 测试工具
- `bench/quick_test.sh` — 快速测试 3 个配置
- `bench/sweep_4090.sh` — 完整扫描所有配置

### 文档
- `OPTIMIZATION_PLAN.md` — 详细优化方案
- `DEPLOY_4090.md` — 部署指南
- `README.md` — 用户文档
- `TUNING_PHASE7.md` — 原始分析（已搁置）

### 部署包
- `optimization_kit.tar.gz` — 包含所有上述文件

---

## 下一步

1. **立即**：测试 GROUPM=16（30秒，零风险）
2. **高优先级**：测试 SMALL_TILE=1（2分钟，高回报）
3. **如果有效**：部署到生产，更新 README
4. **如果无效**：分析 ncu profile，考虑 persistent scheduler

---

## 预期时间线

- **测试**：5-10 分钟
- **部署**：2 分钟
- **验证**：15 秒（等第一个统计表）
- **总计**：<20 分钟从测试到生产

---

**目标**：4090 从 190 TH/s → 270+ TH/s wall-clock (+42%)
