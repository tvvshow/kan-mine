# Phase 7 优化实施方案

> 2026-06-12，基于 4090 260 TH/s kernel / 190 TH/s wall 基线
> 目标：90% roofline ≈ 315+ TH/s kernel

## 现状分析

**瓶颈**：Occupancy 16.7% (1 TB/SM)
- 240 regs + ~100KB smem → 每个 SM 只能放一个 threadblock  
- 4090 有 128 SM → 最多 128 个并发 TB
- Compute 利用率 80.9%，L2 命中 93.9% → 都很好，只有 occupancy 是瓶颈

**Roofline**：4090 纯 int8 GEMM ≈ 350 TH/s (cutlass_int8_bench 测量)
- 当前 260 = 74% roofline
- 理论上限：90%+ roofline ≈ 315+ TH/s

---

## 优化路径（按 ROI 排序）

### 路径 1：GROUPM 调优（5-10% 提升，低成本）

**原理**：4090 L2 = 72MB（vs 3090 的 6MB），更大的 GROUPM 可能提升 B panel 复用

**实施**：
```bash
# 测试 GROUPM=16
GROUPM=16 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 5

# 测试 GROUPM=32
GROUPM=32 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 5

# 测试 GROUPM=64
GROUPM=64 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 5
```

**预期**：
- 如果 GROUPM=16 优于 8：260 → 280 TH/s (+7%)
- 如果无改善：说明 8 已是 4090 最优

**验证**：观察 `ms/draw` 是否降低（270ms baseline）

---

### 路径 2：SMALL_TILE（30-50% 提升，中等成本）

**原理**：TB 尺寸减半（128×256 → 64×128）→ 寄存器和 smem 减半 → 2 TB/SM → 2× occupancy

**实施**：
```bash
# 编译 SMALL_TILE 版本
SMALL_TILE=1 ./build.sh

# 测试
./build/plainproof_gen --cfg real --mine --batch 5
```

**代码已就绪**：
- `src/tc_cutlass_v2.cu` 已添加 `#ifdef SMALL_TILE` 支持
- `build.sh` 已支持 `SMALL_TILE=1` 环境变量

**预期**：
- 保守：260 → 340 TH/s (+30%)
- 乐观：260 → 390 TH/s (+50%)
- 风险：如果 L2 命中率下降，收益可能打折扣

**验证**：
1. `ms/draw` 应降至 180-200ms（vs 270ms baseline）
2. 用 `ncu` profile 确认 occupancy 从 16.7% → 33%+

---

### 路径 3：组合优化（理论最优）

**实施**：
```bash
# SMALL_TILE + 最优 GROUPM
SMALL_TILE=1 GROUPM=16 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 5
```

**预期**：260 → 350-400 TH/s (+35-54%)

---

## 快速测试脚本

已创建 `bench/sweep_4090.sh`：
```bash
cd kan
bash bench/sweep_4090.sh
```

自动测试所有配置并输出对比结果。

---

## 部署步骤

### 方法 A：手动部署（推荐，网络不稳定时）

```bash
# 1. 在本地打包
cd D:\mybitcoin\3\peral
tar czf /tmp/opt.tar.gz build.sh src/tc_cutlass_v2.cu bench/sweep_4090.sh

# 2. 上传到 4090 box
scp /tmp/opt.tar.gz ubuntu@117.50.47.40:/home/ubuntu/

# 3. SSH 到 box
ssh ubuntu@117.50.47.40
cd kan
tar xzf ../opt.tar.gz
bash bench/sweep_4090.sh
```

### 方法 B：自动化（Python）

```python
# deploy_opt.py
import paramiko, sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('117.50.47.40', username='ubuntu', password='P8NG257Wv13OT9c6')

sftp = client.open_sftp()
sftp.put('benchmark_update.tar.gz', '/home/ubuntu/opt.tar.gz')
sftp.close()

stdin, stdout, stderr = client.exec_command('cd kan && tar xzf ../opt.tar.gz && bash bench/sweep_4090.sh', timeout=600)

for line in stdout:
    print(line.rstrip())

client.close()
```

---

## 预期结果示例

```
=== 4090 Kernel Optimization Sweep ===
Baseline: 260 TH/s (GROUPM=8, TB 128×256)

--- Test 1: GROUPM sweep ---
Building GROUPM=16...
  Testing: 3 draws in 2617 ms = 872.3 ms/draw  ← 如果比 270ms 慢，说明 8 最优
Building GROUPM=32...
  Testing: 3 draws in 2415 ms = 805.0 ms/draw
Building GROUPM=64...
  Testing: 3 draws in 2301 ms = 767.0 ms/draw

--- Test 2: SMALL_TILE (2 TB/SM) ---
Building TB 64×128...
  Testing: 3 draws in 1620 ms = 540.0 ms/draw  ← 如果 ~200ms，成功！

--- Test 3: SMALL_TILE + GROUPM=16 ---
  Testing: 3 draws in 1500 ms = 500.0 ms/draw  ← 组合最优
```

**解读**：
- 270ms → 200ms = +35% → 260 → 350 TH/s ✅ 达到 90% roofline
- 270ms → 540ms = -50% → SMALL_TILE 在 4090 上不适用（L2 命中率下降）

---

## 后续路径（如果上述不够）

### 路径 4：Persistent Scheduler（5-15% 提升，高成本）

**原理**：lpminer 使用 `StaticPersistentTileScheduler`
- gridDim = N_SM，每个 TB 循环处理多个 tile
- 消除 kernel launch overhead + 尾部浪费

**成本**：需要重写 CUTLASS host wrapper，工作量大

**实施时机**：仅当路径 1-3 达到 300+ TH/s 但距离 350 还有差距时考虑

---

## 文件清单

- `build.sh` — 已更新，支持 `GROUPM=` 和 `SMALL_TILE=` 环境变量
- `src/tc_cutlass_v2.cu` — 已添加 `#ifdef SMALL_TILE` 分支
- `bench/sweep_4090.sh` — 自动化测试脚本
- `benchmark_update.tar.gz` — 包含上述三个文件的打包

---

## 立即行动

**最低成本测试**（30 秒）：
```bash
ssh ubuntu@117.50.47.40
cd kan
GROUPM=16 ./build.sh && ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep ms/draw
```

如果 ms/draw < 270，说明有效！

**最高 ROI 测试**（2 分钟）：
```bash
SMALL_TILE=1 ./build.sh && ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep ms/draw
```

如果 ms/draw ≈ 200，直接部署到生产！
