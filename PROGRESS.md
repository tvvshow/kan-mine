# Kan v2.0+ 开发进度

**最后更新**：2026-06-12  
**Git HEAD**：868039c

---

## ✅ 已完成

### v2.0.0 优化路径（commit 87fe1e3 + 2780d45）

**目标**：4090 从 260 TH/s → 315+ TH/s (90% roofline)

1. **GROUPM 调优**
   - 支持 `GROUPM=16/32/64` 编译参数
   - 预期：+5-10%

2. **SMALL_TILE**
   - TB 尺寸减半（128×256 → 64×128）
   - Occupancy 翻倍（1 TB/SM → 2 TB/SM）
   - 预期：+30-50%

3. **组合优化**
   - `SMALL_TILE=1 GROUPM=16`
   - 预期：+35-54%（达到 90% roofline）

### Phase 8: Persistent Scheduler（commit 2780d45）

**实现完成**：
- `src/tc_cutlass_persistent.cu`（128 行）
- `build.sh` 支持 `PERSISTENT=1`
- `bench/test_persistent.sh` 自动对比脚本

**预期**：在 v2.0.0 基础上额外 +5-15%

**触发条件**：v2.0.0 达到 300+ TH/s 但仍 <340 TH/s

---

## 🔄 进行中

### RTX 5090 测试部署（2026-06-12）

**机器信息**：
- GPU：RTX 5090 32GB（sm_120，5th gen tensor cores）
- 连接：ssh -p 23 root@117.50.214.122
- 状态：后台安装 CUDA toolkit + 依赖（预计 5-10 分钟）

**预期 roofline**：~450-500 TH/s（vs 4090 的 350 TH/s）

**测试脚本**：
- `bench/bench_5090.sh`（commit 868039c）
- 完整测试：baseline + v2.0.0 所有路径 + Phase 8

**部署任务 ID**：
- bttyc4zll（第一个部署脚本，正在运行）
- btps2ipj9（完整安装脚本，失败 exit 1）

---

## ⏳ 待执行

### 4090 box 测试（阻塞中）

**机器**：117.50.47.40（SSH 连接超时）

**需要**：
```bash
ssh ubuntu@117.50.47.40
cd kan && git pull
bash bench/quick_test.sh
```

**判断标准**：
| ms/draw | TH/s | 行动 |
|---------|------|------|
| <200 | >350 | ✅ 立即部署 |
| 210-240 | 290-330 | 🤔 测试 Phase 8 |
| >260 | <270 | ❌ 回退分析 |

---

## 📦 仓库状态

**远程**：https://cnb.cool/wuyueyi/peral

**最新 3 个 commit**：
1. `868039c` - feat: add RTX 5090 comprehensive benchmark script
2. `2780d45` - feat: Phase 8 persistent scheduler implementation
3. `87fe1e3` - perf: add GROUPM/SMALL_TILE optimization paths for v2.0.0

**文件清单**：
- `src/tc_cutlass_v2.cu` - v2.0.0 内核（GROUPM + SMALL_TILE）
- `src/tc_cutlass_persistent.cu` - Phase 8 内核
- `build.sh` - 统一构建（支持 3 个环境变量）
- `bench/quick_test.sh` - 4090 快速测试（3 个配置）
- `bench/test_persistent.sh` - Phase 8 对比测试
- `bench/bench_5090.sh` - 5090 完整测试
- `ROADMAP.md` - 完整优化路线图
- `DESIGN_persistent.md` - Phase 8 技术文档
- `STATUS_v2.md` - 状态跟踪
- `SUMMARY_v2.md` - 工作总结

---

## 🎯 下一步

### 立即（等待后台任务完成）

1. 等待 5090 box 安装完成（任务 ID: bttyc4zll）
2. 运行 `bench/bench_5090.sh` 获取完整基准数据
3. 对比 5090 vs 4090 性能（预期 5090 = 1.3-1.4× 4090）

### 短期（网络恢复后）

1. 4090 box 测试 v2.0.0
2. 根据结果决定是否测试 Phase 8

### 预期结果

**5090 目标**：
- Baseline：~360 TH/s（4090 的 260 × 1.35 倍）
- v2.0.0 SMALL_TILE：~470 TH/s（90% roofline）
- Phase 8：~500 TH/s（接近理论峰值）

**4090 目标**：
- v2.0.0：315+ TH/s（90% roofline）
- Phase 8：330-350 TH/s（接近理论峰值）

---

## 🚧 当前阻塞

1. **4090 box 网络不稳定**（117.50.47.40 SSH 超时）
2. **5090 box 安装中**（CUDA toolkit，预计还需 3-5 分钟）

**解决方案**：
- 4090：等待用户手动测试
- 5090：等待后台任务完成通知

---

**状态**：代码就绪，等待硬件测试数据
