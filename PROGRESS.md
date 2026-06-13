# Kan v2.1 开发进度

**最后更新**：2026-06-13
**Git HEAD**：f1974d8

---

## ✅ 2026-06-13 修复（commit 5c29095）

### 1. Phase 8 persistent scheduler 重做（旧实现从未能编译）

旧 `src/tc_cutlass_persistent.cu` 四个致命问题（已归档 `archive/dead-kernels/*.broken`）：
- `#undef` 用在函数名上（无效）→ 同一 TU 两个 `extern "C" tc_jackpot_search` 重定义
- 它的 `tc_jackpot_search` 参数顺序/数量与调用方 extern 声明不符（ABI 错位）
- 用 `g_search_stream`/`g_gather_evt` 却不 `ensure_search_stream()`
- 只覆盖了同步入口；生产路径走 `tc_search_launch`（async）→ PERSISTENT=1 实际无效

**新实现**：合并进 `tc_cutlass_v2.cu` —— kernel 改为 grid-stride 遍历 tile id：
- 默认（grid = nbm×nbn）：循环恰好走一次，与旧 launch 完全等价
- `TC_PERSIST=1`（**运行时环境变量，无需重编译**）：grid = nSM×occupancy
  （occupancy API 算），同 pid 顺序 → GROUPM L2 局部性不变
- A/B 测试：`bench/test_persistent.sh`（单次构建）或 CI `gpu_persist_ab.sh`

### 2. redux.sync 恢复（commit 1beb62d 的误回滚）

`__reduce_xor_sync` 是 CUDA 11.0+（不是 12.8+）；当时编译失败是 portable
fatbin 的 **sm_75** pass（redux 需要 cc80+）。现按 `__CUDA_ARCH__>=800` 守卫：
sm_80+ 单指令 warp XOR，sm_75 才走 5 条 shfl 蝶形。fold 热路径每次调用省
32×4 条指令。

### 3. CI 强化

- main 流水线现在 clone CUTLASS → 生产 kernel（tc_cutlass_v2）真正被编译检查
  （之前静默 fallback 到 tc_block，CUTLASS 代码从未在 CI 编译过）
- gpu 流水线（push `gpu` 分支）：build → cpu-verify → `gpu_persist_ab.sh`
  （real-config 中等 nbits 一抽即中 → POSTCHECK ok=1 门 + standard vs
  TC_PERSIST kernel-ms A/B）

---

## 📦 仓库状态

**远程**：https://cnb.cool/wuyueyi/peral
- `f1974d8` ci(gpu): add libssl-dev
- `3d47448` ci(gpu): persist A/B validation script + CUTLASS clone
- `5c29095` fix(kernel): working persistent scheduler + arch-guarded redux

---

## 🎯 下一步（提速方向，按 ROI 排序）

0. **✅ CI 结果已回（cnb-e23-1jqv4a7hg, L40 sm_89, 2026-06-13）**：
   - real-config GPU 中奖 → **POSTCHECK ok=1**（grid-stride 改动正确性确认）
   - kernel A/B：standard **387.4 ms** vs TC_PERSIST=1 **375.5 ms = +3.2%**
   - L40 含 redux 修复跑出 183-187 TH/s
1. **部署**：生产箱启动命令前加 `TC_PERSIST=1`（运行时开关，零重编译风险）；
   5090/4090 上重测 A/B（L2 更大，收益可能不同）
2. **占用率仍是根本瓶颈**（5090: 244 regs + 91KB smem → 1 TB/SM = 17% occ，
   350/1676 峰值 = 21%）。SMALL_TILE 已证伪。真正路径：
   - **kStages=4/5**（smem 还有富余时加深 cp.async 流水线，掩盖 DRAM 延迟）
   - **CuTe WGMMA + TMA 重写**（Blackwell/Hopper 原生，lpminer 同款架构）→
     500+ TH/s 的唯一已知路径，工作量 ~1-2 周
3. **5090/4090 box 已失联**（117.50.214.122 / 106.75.245.163 / 117.50.47.40
   全部超时；117.50.195.208 端口开但密码已变）→ 需要用户续租或换箱
