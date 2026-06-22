# Production GPU Test Plan

日期：2026-06-22  
范围：`peral/` production portable packages、GPU profile、release 验收  
性质：从无 GPU 静态检查到 VPS 真机 benchmark / pool accepted 的分层测试清单

---

## 1. 测试分层

生产 GPU 支持计划分四层验证：

```text
L0: static release-profile checks
L1: portable package build checks
L2: GPU smoke / correctness checks
L3: live pool / benchmark acceptance checks
```

当前本地 Windows 环境已能执行 L0；由于本地没有 `nvcc` / NVIDIA GPU，L1-L3
需要 Linux CUDA 构建环境或 GPU VPS。

---

## 2. L0：无 GPU 静态检查

命令：

```bash
cd peral
bash check_release_profiles.sh
```

检查内容：

```text
1. package_portable.sh / install_kan.sh / check_release_profiles.sh bash 语法；
2. .cnb.yml 默认 release 资产只包含：
   - dist/kan-portable-linux-x64.tar.gz
   - dist/kan-portable-linux-x64-sm86-g8.tar.gz
   - install_kan.sh
3. sm86 历史 sweep 包不作为默认 release 资产；
4. GPU_PROFILES.md 记录 sm86-g8:
   ARCH=sm_86
   GROUPM=8
   KSTAGES=3
   TC_PERSIST=0
5. install_kan.sh 选包矩阵：
   sm_86 -> sm86-g8
   sm_75/sm_80/sm_89/sm_90/sm_120/unknown -> generic
6. package_portable.sh 包含 BUILD_INFO / GPU_PROFILES / install_kan / TC_PERSIST / GPU compute_cap 打印钩子。
```

通过条件：

```text
OK: release profile checks passed
```

---

## 3. L1：portable package build checks

环境要求：

```text
Linux x86-64
CUDA toolkit with nvcc
CUTLASS v3.5.1
patchelf
gcc/g++
git/curl/ca-certificates/libssl-dev
```

命令：

```bash
cd peral

# generic compatibility package
WITH_AB=0 bash package_portable.sh

# sm86 production tuned package
WITH_AB=0 ARCH=sm_86 GROUPM=8 KSTAGES=3 PACKAGE_FLAVOR=sm86-g8 bash package_portable.sh
```

预期产物：

```text
dist/kan-portable-linux-x64.tar.gz
dist/kan-portable-linux-x64-sm86-g8.tar.gz
dist/kan-portable-linux-x64-<version>.tar.gz
dist/kan-portable-linux-x64-<version>-sm86-g8.tar.gz
```

检查包内容：

```bash
tar tzf dist/kan-portable-linux-x64.tar.gz | grep -E 'kan$|run.sh|status.sh|BUILD_INFO.txt|GPU_PROFILES.md|install_kan.sh'
tar tzf dist/kan-portable-linux-x64-sm86-g8.tar.gz | grep -E 'kan$|run.sh|status.sh|BUILD_INFO.txt|GPU_PROFILES.md|install_kan.sh'
```

检查 `BUILD_INFO.txt`：

```bash
tmp="$(mktemp -d)"
tar xzf dist/kan-portable-linux-x64-sm86-g8.tar.gz -C "$tmp"
cat "$tmp/kan-portable-linux-x64/BUILD_INFO.txt"
```

sm86-g8 期望字段：

```text
arch: sm_86
groupm: 8
kstages: 3
package_flavor: sm86-g8
package_policy: tuned
portable: 1
```

generic 期望字段：

```text
arch: portable-fatbin
package_flavor: generic
package_policy: generic-compatible
portable: 1
```

---

## 4. L2：GPU smoke / correctness checks

环境：

```text
NVIDIA GPU VPS
NVIDIA driver
不需要 CUDA toolkit / CUTLASS / compiler
```

安装：

```bash
# 自动检测 GPU 并选择 package
VERSION=<release-tag> bash install_kan.sh

cd ~/kan
./status.sh
```

sm86 期望：

```text
package: kan-portable-linux-x64-sm86-g8.tar.gz
BUILD_INFO:
  arch: sm_86
  groupm: 8
  kstages: 3
  package_flavor: sm86-g8
run.sh:
  TC_PERSIST=0
```

generic fallback 期望：

```text
package: kan-portable-linux-x64.tar.gz
BUILD_INFO:
  arch: portable-fatbin
  package_flavor: generic
```

GPU smoke：

```bash
cd ~/kan
./plainproof_gen --cfg real --mine 1 --target ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff 777 >/tmp/kan_smoke.b64 2>/tmp/kan_smoke.log
grep -E 'POSTCHECK|VALID|MINE done|FUSED|TH/s' /tmp/kan_smoke.log
```

通过条件：

```text
POSTCHECK ok=1
无 CUDA launch error
无 smem attr invalid argument
```

特别检查：

```text
sm86-g8-k4 不应出现在默认 release；
KSTAGES=4 smoke fail 不能作为生产候选。
```

---

## 5. L3：live pool / benchmark acceptance checks

### 5.1 Controlled benchmark

命令：

```bash
cd ~/kan
TC_TIMING=1 ./plainproof_gen --cfg real --mine 15 --tc --breakdown 777 >/tmp/kan_bench.out 2>/tmp/kan_bench.log
grep -E 'prep|search|total|MINE done|TH/s|POSTCHECK' /tmp/kan_bench.log /tmp/kan_bench.out
```

sm86-g8 / RTX 3080 Ti 参考：

```text
search_avg ≈ 99.58 TH/s
total_avg  ≈ 99.03 TH/s
MINE done  ≈ 97.74 TH/s
```

允许短样本波动，但如果显著低于历史值，需要记录：

```text
driver
power limit
temperature
SM clock
mem clock
TC_PERSIST
package BUILD_INFO
```

### 5.2 Pool live production run

原则：

```text
在没有新版本 / 新 package 产出前，VPS 应持续运行上一个已验证生产包；
live pool 不是“限时测试后停止”，而是持续生产运行，并定期采样状态。
```

建议用 tmux/nohup 承载进程；portable `run.sh` 在 pool 模式默认内置
auto-restart/reconnect loop（断连或 job-error 退出后自动重启）。因此目标机
无需再手写外层 restart loop，除非显式设置 `KAN_RESTART=0` 做一次性调试。

```bash
cd ~/kan
nohup ./run.sh --algo pearl \
  --pool stratum+ssl://prl.kryptex.network:8048 \
  --wallet <PRL_ADDRESS.WORKER_OR_ADDRESS> \
  --batch 500 --cfg real --tc >/tmp/fast.log 2>&1 &

# Optional:
#   KAN_RESTART=0       one-shot debug run; do not auto-restart
#   KAN_RESTART_DELAY=5 change reconnect delay
```

观察：

```bash
./status.sh
tail -f /tmp/fast.log
```

最低验收：

```text
1. 成功连接 pool；
2. hashrate table 正常；
3. accepted share 正常出现；
4. rejected 不异常；
5. 无 submit timeout 风暴；
6. 短期 15-30 分钟内稳定，之后持续运行直到新版本替换或用户要求停止；
7. sm86 live pool 接近 93-96 TH/s 历史区间，或解释偏差。
```

持续运行状态采样：

```text
短期 5 分钟：确认 accepted/rejected/submit timeout；
中期 30-60 分钟：确认 live 60s / 15m hashrate；
长期：在新版本发布前保持运行；只在更新包、异常排查或用户要求时重启。
```

---

## 6. 需要从 VPS 收集的信息

如果需要我远程执行 L2/L3，请提供：

```text
1. SSH host / port；
2. SSH user；
3. 认证方式：
   - 如使用密码，请只放在环境变量 BOXPW；
   - 不要把密码写入聊天或仓库文件；
4. GPU 型号；
5. 目标 release tag 或 tarball URL；
6. 是否允许跑 pool live test；
7. pool / wallet / worker：
   - 可以提供测试 worker；
   - 不要提供 wallet seed / 私钥；
8. 是否允许使用 tmux session fast；
9. 期望测试时长：
   - smoke: 1-3 分钟；
   - controlled benchmark: 5-10 分钟；
   - live pool: 15-30 分钟起。
```

默认远程执行原则：

```text
只下载 release 包；
不现场编译；
不修改生产 miner 源码；
日志写到 /tmp/fast.log 或 /tmp/kan_*.log；
测试完成后汇总 BUILD_INFO、GPU 状态、POSTCHECK、hashrate、accepted/rejected。
```

---

## 7. 当前需要 VPS 的测试点

当前本地已经完成：

```text
L0 static release-profile checks
YAML parse
install_kan dry-run selection
```

仍需要 VPS / CUDA 环境完成：

```text
L1 portable package build checks
L2 GPU smoke / POSTCHECK
L3 controlled benchmark / live pool accepted
```
