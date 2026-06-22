# Production GPU Support Plan

日期：2026-06-22  
范围：`peral/` 正式生产版本、portable release、NVIDIA 多架构 GPU 支持  
性质：生产级 GPU 支持策略 + tuned package 发布方案 + 整改路线

---

## 1. 正式生产目标

正式生产版本不能只围绕 RTX 3080 Ti 做单点优化。RTX 3080 Ti 只是
`sm_86 / Ampere gaming` profile 的一个代表样本。

项目正式生产目标应定义为：

```text
支持 NVIDIA Turing / RTX 20 系及以上 GPU，包括家用卡、商用卡和数据中心卡。
```

生产版本需要同时满足：

```text
1. generic portable 包保证 NVIDIA 20 系以上大多数 GPU 可运行；
2. tuned portable 包为已实测主力架构固化最优参数；
3. 目标机器不现场编译，只下载对应 release 包运行；
4. 每个 tuned profile 都有 benchmark / correctness / pool accepted 记录；
5. 未实测架构使用 generic 包或标记为待测，不能假装最优。
```

---

## 2. 核心原则

### 2.1 兼容性与最优性能分层

必须区分：

```text
能运行
```

和：

```text
以该架构已实测最优参数运行
```

因此 release 包分两类：

```text
generic package:
  最大兼容，能跑优先，不承诺最优。

tuned package:
  针对已实测 GPU 架构，固化编译期最优参数和运行期默认参数。
```

### 2.2 目标机器不编译

正式部署模型：

```text
CI / workflow 预编译 portable packages
  ↓
release 发布 tar.gz
  ↓
目标机器检测 GPU 架构
  ↓
下载对应 tuned package，如无 tuned package 则 fallback generic
  ↓
运行 ./run.sh
```

目标机器不应该：

```text
git pull + nvcc 编译
现场 sweep GROUPM/KSTAGES
现场修改源码
```

### 2.3 编译期参数必须通过 flavor 包固化

这些参数是编译期参数：

```text
ARCH
GROUPM
KSTAGES
SMALL_TILE
NVCC_EXTRA 中影响 kernel 的宏
```

它们必须通过 CI 预编译到不同 flavor 包中。

运行时不能切换：

```text
GROUPM
KSTAGES
ARCH SASS
```

运行时可以设置：

```text
TC_PERSIST
TC_TIMING
batch
pool
wallet
worker
agent
```

---

## 3. GPU 架构支持矩阵

当前 generic portable 包目标覆盖：

```text
sm_75
sm_80
sm_86
sm_89
sm_90
compute_90 PTX
```

对应 GPU：

| 架构 | 代表 GPU | 当前定位 |
|---|---|---|
| `sm_75` | RTX 20 系 / Turing | generic 兼容，tuned profile 待测 |
| `sm_80` | A100 / Ampere datacenter | generic 兼容，tuned profile 待测 |
| `sm_86` | RTX 30 系 / 3080 Ti / 3090 | tuned profile 已明确 |
| `sm_89` | RTX 40 系 / Ada / L40 | generic 兼容，需整理 tuned profile |
| `sm_90` | H100 / Hopper | generic 兼容，tuned profile 待测 |
| `sm_120` | RTX 50 系 / Blackwell | 需要 CUDA 13 native package；CUDA 12 generic 只能 PTX JIT |

---

## 4. 当前已知 GPU profile

### 4.1 sm_86 / RTX 30 系 / Ampere gaming

代表：

```text
RTX 3080 Ti
RTX 3090
部分 RTX 30 系
```

已实测最优：

```text
ARCH=sm_86
GROUPM=8
KSTAGES=3
TC_PERSIST=0
```

推荐包：

```text
kan-portable-linux-x64-sm86-g8.tar.gz
```

状态：

```text
已实测；
GROUPM sweep 完成；
KSTAGES=2 不支持；
KSTAGES=4 在 RTX 3080 Ti 上 dynamic smem 超限；
TC_PERSIST=0 优于 TC_PERSIST=1。
```

当前 3080 Ti 最优 controlled benchmark：

```text
search_avg ≈ 99.58 TH/s
total_avg  ≈ 99.03 TH/s
MINE done  ≈ 97.74 TH/s
live pool  ≈ 93-96 TH/s
```

后续方向：

```text
live gap instrumentation
found path 优化
kernel 微优化争取 1-3%
```

### 4.2 sm_75 / RTX 20 系 / Turing

状态：

```text
generic 兼容目标；
tuned profile 未完成。
```

需要验证：

```text
1. CUTLASS path 是否稳定；
2. real cfg 显存是否足够；
3. POSTCHECK ok=1；
4. pool accepted；
5. 是否需要独立 GROUPM / TC_PERSIST；
6. 是否存在性能过低但正确运行的情况。
```

### 4.3 sm_80 / A100 / Ampere datacenter

状态：

```text
generic 兼容目标；
tuned profile 未完成。
```

注意：

```text
A100 与 sm_86 家用 Ampere 不同；
不能直接套用 GROUPM=8 / TC_PERSIST=0。
```

需要验证：

```text
GROUPM
KSTAGES
TC_PERSIST
occupancy
shared memory
power / clock
```

### 4.4 sm_89 / RTX 40 系 / Ada / L40

状态：

```text
generic 包包含 sm_89 SASS；
历史上有 4090 / L40 数据，但需要整理成正式 profile。
```

历史方向：

```text
RTX 4090 kernel 强，但 wall gap 明显；
L40 上 persistent 曾有正收益；
不能套用 sm_86 的 TC_PERSIST=0 结论。
```

需要补齐：

```text
1. sm_89 推荐 GROUPM；
2. TC_PERSIST 默认；
3. controlled benchmark；
4. pool live benchmark；
5. tuned package 名称。
```

### 4.5 sm_90 / H100 / Hopper

状态：

```text
generic 包包含 sm_90 SASS；
tuned profile 未完成。
```

需要单独验证：

```text
GROUPM
KSTAGES
TC_PERSIST
Hopper shared memory / tensor core 行为
```

### 4.6 sm_120 / RTX 50 系 / Blackwell

状态：

```text
CUDA 12.4 portable 包不包含 native sm_120 SASS；
当前只能依赖 compute_90 PTX JIT；
最佳性能需要 CUDA 13 构建 native sm_120 package。
```

需要建立：

```text
CUDA 13 build worker
ARCH=sm_120 tuned package
Blackwell benchmark
```

---

## 5. Release 包策略

### 5.1 每个正式 release 必发

#### generic compatibility package

```text
kan-portable-linux-x64.tar.gz
```

用途：

```text
最大兼容；
NVIDIA 20 系以上大多数 GPU 可运行；
不承诺最优性能。
```

#### sm86 production tuned package

```text
kan-portable-linux-x64-sm86-g8.tar.gz
```

用途：

```text
RTX 30 系 / sm_86 推荐生产包。
```

参数：

```text
ARCH=sm_86
GROUPM=8
KSTAGES=3
TC_PERSIST=0
```

### 5.2 条件成熟后增加

```text
kan-portable-linux-x64-sm75-*.tar.gz
kan-portable-linux-x64-sm80-*.tar.gz
kan-portable-linux-x64-sm89-*.tar.gz
kan-portable-linux-x64-sm90-*.tar.gz
kan-portable-linux-x64-sm120-*.tar.gz
```

每个 tuned 包发布前必须满足：

```text
1. smoke POSTCHECK ok=1；
2. controlled benchmark 完成；
3. pool accepted 正常；
4. rejected 不异常；
5. BUILD_INFO 记录完整；
6. RELEASE_NOTES 标明适用 GPU；
7. 与 generic / 上一生产包有对比数据。
```

### 5.3 生产包和实验包分离

生产推荐包：

```text
sm86-g8
```

实验包：

```text
sm86-g4-k3
sm86-g12-k3
sm86-g16-k3
sm86-g24-k3
```

已知不适用于 RTX 3080 Ti：

```text
sm86-g*-k4
```

release notes 必须明确：

```text
Recommended
Experimental
Known invalid / not for production
```

---

## 6. 包元数据要求

每个 portable 包必须包含：

```text
VERSION
BUILD_INFO.txt
RELEASE_NOTES.txt
run.sh
status.sh
```

`BUILD_INFO.txt` 至少包含：

```text
version
commit
portable
arch
groupm
kstages
small_tile
package_flavor
toolchain
runtime_dependency
```

`run.sh` / `status.sh` 应打印：

```text
version
commit
arch
package_flavor
groupm
kstages
TC_PERSIST
TC_TIMING
GPU model
compute capability
```

目的：

```text
远程排查时一眼确认是否跑了正确 package 和正确 runtime env。
```

---

## 7. 部署脚本策略

应提供生产级下载/部署脚本，例如：

```text
install_kan.sh
```

职责：

```text
1. 检测 nvidia-smi；
2. 读取 GPU compute capability；
3. 根据 GPU profile 选择 package；
4. 下载 release tar.gz；
5. 解压；
6. 打印 BUILD_INFO；
7. 启动 run.sh 或给出启动命令。
```

伪逻辑：

```bash
case "$SM" in
  sm_86)
    pkg="kan-portable-linux-x64-sm86-g8.tar.gz"
    ;;
  sm_89)
    pkg="kan-portable-linux-x64-sm89-<profile>.tar.gz"
    # 如果 tuned 包不存在，则 fallback generic
    ;;
  sm_90)
    pkg="kan-portable-linux-x64-sm90-<profile>.tar.gz"
    ;;
  *)
    pkg="kan-portable-linux-x64.tar.gz"
    ;;
esac
```

原则：

```text
目标机器只下载运行，不编译。
```

---

## 8. Benchmark / 验收标准

每个架构 profile 必须有统一 benchmark 记录：

```text
bench/results/<date>_<gpu>_<arch>.csv
bench/results/<date>_<gpu>_<arch>.md
```

至少记录：

```text
GPU model
driver
CUDA / toolchain
package version
commit
arch
groupm
kstages
TC_PERSIST
power limit
temperature
SM clock
mem clock
POSTCHECK
search TH/s
total TH/s
MINE done TH/s
pool 10s / 60s / 15m TH/s
accepted
rejected
submit timeout
```

进入生产推荐包的最低要求：

```text
POSTCHECK ok=1
pool accepted 正常
rejected 不异常
长时间运行不崩
性能不低于 generic / 上一生产包
```

---

## 9. 当前整改优先级

### P0：文档与包策略收敛

```text
1. 明确 generic vs tuned package；
2. 明确 sm86-g8 是 RTX 30 系推荐生产包；
3. 实验包降级；
4. KSTAGES=4 sm86 包标记为 RTX 3080 Ti 不可用；
5. release notes 写清每个包用途。
```

### P0：建立 GPU profile 权威表

可以从本文件拆分出：

```text
GPU_PROFILES.md
```

作为：

```text
release
install script
README
benchmark
```

的统一依据。

### P1：目标机器自动选择下载包

实现：

```text
检测 GPU -> 选择 tuned 包 -> fallback generic
```

不做现场编译。

### P1：多架构 benchmark 计划

优先级：

```text
1. sm_86 已测，补 live gap；
2. sm_89 / RTX 4090 或 L40；
3. sm_75 / RTX 20 系兼容验证；
4. sm_80 / A100；
5. sm_90 / H100；
6. sm_120 / Blackwell，需 CUDA 13。
```

### P2：逐架构优化

不能把 sm_86 经验全局套用。

每个架构单独确定：

```text
GROUPM
KSTAGES
TC_PERSIST
package flavor
```

---

## 10. 3080 Ti / sm_86 后续工作

sm_86 低风险参数扫描已经基本结束：

```text
GROUPM=8 最优；
KSTAGES=3 唯一可行；
TC_PERSIST=0 最优。
```

继续冲 100+ 的方向：

```text
1. live gap instrumentation；
2. found path CPU rederive / proof 优化；
3. kernel 微优化争取 1-3%；
4. 不再盲目增加 GROUPM/KSTAGES 包。
```

当前 live gap：

```text
controlled MINE done ≈ 97.74 TH/s
live pool ≈ 93-96 TH/s
```

需要记录：

```text
GPU search 时间
draw loop 时间
found rederive 时间
proof 时间
submit wait
job_abort wait
pool notify 间隔
status/log 开销
```

---

## 11. 正式生产版本定义

一个合格的正式生产版本应满足：

```text
1. generic 包存在并可运行；
2. 已实测主力架构存在 tuned 包；
3. tuned 包参数在 BUILD_INFO 中可审计；
4. run.sh 自动设置该包对应 runtime 默认值；
5. install script 能根据 GPU 选择正确包；
6. README / RELEASE_NOTES 明确推荐包；
7. benchmark 数据可追溯；
8. correctness / accepted share 通过；
9. 不把实验包伪装成生产包；
10. 不要求目标机器编译。
```

---

## 12. 结论

正式生产版本不能只为 RTX 3080 Ti 优化。

正确路线是：

```text
多架构 GPU profile
+
CI 预编译 tuned portable 包
+
目标机器自动选择下载 package
+
分架构 benchmark / 验收
+
逐架构优化
```

当前已经明确：

```text
sm_86 profile:
  ARCH=sm_86
  GROUPM=8
  KSTAGES=3
  TC_PERSIST=0
```

但还需要补齐：

```text
sm_75
sm_80
sm_89
sm_90
sm_120
```

的支持状态、推荐包、benchmark 和优化计划。

推荐下一步：

```text
1. 将本文件作为生产 GPU 支持总纲；
2. 拆出 GPU_PROFILES.md；
3. 整理 .cnb.yml release 矩阵；
4. 保留 generic + sm86-g8 生产包；
5. 实现目标机器自动选择下载包；
6. 开始 sm89 / sm75 / sm80 / sm90 / sm120 profile 补齐；
7. 同时推进 sm86 live gap / kernel 微优化。
```

