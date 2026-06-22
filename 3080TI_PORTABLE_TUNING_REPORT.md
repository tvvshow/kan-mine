# RTX 3080 Ti / Portable Package Tuning Report

日期：2026-06-22  
范围：`peral/` active miner、CI 预编译 portable 包、RTX 3080 Ti / sm_86 调优  
性质：阶段性分析报告 + 整改建议 + 后续方向

---

## 1. 背景与目标

当前 `peral/` 项目是 Pearl(PRL) PoUW 自研 C++/CUDA miner。核心目标是通过
CUTLASS fused int8 tensor-core kernel 在 NVIDIA GPU 上获得高性能 mining 能力。

当前重点目标：

```text
RTX 3080 Ti / sm_86 上稳定达到 100 TH/s 以上。
```

部署模型已经明确：

```text
不在目标 VPS / GPU 机器上现场编译；
CI / workflow 预先编译 portable 便携包；
目标机器只下载、解压、运行对应 GPU 架构的 tuned package。
```

因此，后续设计重点不是“目标机器本地自动编译调参”，而是：

```text
release 预编译 flavor 包矩阵
+
目标机器选择正确 portable 包
+
run.sh 设置运行期参数
```

---

## 2. 正确的 portable 发布模型

### 2.1 编译期参数必须在 CI 固化

关键性能参数：

```text
ARCH
GROUPM
KSTAGES
```

是编译期参数，在 `build.sh` 中通过 nvcc macro 固化：

```bash
-arch=${ARCH}
-DGROUPM=${GROUPM}
-DKSTAGES=${KSTAGES}
```

因此：

```text
GROUPM=8 的 binary
GROUPM=12 的 binary
KSTAGES=3 的 binary
KSTAGES=4 的 binary
```

是不同编译产物。目标机器只下载 portable 包时，不能运行时切换这些参数。

### 2.2 运行期可调整参数

运行时可调整的是：

```text
TC_PERSIST
TC_TIMING
batch
pool URL
wallet
worker
agent
```

其中 `TC_PERSIST` 对 sm_86 很关键，最后实测确认：

```text
RTX 3080 Ti 上 TC_PERSIST=0 明显更快。
```

---

## 3. RTX 3080 Ti / sm_86 已知最优参数

当前已知最优组合：

```text
ARCH=sm_86
GROUPM=8
KSTAGES=3
TC_PERSIST=0
```

对应推荐 portable 包：

```text
kan-portable-linux-x64-sm86-g8.tar.gz
```

该包应在 `BUILD_INFO.txt` 中记录：

```text
arch: sm_86
groupm: 8
kstages: 3
package_flavor: sm86-g8
```

运行时应默认：

```text
TC_PERSIST=0
```

---

## 4. RTX 3080 Ti 实测结果

### 4.1 TC_PERSIST 对比

v1.2.11 `sm86-g8-k3` controlled benchmark：

#### TC_PERSIST=1

```text
samples=11
prep_med=3.93ms
search_med=98.13 TH/s
search_avg=97.49 TH/s
total_med=97.59 TH/s
total_avg=96.96 TH/s
MINE done=94.37 TH/s
```

#### TC_PERSIST=0

```text
samples=11
prep_med=3.86ms
search_med=99.95 TH/s
search_avg=100.02 TH/s
total_med=99.41 TH/s
total_avg=99.48 TH/s
MINE done=97.83 TH/s
```

结论：

```text
TC_PERSIST=0 明显优于 TC_PERSIST=1。
```

但：

```text
即使用 TC_PERSIST=0，g8-k3 仍未稳定超过 100 TH/s。
```

### 4.2 GROUPM sweep

v1.2.12 对 `GROUPM=4/8/12/16/24` 做过测试：

| 排名 | 包 | GROUPM | KSTAGES | prep 中位数 | search 中位数 | search 平均 | total 中位数 | total 平均 | MINE done |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `sm86-g8` | 8 | 3 | 3.887 ms | 99.45 TH/s | 99.58 TH/s | 98.90 TH/s | 99.03 TH/s | 97.74 TH/s |
| 2 | `sm86-g12-k3` | 12 | 3 | 3.900 ms | 98.91 TH/s | 98.99 TH/s | 98.37 TH/s | 98.45 TH/s | 97.20 TH/s |
| 3 | `sm86-g4-k3` | 4 | 3 | 3.883 ms | 98.86 TH/s | 98.92 TH/s | 98.32 TH/s | 98.38 TH/s | 97.20 TH/s |
| 4 | `sm86-g16-k3` | 16 | 3 | 3.908 ms | 98.10 TH/s | 98.12 TH/s | 97.57 TH/s | 97.59 TH/s | 96.23 TH/s |
| 5 | `sm86-g24-k3` | 24 | 3 | 3.920 ms | 97.00 TH/s | 96.97 TH/s | 96.48 TH/s | 96.45 TH/s | 95.24 TH/s |

结论：

```text
GROUPM=8 最优；
GROUPM=12 / 4 略慢；
GROUPM=16 / 24 更慢；
继续扫大 GROUPM 意义不大。
```

### 4.3 KSTAGES=2

v1.2.13 尝试：

```text
sm86-g8-k2
```

CI 编译失败，错误包括：

```text
identifier "acc" is undefined
identifier "c" is undefined
identifier "itA" is undefined
identifier "itB" is undefined
...
45 errors detected
```

结论：

```text
KSTAGES=2 当前不支持。
```

### 4.4 KSTAGES=4

v1.2.14 生成了：

```text
sm86-g8-k4
sm86-g4-k4
sm86-g12-k4
```

但在 RTX 3080 Ti 上全部 smoke 失败：

```text
tc_cutlass: smem attr (115712 B) err invalid argument
tc_cutlass: LAUNCH err invalid argument (tpb=256, grid=80, smem=115712B)
```

结论：

```text
KSTAGES=4 在 RTX 3080 Ti / sm_86 上 dynamic shared memory 超限，不能运行。
```

### 4.5 当前 3080 Ti 性能现状

当前最优 controlled benchmark：

```text
search_avg ≈ 99.58 TH/s
total_avg  ≈ 99.03 TH/s
MINE done  ≈ 97.74 TH/s
```

live pool：

```text
约 93 - 96 TH/s
```

因此：

```text
低风险参数调优已经基本完成；
仍未稳定 100+；
要继续提升必须进入代码级优化和 live gap 定位。
```

---

## 5. v1.2.10 - v1.2.14 版本结论

### v1.2.10

```text
a887b45 perf(proof): parallelize Merkle tree construction
```

核心价值：

```text
并行化 Merkle tree construction；
减少 found share 后 proof 构造 latency；
是当前稳定 baseline。
```

### v1.2.11

```text
acd29d8 perf(kernel): add optional CUTLASS timing sweep tooling
```

作用：

```text
增加 TC_TIMING=1；
增加 prep+gather/search/total 分段计时；
增加 sweep 脚本；
证明 TC_PERSIST=0 在 3080 Ti 上更快。
```

### v1.2.12

```text
595373a ci(release): add sm86 GROUPM tuning packages
```

作用：

```text
发布 sm86 GROUPM 调参包；
实测确认 GROUPM=8 仍然最好；
没有突破 100+。
```

### v1.2.13

```text
17f6dbd ci(release): add sm86 KSTAGES tuning packages
```

问题：

```text
KSTAGES=2 包编译失败，导致 release 工作流失败。
```

同时加入 portable `run.sh` 中 sm86 默认：

```text
TC_PERSIST=0
```

### v1.2.14

```text
dfa7209 ci(release): remove unsupported sm86 k2 package
```

结果：

```text
移除 KSTAGES=2；
保留 KSTAGES=4 测试包；
KSTAGES=4 包能编译，但 3080 Ti 上 launch fail。
```

实际最佳运行组合仍是：

```text
sm86-g8-k3 + TC_PERSIST=0
```

---

## 6. NVIDIA 20 系以上支持现状

项目设计目标：

```text
支持 NVIDIA Turing / RTX 20 系及以上 GPU，包括家用卡和部分商用/数据中心卡。
```

generic portable 包当前覆盖：

```text
sm_75
sm_80
sm_86
sm_89
sm_90
compute_90 PTX
```

对应：

| 架构 | 代表 GPU |
|---|---|
| `sm_75` | RTX 20 系 / Turing |
| `sm_80` | A100 / Ampere DC |
| `sm_86` | RTX 30 系 / 3080 Ti / 3090 |
| `sm_89` | RTX 40 系 / Ada |
| `sm_90` | H100 / Hopper |
| `compute_90 PTX` | 对新架构 forward compatibility，如部分 Blackwell JIT |

需要区分：

```text
generic 包：覆盖 20 系及以上大多数 NVIDIA GPU，目标是能跑；
tuned 包：只对部分架构固化实测最优参数。
```

当前最明确的 tuned profile：

```text
sm_86:
  GROUPM=8
  KSTAGES=3
  TC_PERSIST=0
```

其他架构：

```text
sm_75 / sm_80 / sm_89 / sm_90 / sm_120
```

仍需逐步补齐 tuned profile。

---

## 7. 当前存在问题

### 问题 1：容易混淆“自动调参”和“预编译 tuned 包”

错误理解：

```text
程序运行时检测显卡，然后自动选择最优 GROUPM/KSTAGES。
```

实际情况：

```text
GROUPM/KSTAGES 是编译期参数；
必须通过 CI 预编译 flavor 包固化；
目标机器不能运行时切换。
```

### 问题 2：sm86 生产推荐包需要明确

应明确推荐：

```text
RTX 3080 Ti / 3090:
  kan-portable-linux-x64-sm86-g8.tar.gz
```

参数：

```text
ARCH=sm_86
GROUPM=8
KSTAGES=3
TC_PERSIST=0
```

### 问题 3：v1.2.14 的 KSTAGES=4 包不应作为生产候选

KSTAGES=4 在 3080 Ti 上：

```text
dynamic shared memory 115712 B 超限；
smoke fail；
不能运行。
```

### 问题 4：live pool 与 controlled benchmark 存在 gap

当前：

```text
controlled MINE done ≈ 97.74 TH/s
live pool ≈ 93-96 TH/s
```

可能来源：

```text
found share 后 CPU rederive / proof 开销；
job_abort 频率；
submit_wait；
pool job 切换；
status/log 统计窗口；
found attempt 低样本波动。
```

### 问题 5：跨架构 tuned profile 不完整

当前明确实测的是：

```text
sm_86
```

其他架构还需要建立：

```text
对应 best GROUPM
对应 KSTAGES
对应 TC_PERSIST
对应 package flavor
对应 benchmark 记录
```

---

## 8. 整改建议

### 8.1 建立 GPU Profile 表

建议新增：

```text
peral/GPU_PROFILES.md
```

内容包括：

```text
架构
代表显卡
编译参数
运行参数
对应 release package
实测结果
状态
```

示例：

| Arch | GPU | Package | ARCH | GROUPM | KSTAGES | TC_PERSIST | 状态 |
|---|---|---|---|---:|---:|---:|---|
| sm_75 | RTX 20 | generic / TBD | sm_75 | TBD | 3 | TBD | 未调优 |
| sm_80 | A100 | TBD | sm_80 | TBD | 3 | TBD | 未调优 |
| sm_86 | RTX 3080 Ti / 3090 | sm86-g8 | sm_86 | 8 | 3 | 0 | 已实测 |
| sm_89 | RTX 4090 / L40 | TBD | sm_89 | TBD | 3 | TBD | 需整理历史数据 |
| sm_90 | H100 | TBD | sm_90 | TBD | 3 | TBD | 未调优 |
| sm_120 | RTX 5090 | sm120 TBD | sm_120 | TBD | TBD | TBD | 需 CUDA 13 |

### 8.2 调整 release 包策略

每个 release 至少包含：

#### 兼容包

```text
kan-portable-linux-x64.tar.gz
```

特点：

```text
multi-arch fatbin；
sm_75/sm_80/sm_86/sm_89/sm_90/compute_90；
能跑优先，不保证最优。
```

#### tuned 包

例如：

```text
kan-portable-linux-x64-sm86-g8.tar.gz
```

特点：

```text
针对已实测架构；
编译期参数固化；
run.sh 设置 runtime 最优参数；
作为生产推荐包。
```

### 8.3 明确 sm86 生产推荐包

推荐：

```text
RTX 3080 Ti / 3090:
  kan-portable-linux-x64-sm86-g8.tar.gz
```

不要推荐：

```text
sm86-g4-k3
sm86-g12-k3
sm86-g16-k3
sm86-g24-k3
sm86-g*-k4
```

除非作为实验包。

### 8.4 部署脚本按 GPU 架构选择 package

建议新增或改造：

```text
install_or_update.sh
```

逻辑：

```text
检测 compute capability：
  sm_86 -> 下载 sm86-g8
  sm_89 -> 下载 sm89 tuned 包，如存在，否则 generic
  sm_90 -> 下载 sm90 tuned 包，如存在，否则 generic
  unknown -> 下载 generic
```

该脚本只下载/运行，不编译。

### 8.5 release notes 明确每个包用途

每个 release 应写清：

```text
generic 包：兼容包；
sm86-g8：RTX 30 系推荐生产包；
其他 sm86-g*-k3：历史实验包 / 不再推荐；
sm86-g*-k4：不适用于 RTX 3080 Ti。
```

### 8.6 运行时打印实际参数

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
```

这样远程排查时可以一眼确认：

```text
是否跑了正确 package；
是否 runtime env 正确。
```

---

## 9. 3080 Ti 后续优化方向

### 9.1 减少 live gap

目标：

```text
让 live pool 更接近 controlled benchmark。
```

当前 gap：

```text
controlled MINE done ≈ 97.74 TH/s
live ≈ 93-96 TH/s
```

需要增加 live timing：

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

重点关注 found path：

```text
CPUPREP_PROOF RNG 约 156-173ms
CPUPREP_PROOF blake3 约 213-231ms
PROOF total 约 131-169ms
```

### 9.2 kernel 微优化拿 1-3%

当前 controlled search：

```text
search_avg ≈ 99.58 TH/s
```

距离稳定 100+ 只差：

```text
约 1% - 3%
```

但 GROUPM/KSTAGES/TC_PERSIST 已经基本没有空间。

下一步应分析：

```text
tc_cutlass_v2.cu:
  raster 映射是否还能减少 L2 miss；
  fold callback 是否可减少寄存器/指令；
  jackpot hash/final path 是否有冗余；
  gather/search overlap 是否还有边界开销；
  persistent path 是否应在 sm86 完全禁用；
  A2 register transcript 路径是否可继续压。
```

### 9.3 不再盲目发参数包

已经排除：

```text
GROUPM=4/12/16/24 均不如 8；
KSTAGES=2 不支持；
KSTAGES=4 sm_86 超限。
```

因此不建议继续：

```text
v1.2.15 = 再加一批 GROUPM/KSTAGES 包
```

除非有新的 kernel 改动或明确假设。

---

## 10. 建议执行路线

### 阶段 1：文档与 release 策略收敛

行动：

```text
1. 新增 GPU_PROFILES.md；
2. README / RELEASE_NOTES 明确 generic vs tuned；
3. 明确 sm86 推荐包是 sm86-g8；
4. 标记 KSTAGES=4 sm86 包不可用于 RTX 3080 Ti；
5. 确保 status.sh 打印 BUILD_INFO 与 TC_PERSIST。
```

### 阶段 2：发布干净生产包

建议后续发布：

```text
v1.2.15
```

定位：

```text
release hygiene / sm86 production package
```

内容：

```text
1. 保留 generic 包；
2. 保留 sm86-g8-k3 生产包；
3. sm86-g8 run.sh 默认 TC_PERSIST=0；
4. 移除或降级标记失败/无收益实验包；
5. release notes 写明：
   RTX 3080 Ti / 3090 use sm86-g8。
```

### 阶段 3：live gap instrumentation

在不改变核心算法的情况下，增加轻量 timing：

```text
每 attempt:
  search elapsed
  found rederive elapsed
  proof elapsed
  submit wait
  job wait / abort reason
```

目标：

```text
解释 controlled 97.7 到 live 93-96 的损失。
```

### 阶段 4：kernel 微优化

只有在 live gap 定位清楚后再动 kernel。

候选方向：

```text
1. fold callback 指令数；
2. raster/L2 locality；
3. final jackpot path；
4. sm86 non-persistent 专门路径简化；
5. 减少不必要 event/sync/log；
6. found path CPU rederive 压缩。
```

每个改动必须：

```text
POSTCHECK ok=1
controlled benchmark 对比
pool accepted 正常
```

---

## 11. 核心结论

1. RTX 3080 Ti 当前已知最优参数：

   ```text
   ARCH=sm_86
   GROUPM=8
   KSTAGES=3
   TC_PERSIST=0
   ```

2. 这些最优参数应通过 CI 预编译 portable 包固化，不应依赖目标机器运行时编译/调参。

3. 当前低风险参数调优已经基本结束，继续靠 GROUPM/KSTAGES 包不能稳定突破 100+。

4. 要稳定超过 100 TH/s，下一步需要做：

   ```text
   live gap 定位
   +
   kernel / found path 代码级微优化
   ```

5. 项目目标支持 NVIDIA 20 系以上家用/商用显卡是合理的；但当前“最优 tuned profile”只对 sm_86 最明确，其他架构还需补齐 profile 和 tuned package。

---

## 12. 推荐下一步

建议下一步做小型整改：

```text
1. 写 GPU_PROFILES.md；
2. 整理 release package 策略；
3. 确认 .cnb.yml 中 sm86-g8 作为生产包保留；
4. 确认 sm86-g8 run.sh 默认 TC_PERSIST=0；
5. 在 release notes 中标注：
   RTX 3080 Ti / 3090 推荐 sm86-g8；
   KSTAGES=4 不适用于 RTX 3080 Ti；
6. 然后进入 v1.2.15 live gap instrumentation。
```

一句话：

```text
先把“预编译 tuned 包体系”整理正确，再做 3080 Ti 的 live gap / kernel 微优化。
```
