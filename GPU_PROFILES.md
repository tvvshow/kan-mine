# GPU Profiles

日期：2026-06-22
范围：`peral/` production miner、portable release、NVIDIA GPU package selection
性质：GPU 架构 profile 权威表；供 release matrix、install script、README、benchmark 记录统一引用

---

## 1. 目标与边界

本文件定义 `peral/` 正式生产版本的 GPU profile 体系。

正式目标是逐步覆盖 NVIDIA Ampere / Ada / Hopper / Blackwell 以及经验证的 Turing
fallback，而不是只支持 RTX 3080 Ti。当前 production 推荐从 `sm_86` 开始；`sm_75`
(Turing / RTX 20 系) 保留在 generic fatbin 中用于兼容性试跑，但在缺少实机
POSTCHECK + pool accepted 记录前只能标为 experimental fallback。

当前 production release 的实际 CUDA fatbin 从 `sm_75` 开始。Volta / `sm_70`
（V100/V100S）不在当前 production 支持范围内；不要把“有 Tensor Core”误解为
可以运行本项目的 Sm80 风格 int8 CUTLASS 生产内核。

生产发布必须同时区分：

```text
generic package:
  最大兼容，能跑优先，不承诺最优。

tuned package:
  针对已实测 GPU 架构固化最优编译期参数和默认运行期参数。
```

本文件是以下组件的统一依据：

```text
1. CI / workflow release matrix；
2. portable package 命名；
3. install script 自动选包；
4. run.sh / status.sh 默认运行参数；
5. README / RELEASE_NOTES 推荐说明；
6. bench/results/* 的 profile 字段。
```

---

## 2. 参数分层

### 2.1 编译期参数

以下参数会改变 CUDA binary，必须在 CI 中预编译进不同 flavor 包：

```text
ARCH
GROUPM
KSTAGES
SMALL_TILE
影响 kernel 的 NVCC_EXTRA macro
```

目标机器不能在运行时切换这些参数。

错误模型：

```text
目标机器检测显卡 -> 现场编译 -> sweep GROUPM/KSTAGES -> 运行
```

正确模型：

```text
CI 预编译 generic / tuned packages
  ↓
release 发布 tar.gz
  ↓
目标机器检测 GPU 架构
  ↓
下载对应 tuned package；如不存在则 fallback generic
  ↓
运行 ./run.sh
```

### 2.2 运行期参数

以下参数可以由 `run.sh`、环境变量或启动命令设置：

```text
TC_PERSIST
TC_TIMING
batch
pool
wallet
worker
agent
```

其中 `TC_PERSIST` 属于 profile 相关运行期默认值。不同架构不能互相套用。

### 2.3 生产运行期策略

以下行为属于所有 production package 的通用运行期策略，不改变 GPU profile
选择矩阵，也不改变 proof 格式：

```text
pool auto-restart:
  portable run.sh 在 pool 模式默认 KAN_RESTART=auto；
  断连或异常退出后自动重启/重连。

async share submit:
  fresh share proof 生成后由 submit worker 发送并等待矿池响应；
  mining 主循环不等待 submit_wait 网络延迟。

stale proof early-abort:
  如果新 job 在 win/proof 期间到达，过期 proof 会在 CPU rederive /
  POSTCHECK / Merkle 阶段之间提前退出，不提交 stale proof。

multi-GPU auto fanout:
  pool 模式支持单机单卡和单机多卡，属于正式生产运行能力。
  未设置 CUDA_VISIBLE_DEVICES 时，父进程作为 supervisor 自动检测并使用
  所有 GPU，为每张物理 GPU fork 一个隔离 lane 进程；每个 lane 各自建立
  一条 stratum 连接，并以同一 worker 名认证，矿池按 worker 聚合整机。
  --devices 选择物理 GPU 子集；外部设置 CUDA_VISIBLE_DEVICES 时 miner
  尊重它并禁用 auto fanout（单 lane）。--devices 与 CUDA_VISIBLE_DEVICES 互斥。
  说明：这是 parent supervisor + per-GPU isolated lane 模型；单一 stratum
  session / 统一父进程 stats 是未来项，尚未完成。
  multi-GPU 不改变下方 GPU profile 选包矩阵：每个 lane 仍按本文件的架构
  profile 选择编译期/运行期参数。
```

这些策略是 live pool wall-clock 优化；进入 production 的最低要求仍然是：

```text
POSTCHECK ok=1
pool accepted > 0
rejected / stale 不异常
submit_timeout 无风暴
controlled benchmark 不退化
```

---

## 3. Release 包类型

### 3.1 Generic compatibility package

包名：

```text
kan-portable-linux-x64.tar.gz
```

定位：

```text
NVIDIA Ampere / Ada / Hopper 以及部分 Turing fallback 的兼容包；
能运行优先；
不承诺最优性能；
`sm_75` 在获得正式实机记录前属于 experimental fallback。
```

当前 generic 覆盖目标：

```text
sm_75
sm_80
sm_86
sm_89
sm_90
compute_90 PTX
```

注意：

```text
CUDA 12 generic 对 Blackwell / sm_120 只能依赖 forward-compatible PTX JIT；
最佳 sm_120 性能需要 CUDA 13 native sm_120 package。
```

### 3.2 Tuned production package

包名格式：

```text
kan-portable-linux-x64-<profile>.tar.gz
```

示例：

```text
kan-portable-linux-x64-sm86-g8.tar.gz
```

定位：

```text
针对已实测架构；
固化编译期参数；
run.sh 设置该 profile 的默认运行期参数；
作为生产推荐包或候选包。
```

发布 tuned package 前必须具备：

```text
1. POSTCHECK ok=1；
2. official verifier VALID，如该模式需要；
3. controlled benchmark；
4. pool accepted 正常；
5. rejected 不异常；
6. 长时间运行不崩；
7. BUILD_INFO.txt 完整；
8. RELEASE_NOTES.txt 明确适用 GPU、状态和限制；
9. 与 generic / 上一生产包有对比数据。
```

### 3.3 Experimental package

实验包必须显式标记为：

```text
Experimental
Not recommended for production
```

不能把参数 sweep 包伪装成生产推荐包。

---

## 4. Profile 总表

| Arch | 代表 GPU | 推荐 package | 编译 ARCH | GROUPM | KSTAGES | 默认 TC_PERSIST | 状态 | 备注 |
|---|---|---|---|---:|---:|---:|---|---|
| `sm_70` | V100 / V100S / Volta | 不支持当前 release | N/A | N/A | N/A | N/A | 不支持 | portable fatbin 不含 `sm_70`；生产内核不是 Volta 路径 |
| `generic` | Ampere/Ada/Hopper + Turing experimental fallback | `kan-portable-linux-x64.tar.gz` | multi-arch fatbin | build default / generic | 3 | package default | 生产兼容包；sm75 experimental | 能跑优先，不承诺最优 |
| `sm_75` | RTX 20 系 / Turing | generic experimental fallback；tuned TBD | `sm_75` | TBD | 3 | TBD | **未验证 / 可能不支持** | 需要兼容性、正确性、pool accepted 验证；若 shared-memory 超限则视为 unsupported |
| `sm_80` | A100 / Ampere datacenter | generic；tuned TBD | `sm_80` | TBD | 3 | TBD | 待测 | 不可直接套用 sm_86 参数 |
| `sm_86` | RTX 30 系 / RTX 3080 Ti / RTX 3090 | `kan-portable-linux-x64-sm86-g8.tar.gz` | `sm_86` | 8 | 3 | 0 | 已实测；生产推荐 | 当前唯一明确 production tuned profile |
| `sm_89` | RTX 40 系 / RTX 4090 / L40 / Ada | generic；tuned TBD | `sm_89` | TBD | 3 | TBD | 需整理历史数据 | 4090 kernel 强但 wall gap 明显；L40 persistent 可能有正收益 |
| `sm_90` | H100 / Hopper | generic；tuned TBD | `sm_90` | TBD | 3 | TBD | 待测 | 需单独验证 Hopper tensor core / shared memory 行为 |
| `sm_120` | RTX 50 系 / Blackwell | CUDA 13 native package TBD；CUDA 12 fallback generic/PTX | `sm_120` | TBD | TBD | TBD | 需 CUDA 13 | CUDA 12 portable 无 native sm_120 SASS |

---

## 5. 当前生产推荐 profile

### 5.1 `sm_86` / RTX 30 系 / Ampere gaming

适用 GPU：

```text
RTX 3080 Ti
RTX 3090
其他 sm_86 RTX 30 系 GPU，需按显存/功耗/散热确认稳定性
```

推荐包：

```text
kan-portable-linux-x64-sm86-g8.tar.gz
```

编译期参数：

```text
ARCH=sm_86
GROUPM=8
KSTAGES=3
```

运行期默认：

```text
TC_PERSIST=0
```

状态：

```text
Production recommended
```

已知结论：

```text
1. GROUPM=8 是 RTX 3080 Ti 当前最优；
2. GROUPM=4/12 略慢；
3. GROUPM=16/24 更慢；
4. KSTAGES=2 当前不支持，曾导致编译失败；
5. KSTAGES=4 在 RTX 3080 Ti 上 dynamic shared memory 超限；
6. TC_PERSIST=0 明显优于 TC_PERSIST=1；
7. 不应继续盲目发布更多 sm86 GROUPM/KSTAGES 参数包。
```

当前 3080 Ti controlled benchmark：

```text
search_avg ≈ 99.58 TH/s
total_avg  ≈ 99.03 TH/s
MINE done  ≈ 97.74 TH/s
```

补充 VPS 实测记录：

```text
2026-06-22 / RTX 3080 Ti / driver 595.80 / v1.2.11 sm86-g8:
  TC_PERSIST=0, hard target, samples=15
  search_avg ≈ 102.50 TH/s
  total_avg  ≈ 101.93 TH/s
  MINE done  ≈ 99.65 TH/s
  POSTCHECK ok=1 in smoke test

记录文件:
  bench/results/2026-06-22_rtx3080ti_sm86_vps.md
  bench/results/2026-06-22_rtx3080ti_sm86_vps.csv
```

当前 live pool：

```text
≈ 93-96 TH/s
```

需要继续推进：

```text
1. live gap instrumentation；
2. found path CPU rederive / proof 优化；
3. kernel 微优化争取 1-3%；
4. 不再把 KSTAGES=4 作为 RTX 3080 Ti 生产候选。
```

---

## 6. 待补齐 profile

### 6.0 `sm_70` / V100 / V100S / Volta

当前定位：

```text
不属于当前 production release 支持范围。
```

原因：

```text
1. portable fatbin 只包含 sm_75、sm_80、sm_86、sm_89、sm_90 SASS
   以及 compute_90 PTX；
2. 当前 production kernel `tc_cutlass_v2.cu` 使用 Sm80 风格 CUTLASS
   int8 Tensor-Core / cp.async 路径；
3. V100/V100S 的 Volta Tensor Core 路径不同，不能通过 fallback generic
   视为已支持。
```

推荐当前部署：

```text
不要把 V100/V100S 用作 production validation GPU；
不要用它验证 sm86-g8 或 sm120 generic fallback；
如果安装脚本 fallback generic 后运行失败，应记录为 unsupported GPU，
而不是 production regression。
```

若未来需要支持：

```text
需要单独 sm_70 kernel / CUTLASS Volta path、POSTCHECK、controlled benchmark、
official verifier / pool accepted 和长时间稳定性验证。
```

### 6.1 `sm_75` / RTX 20 系 / Turing

当前定位：

```text
generic 兼容目标；
tuned profile 未完成。
```

推荐当前部署：

```text
优先使用 kan-portable-linux-x64.tar.gz；
不要发布 production tuned 包，直到 benchmark / correctness / pool accepted 完成。
```

需验证：

```text
1. CUTLASS path 是否稳定；
2. real cfg 显存是否足够；
3. POSTCHECK ok=1；
4. pool accepted；
5. rejected 是否异常；
6. GROUPM 最优值；
7. KSTAGES=3 是否稳定；
8. TC_PERSIST 默认值；
9. 长时间运行稳定性。
```

### 6.2 `sm_80` / A100 / Ampere datacenter

当前定位：

```text
generic 兼容目标；
tuned profile 未完成。
```

注意：

```text
A100 / sm_80 与 RTX 30 系 / sm_86 不同；
不能直接套用 GROUPM=8 / TC_PERSIST=0。
```

需验证：

```text
1. GROUPM sweep；
2. KSTAGES 支持范围；
3. TC_PERSIST=0/1 对比；
4. occupancy；
5. dynamic shared memory；
6. power limit / SM clock；
7. pool accepted 和长时间稳定性。
```

### 6.3 `sm_89` / RTX 40 系 / Ada / L40

当前定位：

```text
generic 包包含 sm_89 SASS；
历史上有 RTX 4090 / L40 数据；
仍需整理为正式 profile。
```

已知方向：

```text
1. RTX 4090 kernel 可能很强，但 wall gap 需要解释；
2. L40 上 persistent 曾可能有正收益；
3. 不能套用 sm_86 的 TC_PERSIST=0 结论；
4. 需要将历史 benchmark 转成统一 bench/results 记录。
```

需补齐：

```text
1. sm_89 推荐 GROUPM；
2. KSTAGES 是否保持 3；
3. TC_PERSIST 默认值；
4. controlled benchmark；
5. pool live benchmark；
6. tuned package 名称。
```

### 6.4 `sm_90` / H100 / Hopper

当前定位：

```text
generic 包包含 sm_90 SASS；
tuned profile 未完成。
```

需验证：

```text
1. GROUPM；
2. KSTAGES；
3. TC_PERSIST；
4. Hopper shared memory limit；
5. Hopper tensor core 行为；
6. datacenter power / clock 策略；
7. pool accepted 和长时间稳定性。
```

### 6.5 `sm_120` / RTX 50 系 / Blackwell

当前定位：

```text
CUDA 12 portable 包不包含 native sm_120 SASS；
当前只能依赖 compute_90 PTX JIT 作为 fallback；
最佳性能需要 CUDA 13 构建 native sm_120 package。
```

当前 generic fallback baseline：

```text
2026-06-22 / RTX 5090 / sm_120 / driver 595.80 / v1.2.15 generic:
  package:   kan-portable-linux-x64.tar.gz
  arch:      portable-fatbin
  GROUPM:    128
  KSTAGES:   3
  path:      CUDA 12 generic fatbin / compute_90 PTX fallback

  controlled search_avg ≈ 327.82 TH/s
  controlled total_avg  ≈ 324.91 TH/s
  controlled MINE done  ≈ 306.86 TH/s

  live 60s              ≈ 300.36 TH/s
  live early 15m        ≈ 292.73 TH/s
  accepted/rejected     = 17 / 0
  submit_timeout        = 0
  submit_wait_avg       ≈ 0.80s

  record:
    bench/results/2026-06-22_rtx5090_sm120_vps.md
    bench/results/2026-06-22_rtx5090_sm120_vps.csv
    bench/results/2026-06-23_rtx5090_sm120_vps106.md
    bench/results/2026-06-23_rtx5090_sm120_vps106.csv
```

结论：

```text
v1.2.15 generic fallback 在 RTX 5090 上健康可运行，但不是 tuned sm_120
生产 profile。不得把该 fallback 误标为 Blackwell 最优；sm_120 自动选包
仍应 fallback generic，直到 CUDA 13 native package 通过 POSTCHECK、
controlled benchmark、live pool accepted 和长时间稳定性验证。
```

需建立：

```text
1. CUDA 13 build worker；
2. ARCH=sm_120 package；
3. Blackwell benchmark；
4. GROUPM / KSTAGES / TC_PERSIST profile；
5. 与 CUDA 12 generic PTX JIT fallback 的对比。
```

---

## 7. 安装脚本选包规则

安装脚本应做：

```text
检测 GPU compute capability
  ↓
查找本文件中的 tuned profile
  ↓
如果 tuned package 存在且状态为 Production recommended，则下载 tuned
  ↓
否则 fallback generic
```

建议伪逻辑：

```bash
case "$SM" in
  sm_70)
    # Volta / V100 / V100S is not covered by current production packages.
    # Installer may still select generic for dry-run/download consistency, but
    # operators must treat runtime launch failure as unsupported GPU, not a
    # production regression.
    pkg="kan-portable-linux-x64.tar.gz"
    ;;
  sm_86)
    pkg="kan-portable-linux-x64-sm86-g8.tar.gz"
    ;;
  sm_89)
    # tuned profile 未正式发布前必须 fallback generic
    pkg="kan-portable-linux-x64.tar.gz"
    ;;
  sm_75|sm_80|sm_90)
    pkg="kan-portable-linux-x64.tar.gz"
    ;;
  sm_120)
    # CUDA 13 native package 存在后再切换 sm120 tuned；
    # CUDA 12 release 只能 fallback generic/PTX。
    pkg="kan-portable-linux-x64.tar.gz"
    ;;
  *)
    pkg="kan-portable-linux-x64.tar.gz"
    ;;
esac
```

原则：

```text
目标机器只下载、解压、运行；
不 git pull；
不 nvcc 编译；
不现场 sweep。
```

---

## 8. Package 元数据要求

每个 portable 包必须包含：

```text
VERSION
BUILD_INFO.txt
RELEASE_NOTES.txt
run.sh
status.sh
```

`BUILD_INFO.txt` 至少记录：

```text
version
commit
portable
package_flavor
arch
groupm
kstages
small_tile
toolchain
cuda_version
runtime_dependency
build_time
```

`run.sh` / `status.sh` 应打印：

```text
version
commit
package_flavor
arch
groupm
kstages
TC_PERSIST
TC_TIMING
GPU model
compute capability
driver version
CUDA runtime / toolkit info，如可获得
```

目的：

```text
远程排查时一眼确认：
1. 是否跑了正确 package；
2. 是否是 tuned / generic；
3. 编译期参数是否正确；
4. 运行期默认值是否正确；
5. GPU 架构是否与 package 匹配。
```

---

## 9. Benchmark 记录格式

每个 profile 的 benchmark 应归档到：

```text
bench/results/<date>_<gpu>_<arch>.csv
bench/results/<date>_<gpu>_<arch>.md
```

至少记录：

```text
GPU model
compute capability
driver
CUDA / toolchain
package version
commit
package_flavor
arch
groupm
kstages
TC_PERSIST
TC_TIMING
power limit
temperature
SM clock
mem clock
POSTCHECK
official verifier result，如适用
search TH/s
total TH/s
MINE done TH/s
pool 10s TH/s
pool 60s TH/s
pool 15m TH/s
accepted
rejected
submit timeout
job abort / notify stats，如已具备 instrumentation
```

进入 `Production recommended` 的最低条件：

```text
POSTCHECK ok=1
pool accepted 正常
rejected 不异常
长时间运行不崩
性能不低于 generic / 上一生产包
RELEASE_NOTES 明确适用 GPU 与限制
```

---

## 10. 状态定义

| 状态 | 含义 | 是否可作为默认下载 |
|---|---|---|
| `Production recommended` | 已实测、正确性和 pool 通过、优于 generic 或上一生产包 | 可以 |
| `Candidate` | correctness 通过，benchmark 初步可用，但缺少长时间 pool 或跨机器验证 | 默认不选，除非用户显式指定 |
| `Experimental` | 参数 sweep / 假设验证 / 可能不稳定 | 不可默认选 |
| `Generic compatible` | 作为兼容 fallback，能跑优先，不承诺最优 | tuned 不存在时可以 |
| `TBD` | 未完成 profile | 不可 |
| `Known invalid` | 已知不能运行或不适用 | 不可 |

当前状态摘要：

```text
generic: Generic compatible
sm_86 / sm86-g8: Production recommended
sm_75: TBD
sm_80: TBD
sm_89: TBD / historical data needs consolidation
sm_90: TBD
sm_120: TBD / requires CUDA 13 native build
sm86 KSTAGES=4 on RTX 3080 Ti: Known invalid
```

---

## 11. 已知不推荐 / 禁止默认选择

### 11.1 sm86 GROUPM 实验包

以下包只可作为历史实验或对比，不应作为 RTX 3080 Ti / RTX 3090 默认生产推荐：

```text
kan-portable-linux-x64-sm86-g4-k3.tar.gz
kan-portable-linux-x64-sm86-g12-k3.tar.gz
kan-portable-linux-x64-sm86-g16-k3.tar.gz
kan-portable-linux-x64-sm86-g24-k3.tar.gz
```

原因：

```text
RTX 3080 Ti 实测均不优于 sm86-g8。
```

### 11.2 sm86 KSTAGES=4 包

以下类型包不得作为 RTX 3080 Ti 生产候选：

```text
kan-portable-linux-x64-sm86-g*-k4.tar.gz
```

已知错误：

```text
tc_cutlass: smem attr (115712 B) err invalid argument
tc_cutlass: LAUNCH err invalid argument (tpb=256, grid=80, smem=115712B)
```

结论：

```text
KSTAGES=4 在 RTX 3080 Ti / sm_86 上 dynamic shared memory 超限。
```

### 11.3 KSTAGES=2

当前 `KSTAGES=2` 不支持，曾出现编译失败：

```text
identifier "acc" is undefined
identifier "c" is undefined
identifier "itA" is undefined
identifier "itB" is undefined
...
```

在修复 kernel 源码对 KSTAGES=2 的支持前，不得加入 release matrix。

---

## 12. 下一步 profile 补齐顺序

优先级：

```text
P0:
  1. 保持 generic + sm86-g8 生产包清晰；
  2. release notes 明确 generic vs tuned；
  3. install script 以本文件作为选包依据；
  4. sm86-g8 run.sh 默认 TC_PERSIST=0。

P1:
  1. 整理 sm_89 / RTX 4090 或 L40 历史数据；
  2. 建立 sm_89 controlled + live pool benchmark；
  3. 验证 sm_75 兼容性；
  4. 验证 sm_80 / A100；
  5. 验证 sm_90 / H100。

P2:
  1. 建立 CUDA 13 sm_120 build worker；
  2. 发布 Blackwell native package；
  3. 逐架构优化 GROUPM / KSTAGES / TC_PERSIST。
```

---

## 13. 当前结论

当前 production tuned profile 只有一个明确结论：

```text
sm_86:
  package:   kan-portable-linux-x64-sm86-g8.tar.gz
  ARCH:      sm_86
  GROUPM:    8
  KSTAGES:   3
  runtime:   TC_PERSIST=0
  status:    Production recommended
```

项目正式生产路线应保持：

```text
generic compatibility package
+
tuned production packages for validated architectures
+
目标机器自动选择下载 package
+
逐架构 benchmark / correctness / pool accepted 验收
```

在其他架构 profile 补齐前：

```text
sm_75 / sm_80 / sm_89 / sm_90 / sm_120 默认 fallback generic；
不得把 sm_86 经验直接推广为全架构最优；
不得要求目标机器现场编译或调参。
```
