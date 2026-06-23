# Kan

**Kan** — 高性能 Pearl (PRL) PoUW 挖矿软件。基于 CUTLASS int8 Tensor-Core 内核，RTX 3090 (sm_86) 实测 **106+ TH/s**（随 release 发布 tuned 包）。RTX 4090 / 5090 更高算力来自实验分支，尚未随正式 release 发布——详见下方[性能参考](#性能参考)。

- 矿池挖矿（LuckyPool / Kryptex）
- Solo 挖矿（pearld RPC）
- GPU 全流程：RNG + blake3 哈希 + 噪声生成 + 搜索均在 GPU 完成
- 实时算力显示（15 秒首表，500ms 采样）
- NVML 硬件监控（温度 / 风扇 / 功耗 / 能效）
- pool share 异步提交：网络 `submit_wait` 不阻塞下一轮 mining
- stale proof early-abort：新 job 到达后跳过不可提交的过期 proof
- 零开发费

---

## 目录

- [硬件要求](#硬件要求)
- [快速开始](#快速开始)
- [便携版发布包（开箱即用）](#便携版发布包开箱即用)
- [依赖安装](#依赖安装)
- [克隆与编译](#克隆与编译)
- [运行](#运行)
  - [矿池模式](#矿池模式)
  - [Solo 模式](#solo-模式)
- [命令行参数](#命令行参数)
- [运行时交互](#运行时交互)
- [日志输出格式](#日志输出格式)
- [性能参考](#性能参考)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 硬件要求

| 项目 | 要求 |
|------|------|
| **GPU** | NVIDIA Turing / RTX 20 系及以上，Compute Capability ≥ 7.5；当前 production 包覆盖 `sm_75` 起 |
| **显存** | ≥ 4 GB（实配使用约 2 GB） |
| **CUDA** | 12.x（已在 12.8 上测试通过） |
| **系统** | Linux x86_64（已在 Ubuntu 22.04 上测试通过） |
| **CPU** | 无特殊要求（GPU 执行全部计算） |

### GPU profile 状态

正式发布采用 **generic + tuned package** 体系，而不是要求矿机切换分支或现场编译。
完整权威表见 [`GPU_PROFILES.md`](GPU_PROFILES.md)。

| GPU / 架构 | 当前 release 定位 | 推荐包 |
|-----|------|------|
| V100 / V100S / `sm_70` | **不属于当前 production 支持范围**；portable 包不含 `sm_70` SASS/PTX | 不推荐 / 不支持 |
| RTX 20 系 / `sm_75` | generic 兼容目标；tuned profile 待测 | `kan-portable-linux-x64.tar.gz` |
| A100 / `sm_80` | generic 兼容目标；tuned profile 待测 | `kan-portable-linux-x64.tar.gz` |
| RTX 3080 Ti / RTX 3090 / `sm_86` | 已实测 production tuned profile | `kan-portable-linux-x64-sm86-g8.tar.gz` |
| RTX 4090 / L40 / `sm_89` | generic 兼容；历史数据待整理成正式 profile | `kan-portable-linux-x64.tar.gz` |
| H100 / `sm_90` | generic 兼容目标；tuned profile 待测 | `kan-portable-linux-x64.tar.gz` |
| RTX 50 系 / `sm_120` | CUDA 12 只能 PTX JIT fallback；native tuned 需 CUDA 13 | 当前 fallback generic，未来 CUDA 13 native 包 |

> 不能把 `sm_86` 的 `GROUPM=8 / KSTAGES=3 / TC_PERSIST=0` 结论直接套用到
> `sm_75`、`sm_80`、`sm_89`、`sm_90` 或 `sm_120`。每个 tuned profile 都需要
> benchmark、POSTCHECK 和 pool accepted 记录后才能成为默认推荐。
>
> Volta / `sm_70`（例如 V100/V100S）虽然是 Tensor-Core GPU，但当前
> `tc_cutlass_v2.cu` 生产内核和 portable fatbin 面向 `sm_75+` / Sm80 风格
> int8 Tensor-Core 路径；`sm_70` 不在当前 release 覆盖范围内。

---

## 快速开始

如果只是部署到矿机，优先下载 Release 里的便携包；它不需要 CUDA Toolkit、
CUTLASS 或编译器。下面的源码编译流程只适合开发/自编译。

```bash
# 1. 安装依赖
sudo apt install -y libssl-dev

# 2. 克隆 CUTLASS（仅首次）
git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass

# 3. 克隆 Kan
git clone https://cnb.cool/wuyueyi/peral kan
cd kan

# 4. 编译
export CUTLASS_HOME=~/cutlass
./build.sh

# 5. 运行矿池挖矿
./build/kan --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet 你的PRL地址.矿工名
```

---

## 便携版发布包（开箱即用）

每个正式版本通过 **git tag** 触发 CNB Release，自动生成下载即用的 portable
包。当前生产发布分两层：

```text
kan-portable-linux-x64.tar.gz
  generic compatibility package
  面向 NVIDIA RTX 20 系 / Turing 及以上大多数 GPU；
  能运行优先，不承诺单架构最优性能。

kan-portable-linux-x64-sm86-g8.tar.gz
  tuned production package
  面向 sm_86 / RTX 3080 Ti / RTX 3090 class GPU；
  ARCH=sm_86 GROUPM=8 KSTAGES=3，默认 TC_PERSIST=0。
```

完整 GPU profile 与自动选包依据见 [`GPU_PROFILES.md`](GPU_PROFILES.md)。包内包含：

- `kan`：当前主程序；
- `pearl-miner`：兼容旧启动脚本的同构别名；
- `plainproof_gen`：离线 proof / kernel benchmark；
- `run.sh`：检查 NVIDIA 驱动、设置 package 默认运行参数，并在 pool 模式默认
  自动重启/重连；
- `status.sh`：进程、GPU、hashrate、share、perf 和 runtime event 快照；
- `VERSION`、`BUILD_INFO.txt`、`RELEASE_NOTES.txt`：版本号、构建信息和该
  tag 的版本说明。
- `CHANGELOG.md`：公开生产变更记录。
- `GPU_PROFILES.md`：generic / tuned GPU profile 权威表。
- `install_kan.sh`：按 GPU profile 选择 tuned 包或 fallback generic 的安装/更新脚本。

矿机上只需要 NVIDIA 驱动和 Linux x86-64 / glibc ≥ 2.35：

```bash
# 直接使用已下载的 generic 包
tar xzf kan-portable-linux-x64.tar.gz
cd kan-portable-linux-x64
./run.sh --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet 你的PRL地址.矿工名 \
  --batch 1000 --cfg real --tc
```

或者使用安装脚本自动选包：

```bash
# 默认根据 nvidia-smi 检测 GPU：
#   sm_86 -> kan-portable-linux-x64-sm86-g8.tar.gz
#   其他 / 未调优架构 -> kan-portable-linux-x64.tar.gz
VERSION=v1.2.17 ./install_kan.sh

# 如 release 地址不是默认 CNB 格式，可显式指定：
RELEASE_BASE_URL=https://example/releases/v1.2.17 ./install_kan.sh
```

发版前可先做不依赖 GPU 的静态检查：

```bash
bash check_release_profiles.sh
```

它会验证 `.cnb.yml` 只发布 generic + `sm86-g8` 默认资产、历史 sweep 包未被默认上传、
`install_kan.sh` 选包矩阵正确，以及 portable 元数据生成钩子存在。
完整 L0-L3 测试流程见 [`PRODUCTION_GPU_TEST_PLAN.md`](PRODUCTION_GPU_TEST_PLAN.md)。

如果要直接使用 CNB 的 GPU runner 做发布前 L2 验证，可在 `main` 分支详情页点击
`.cnb/web_trigger.yml` 暴露的 **GPU 验证** 按钮。它会触发
`web_trigger_gpu_verify` 流水线，申请 `cnb:arch:amd64:gpu` runner，并执行：

```text
check_release_profiles.sh
build.sh
run_test.sh
real-cfg easy-target POSTCHECK ok=1
generic portable package build
generic portable package POSTCHECK ok=1
```

这个 CNB GPU 验证默认用于 **generic package / 当前分配 GPU** 的 L2 correctness
gate。若 CNB 分配的是 L40 / H20 等 datacenter GPU，它不能替代 RTX 3080 Ti /
3090 上的 `sm86-g8` tuned profile 实机性能验收；只有当 runner 实际是 `sm_86`
时，才可作为 sm86-g8 runtime 验证依据。可选的 live pool smoke 只有在手动设置
`GPU_VERIFY_POOL_SECONDS>0` 且提供 `KAN_WALLET` 时才会运行。

如果不想通过网页按钮，也可以把当前 commit 推到专用验证分支自动触发同一套 GPU
流水线：

```bash
git push origin HEAD:gpu-verify
```

该分支只用于 CNB GPU 验证，不作为发布分支；验证通过后再按 tag 流程发 Release。

发版约定：

```bash
# 在 main 通过 build/cpu_test 后，创建带说明的 tag 即可触发 Release
git tag -a v1.2.17 -m "v1.2.17 — production release: async submit, stale-proof abort, docs/profile refresh"
git push origin v1.2.17
```

`package_portable.sh` 会同时产出版本化文件名和稳定上传别名，例如：

```text
dist/kan-portable-linux-x64-<version>.tar.gz
dist/kan-portable-linux-x64.tar.gz
dist/kan-portable-linux-x64-<version>-sm86-g8.tar.gz
dist/kan-portable-linux-x64-sm86-g8.tar.gz
```

稳定别名用于 CNB 附件上传，包内文件保存精确版本信息。

### 显卡专用便携包

3080 Ti / 3090 等 Ampere `sm_86` 显卡不要只看 generic 包；Release 会上传
生产推荐 tuned 包：

```text
kan-portable-linux-x64-sm86-g8.tar.gz
```

它使用当前已实测最优的 Ampere 编译参数：

```bash
ARCH=sm_86 GROUPM=8 KSTAGES=3 WITH_AB=0 PACKAGE_FLAVOR=sm86-g8 bash package_portable.sh
```

generic 包优先兼容多架构；`sm86-g8` 包优先 3080 Ti / 3090 生产速度。`sm86-g8`
包内 `run.sh` 默认设置：

```bash
TC_PERSIST=0
KAN_RESTART=auto   # pool 模式默认；断连/退出后自动重启
```

如果要做一次性调试运行，可显式关闭：

```bash
KAN_RESTART=0 ./run.sh --algo pearl --pool ...
```

包内 `BUILD_INFO.txt` 会记录 `arch`、`groupm`、`kstages`、`package_flavor`，
`run.sh` / `status.sh` 会打印 package 与 runtime 信息。排查性能时先确认这些字段。

以下历史 sweep 包不再作为正式 release 默认资产：

```text
sm86-g4-k3 / sm86-g12-k3 / sm86-g16-k3 / sm86-g24-k3
sm86-g*-k4
```

其中 `KSTAGES=4` 在 RTX 3080 Ti 上已知 dynamic shared memory 超限，不能作为
3080 Ti 生产候选。`KSTAGES=2` 当前不支持，不应加入 release matrix。

---

## 依赖安装

### 必需

```bash
# CUDA Toolkit（通常随 NVIDIA 驱动安装，确认 nvcc 可用）
nvcc --version

# OpenSSL（用于 TLS 矿池连接和 Solo HTTPS RPC）
sudo apt install -y libssl-dev

# C++ 编译器（GCC 9+ 即可，Ubuntu 22.04 默认满足）
g++ --version
```

### CUTLASS（高速内核必需）

Kan 的 Tensor-Core 内核基于 NVIDIA CUTLASS 3.5.1。**不安装 CUTLASS 会回退到低速 WMMA 内核（~30 TH/s）。**

```bash
git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass
```

安装后设置环境变量：

```bash
export CUTLASS_HOME=~/cutlass
```

也可以将 CUTLASS 放在 `~/cutlass`，build.sh 会自动检测，无需设置环境变量。

---

## 克隆与编译

```bash
git clone https://cnb.cool/wuyueyi/peral kan
cd kan
```

### 编译

```bash
# 设置 CUTLASS 路径（如果不在 ~/cutlass）
export CUTLASS_HOME=/path/to/cutlass

# 一键编译
./build.sh
```

编译成功后输出：

```
BUILD OK:
  build/plainproof_gen   (CLI 证明生成器，独立使用)
  build/kan              (矿池 + Solo 一体化二进制)
```

### 编译选项

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `CUTLASS_HOME` | CUTLASS 源码路径 | `~/cutlass` |
| `ARCH` | nvcc 目标架构（如 `sm_89`） | 自动从 nvidia-smi 检测 |
| `CUDA_HOME` | CUDA Toolkit 安装路径 | `/usr/local/cuda` |

**手动指定架构**（例如交叉编译或自动检测失败时）：

```bash
ARCH=sm_89 ./build.sh     # RTX 4090
ARCH=sm_86 ./build.sh     # RTX 3090 / 3080
ARCH=sm_80 ./build.sh     # RTX 3080 / A100
```

---

## 运行

### 矿池模式

#### 明文 TCP（推荐，端口 7048）

```bash
./build/kan --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet prl1qyouraddress.worker_name
```

#### TLS 加密（端口 8048）

```bash
./build/kan --algo pearl \
  --pool stratum+ssl://prl.kryptex.network:8048 \
  --wallet prl1qyouraddress.worker_name
```

#### 钱包地址格式

```
--wallet PRL地址.矿工名
```

- **PRL 地址**：bech32 格式，以 `prl1` 开头
- **矿工名**：可选，用 `.` 分隔。例如 `prl1abc...xyz.rig01`
- 也可以分开指定：`--wallet prl1abc...xyz --worker rig01`

#### 后台运行

```bash
nohup ./build/kan --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet prl1qyouraddress.worker \
  > kan.log 2>&1 &

# 查看日志
tail -f kan.log
```

### Solo 模式

Solo 挖矿需要本地运行 pearld 节点，且需要 `zkprove` 工具将 PlainProof 转换为 ZK 证明。

```bash
./build/kan --solo \
  --node 127.0.0.1:44107 \
  --rpcuser your_rpc_user \
  --rpcpass your_rpc_pass \
  --addr prl1qyour_p2tr_address \
  --zkprove /path/to/zkprove
```

---

## 命令行参数

### 矿池模式参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--algo pearl` | 指定算法（仅支持 pearl） | 必填 |
| `--pool URL` | 矿池地址，格式 `stratum+tcp://host:port` 或 `stratum+ssl://host:port` | 必填 |
| `--wallet ADDR[.WORKER]` | PRL 钱包地址，可带矿工名 | 必填 |
| `--worker NAME` | 矿工名（也可合并在 wallet 中用 `.` 分隔） | `pm` |
| `--agent STRING` | 自定义 agent 标识 | `Kan/1.0.0` |
| `--batch N` | 每轮最大 draw 数（新 job 前最多搜索 N 次） | `1000` |
| `--breakdown` | 打印每 draw 的详细计时 | 关闭 |

### Solo 模式参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--solo` | 启用 solo 模式 | — |
| `--node HOST:PORT` | pearld 节点地址 | `127.0.0.1:44107` |
| `--rpcuser USER` | RPC 用户名 | 必填 |
| `--rpcpass PASS` | RPC 密码 | 必填 |
| `--addr ADDR` | P2TR 接收地址 | 必填 |
| `--zkprove PATH` | zkprove 工具路径 | `./zkprove` |

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--cfg real` | 使用真实网络配置 (m=n=131072, k=4096, rank=256) | `real` |
| `--tc` | 强制使用 Tensor-Core 内核 | 开启 |
| `--help` | 显示帮助信息 | — |

---

## 运行时交互

矿机运行中可按以下按键：

| 按键 | 功能 |
|------|------|
| `s` | 立即打印统计表（否则每 120 秒自动打印一次） |
| `q` | 优雅退出 |

---

## 日志输出格式

Kan 的日志格式与 lpminer 兼容，包含以下内容：

### 启动信息

```
11:08:59  info   about          Kan/1.0.0
11:08:59  info   cpu            AMD EPYC 7542 32-Core Processor (16 threads)
11:08:59  info   algo           pearl
11:08:59  info   pool           stratum+tcp://prl.kryptex.network:7048
11:08:59  info   wallet         <PRL_ADDRESS>.pm
11:08:59  info   worker         pm
11:08:59  info   commands       s (stats), q (quit); table every 120s
11:08:59  info   detected       1 devices - driver 12.8
11:08:59  info   GPU            #0 RTX 4090           23GB sm_89 bus:00 enabled
11:08:59  info   devfee         0%
11:08:59  info   stratum        connecting to stratum+tcp://prl.kryptex.network:7048 ...
11:08:59  info   stratum        authorize: ok wallet=... agent=Kan/1.0.0
11:08:59  info   stratum        new job id=589d9fdd_2097152 height=71733 diff=2097152 seq=1
```

### 统计表（首次 15 秒后显示，之后每 120 秒）

```
-----<PRL_ADDRESS>...pm--------------------stratum+tcp://prl.kryptex.network:7048-----
 DEVICE MODEL              HASHRATE  TEMP  FAN POWER      EFFIC       A    R  LAST
----------------------------------------------------------------------------------------
 GPU #0 RTX 4090           197.00 TH/s    70C   52%  452W  435.5 GH/W       0    0     -
----------------------------------------------------------------------------------------
 10s                       244.40 TH/s                0W           A: 0
 60s                       197.00 TH/s                           R: 0
 15m                       197.00 TH/s                           S: 0
[0 days 00:00:15]-------------------------------------[0.0% accept - ver. 1.0.0]
```

各列说明：
- **HASHRATE**：60 秒窗口平均算力
- **TEMP**：GPU 温度
- **FAN**：风扇转速百分比
- **POWER**：GPU 功耗（瓦特）
- **EFFIC**：能效比（GH/W）
- **A / R**：已接受 / 已拒绝的份额数
- **LAST**：距上次接受份额的时间
- **10s / 60s / 15m**：三个时间窗口的独立算力

### 事件日志

```
11:09:20  info   GPU #0         share accepted        ← 份额被矿池接受
11:09:30  info   stratum        new job id=...         ← 矿池推送新任务
11:09:31  info   GPU #0         share rejected: ...    ← 份额被拒绝（罕见）
```

### Pool runtime 行为

生产 pool 模式默认包含两层运行时保护：

```text
run.sh auto-restart:
  KAN_RESTART=auto 时，pool 断连或进程退出后自动重启/重连。

async share submit:
  fresh share proof 生成后先入队，由 submit worker 发送并等待矿池响应；
  mining 主循环立即进入下一轮，不再被 submit_wait 网络延迟阻塞。

stale proof early-abort:
  如果新 job 在 win/proof 期间到达，过期 proof 会在 CPU rederive、
  POSTCHECK 或 Merkle 阶段之间提前退出，避免为不可提交的 stale share
  继续消耗 CPU。
```

日志中可见：

```text
stratum        async share submit worker active
GPU #0         share accepted submit_wait=...
perf           attempt=... reason=job_abort ...
```

`submit_wait` 本身仍表示矿池响应耗时；优化点是 mining loop 不再等待该响应。

---

## 性能参考

### 内核优化历程

| 阶段 | 技术要点 | 3090 TH/s |
|------|---------|-----------|
| 手写 WMMA | dp4a → WMMA 基础内核 | 30 |
| 手写 IMMA | mma.sync.m16n8k32 + register-only fold | 56 |
| CUTLASS 融合 | 3-stage cp.async 流水线 + FoldMmaMultistage | 108 |
| lane-distributed fold | `__reduce_xor_sync` 消除 lane-0 串行瓶颈 | 112 |
| grouped raster | GROUPM=8 列优先栅格化，L2 命中 93.9% | 108 |
| GPU-resident pipeline | GPU 端 RNG + blake3 树哈希 + 噪声生成 | 106.5 wall |
| async overlap | search(N) 与 prep(N+1) 重叠执行 | 106.5 wall |

### Release / baseline 性能参考

| GPU / 架构 | 包定位 | 核心/controlled | live / 端到端 | 记录 |
|-----|------|------|------|------|
| RTX 3080 Ti / RTX 3090 / `sm_86` | 已发布 `sm86-g8` production tuned package | 约 102-106 TH/s | 约 100+ TH/s | [`bench/results/2026-06-22_rtx3080ti_sm86_vps.md`](bench/results/2026-06-22_rtx3080ti_sm86_vps.md) |
| RTX 5090 / `sm_120` | 当前 `v1.2.15` generic fallback baseline，非 tuned profile | controlled total avg 约 319-325 TH/s | live 60s 约 287-300 TH/s | [`2026-06-22`](bench/results/2026-06-22_rtx5090_sm120_vps.md), [`2026-06-23`](bench/results/2026-06-23_rtx5090_sm120_vps106.md) |
| RTX 4090 / L40 / `sm_89` | generic 兼容；历史数据待整理成正式 profile | 历史实验约 260 TH/s | 历史实验约 190-220 TH/s | 待整理 |

> 实际收益取决于网络难度、矿池分配、submit 等待、job abort 和 found/proof 路径。
> 表中只有带 `bench/results/` 记录的数字可作为当前公开 release 的可追溯依据。
>
> RTX 5090 / `sm_120` 当前公开包仍是 CUDA 12 generic PTX fallback；最佳性能需要
> CUDA 13 native `sm_120` package 通过 POSTCHECK、controlled benchmark 和 live
> pool accepted 验证后，才能升级为自动选择的 tuned production package。

---

## 项目结构

```
kan/
├── build.sh              # 一键编译脚本
├── package_portable.sh   # 便携包打包脚本（下载即用）
├── README.md             # 本文件
├── src/
│   ├── miner_main.cpp     # 主程序（矿池 + Solo + 日志 + NVML）
│   ├── plainproof_gen.cpp # PlainProof 生成器（挖矿核心循环）
│   ├── tc_cutlass_v2.cu   # CUTLASS 融合 Tensor-Core 搜索内核
│   ├── gpu_prep.cu        # GPU 端 RNG + blake3 + 噪声生成
│   └── prover.h           # MineParams / MineResult API 定义
├── blake3/                # BLAKE3 哈希库（含 SIMD 汇编加速）
└── build/                 # 编译输出目录
    ├── kan                # 挖矿二进制
    └── plainproof_gen     # 独立证明生成器 CLI
```

---

## 常见问题

### 编译报错 "CUTLASS not found"

```bash
git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass
./build.sh
```

**不装 CUTLASS 也能编译**，但会回退到 WMMA 内核，算力仅 ~30 TH/s。

### 编译报错 "nvcc: command not found"

CUDA Toolkit 未加入 PATH：

```bash
export PATH=/usr/local/cuda/bin:$PATH
./build.sh
```

### NVML 温度/风扇/功耗显示为 `--`

非致命问题，统计表正常工作。确保系统中存在 `libnvidia-ml.so.1`（随 NVIDIA 驱动安装）。

### 算力偏低（低于预期的 50%）

1. **检查 GPU 时钟频率**：`nvidia-smi` 查看，核心应在 1500MHz+
2. **杀掉残留 profiler**：`sudo pkill ncu`（ncu 会限制 SM 频率）
3. **确认 CUTLASS 内核已链接**：编译输出应显示 `CUTLASS at ~/cutlass -> tc_cutlass_v2`

### 矿池拒绝所有份额

1. 检查钱包地址格式：PRL bech32 地址，以 `prl1` 开头
2. 确认矿池可达：`telnet prl.kryptex.network 7048`
3. TLS 连接失败时，改用明文端口 7048
4. 查看日志中是否有 `share rejected` 信息

### GPU 温度过高

Kan 使 GPU 满载运行。建议：
- 确保机箱散热良好
- 监控 `nvidia-smi` 温度（80°C 以下安全）
- 必要时降低功耗限制：`sudo nvidia-smi -pl 300`（以瓦特为单位）

---

## 技术细节

### CUTLASS 内核配置

```
DefaultMma<int8 RowMajor, int8 ColMajor, int32 RowMajor,
           TensorOp, Sm80,
           TBShape 128×256×64,
           WarpShape 64×64×64,
           InstShape 16×8×32,
           stages=3, OpMultiplyAddSaturate>
```

- 融合 fold 回调：每 4 次 mac_loop_iter（rank/64）触发一次
- Grouped raster `GROUPM=8`：列优先 blockIdx 分配，最大化 L2 命中
- lane-distributed fold：32 lane ↔ 32 jackpot tile 一一对应，单条 warp-wide 并行 RMW
- 异步 search/prep 重叠：search(N) 与 prep(N+1) 并行执行

### GPU-Resident Draw Pipeline

每 draw 的 CPU 开销从 1490ms 降至 ~10ms：
- **RNG fill**：closed-form splitmix64，GPU 并行（1.2ms）
- **blake3 tree hash**：512MiB 矩阵的 keyed blake3 树哈希（3.7ms）
- **noise add**：每行 keyed blake3 + permutation diff（5.5ms）

---

## 致谢

- **Pearl 区块链**：https://github.com/pearl-network/pearl
- **NVIDIA CUTLASS**：https://github.com/NVIDIA/cutlass
- **BLAKE3**：https://github.com/BLAKE3-team/BLAKE3

---

*RTX 4090: 260 TH/s kernel / 190+ TH/s wall · RTX 3090: 106+ TH/s · 100% accept rate · Zero devfee*
