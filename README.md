# Kan

**Kan** — 高性能 Pearl (PRL) PoUW 挖矿软件。基于 CUTLASS int8 Tensor-Core 内核，RTX 4090 达 **190+ TH/s**，RTX 3090 达 **106+ TH/s**。

- 矿池挖矿（LuckyPool / Kryptex）
- Solo 挖矿（pearld RPC）
- GPU 全流程：RNG + blake3 哈希 + 噪声生成 + 搜索均在 GPU 完成
- 实时算力显示（15 秒首表，500ms 采样）
- NVML 硬件监控（温度 / 风扇 / 功耗 / 能效）
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
| **GPU** | NVIDIA Tensor-Core 显卡（RTX 20 系及以上，Compute Capability ≥ 7.0） |
| **显存** | ≥ 4 GB（实配使用约 2 GB） |
| **CUDA** | 12.x（已在 12.8 上测试通过） |
| **系统** | Linux x86_64（已在 Ubuntu 22.04 上测试通过） |
| **CPU** | 无特殊要求（GPU 执行全部计算） |

### 已测试显卡

| GPU | 架构 | 内核算力 | 端到端算力 | 功耗 | 推荐分支 |
|-----|------|---------|-----------|------|---------|
| **RTX 5090** | sm_120 (Blackwell) | 354 TH/s | — | ~550W | `arch/5090` |
| **RTX 4090** | sm_89 (Ada) | 260 TH/s | 190-220 TH/s | ~450W | `arch/4090` |
| **RTX 3090** | sm_86 (Ampere) | 112 TH/s | 106 TH/s | ~345W | `main`（`ARCH=sm_86`） |

> 估算其他显卡：RTX 4080 ≈ 160 TH/s，RTX 3080 ≈ 90 TH/s，RTX 4070 ≈ 100 TH/s（基于显存带宽比例）。
>
> **选择分支**：每张卡有对应分支以发挥原生 SASS（5090 原生 sm_120 比 PTX-JIT 快 +3.5%）。
> 矿工代码完全相同，分支只差 `build.sh` 默认架构。完整版本地图见 [`BRANCHES.md`](BRANCHES.md)。

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

每个正式版本通过 **git tag** 触发 CNB Release，自动生成下载即用的
`kan-portable-linux-x64.tar.gz`。包内包含：

- `kan`：当前主程序；
- `pearl-miner`：兼容旧启动脚本的同构别名；
- `plainproof_gen`：离线 proof / kernel benchmark；
- `run.sh`：只检查 NVIDIA 驱动并转发参数到 `kan`；
- `VERSION`、`BUILD_INFO.txt`、`RELEASE_NOTES.txt`：版本号、构建信息和该
  tag 的版本说明。

矿机上只需要 NVIDIA 驱动和 Linux x86-64 / glibc ≥ 2.35：

```bash
tar xzf kan-portable-linux-x64.tar.gz
cd kan-portable-linux-x64
./run.sh --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet 你的PRL地址.矿工名 \
  --batch 500 --cfg real --tc
```

发版约定：

```bash
# 在 main 通过 build/cpu_test 后，创建带说明的 tag 即可触发 Release
git tag -a v1.2.1 -m "v1.2.1 — portable release: 说明本版变更、性能、兼容性"
git push origin v1.2.1
```

`package_portable.sh` 会同时产出版本化文件名
`dist/kan-portable-linux-x64-<version>.tar.gz` 和稳定上传别名
`dist/kan-portable-linux-x64.tar.gz`；稳定别名用于 CNB 附件上传，包内文件保存
精确版本信息。

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
11:08:59  info   wallet         prl1patz...apmv.pm
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
-----prl1patz2m...apmv---------------------stratum+tcp://prl.kryptex.network:7048-----
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

### 各卡端到端算力

| GPU | 核心算力 | 端到端算力 | 每 draw 耗时 | 每 share 间隔 |
|-----|---------|-----------|------------|-------------|
| RTX 5090 | 354 TH/s | — | ~199ms | ~30s |
| RTX 4090 | 260 TH/s | 190-220 TH/s | ~270ms | ~40s |
| RTX 3090 | 112 TH/s | 106 TH/s | ~650ms | ~90s |

> 实际收益取决于网络难度和矿池分配。以上为算力参考。
>
> **算法已贴近硬件天花板**：5090 内核 354 TH/s = 该卡 dense int8 GEMM roofline(382) 的 93%。
> dense int8 是协议写死的精度（不能用 FP4/稀疏），各卡已跑在 76-93% 峰值，无 2× 空间。
> 详见 [`BRANCHES.md`](BRANCHES.md) §3。

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
