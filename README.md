# Kan — Pearl (PRL) PoUW 高性能 GPU 挖矿软件

[中文](README.md) | **English**

[![Release](https://img.shields.io/badge/release-v1.2.22-blue)](https://github.com/tvvshow/kan-mine/releases/tag/v1.2.22)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-green)](#下载与快速开始)
[![License](https://img.shields.io/badge/license-NonCommercial%20%7C%20NoFee%20%7C%20Attribution-important)](LICENSE)

> **🔴 重要声明（务必阅读）**
>
> 本项目 **100% 开源、零开发费 / 零抽水（Zero Dev Fee）**。
>
> - 矿工提交给矿池的全部有效 share，收益**归矿工本人所有**——源码中**没有任何**算力分流、份额重定向、隐藏 devfee 的逻辑，欢迎审计。
> - **严禁商用**（不得做成收费托管 / 云算力 / 收费套件）。
> - **二次开发禁止加入任何抽水机制**（不得加 devfee、不得分流 share、不得篡改 wallet/worker）。
> - **二次开发必须标注出处**（保留 `Based on Kan by tvvshow/kan-mine`）并保留 [LICENSE](LICENSE) 全文。
>
> 完整法律条款见 [LICENSE](LICENSE) 文件。违反任一条款 = 自动失去授权。

---

## 目录

- [功能特性](#功能特性)
- [硬件要求与 GPU profile](#硬件要求与-gpu-profile)
- [下载与快速开始](#下载与快速开始)
- [矿池挖矿](#矿池挖矿)
- [Solo 挖矿](#solo-挖矿)
- [命令行参数完整参考](#命令行参数完整参考)
- [环境变量与运行时开关](#环境变量与运行时开关)
- [单机多卡（多 GPU）](#单机多卡多-gpu)
- [生产部署（systemd / HiveOS）](#生产部署)
- [运行时交互与日志格式](#运行时交互与日志格式)
- [性能参考](#性能参考)
- [从源码编译](#从源码编译)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [许可与出处](#许可与出处)

---

## 功能特性

- **矿池挖矿**（Stratum V1，LuckyPool / Kryptex 等）+ **Solo 挖矿**（pearld RPC）
- **跨平台**：Linux x86-64 + **Windows x64**，均提供开箱即用便携版
- **高性能 CUTLASS int8 Tensor-Core 内核**（详见[性能参考](#性能参考)）
- **GPU 全流程**：RNG + blake3 树哈希 + 噪声生成 + jackpot 搜索全部在 GPU 完成
- **单机多卡**：pool 模式默认自动 fanout 到所有检测到的 GPU（每卡一个隔离 lane 进程，共享 worker 名）
- **实时算力**：首表 15 秒、500ms 采样；10s/60s/15m 三窗口
- **NVML 硬件监控**：温度 / 风扇 / 功耗 / 能效比
- **异步 share 提交**：网络 `submit_wait` 不阻塞下一轮 mining
- **stale proof early-abort**：新 job 到达后跳过不可提交的过期 proof
- **断线自动重启**：`run.sh` / `run.bat` 内置 pool 模式重启循环
- **多矿池故障转移**：`--pool` 可重复指定，主池不可达自动切换
- **HTTP/JSON 监控 API**：对接 HiveOS / mmpOS / curl（`--api-port`）
- **零开发费 / 零抽水**

---

## 硬件要求与 GPU profile

| 项目 | 要求 |
|------|------|
| **GPU** | NVIDIA Tensor-Core GPU（Ampere / Turing 及以上推荐，见下表） |
| **显存** | ≥ 4 GB（实际使用约 2 GB） |
| **驱动** | 较新的 NVIDIA 驱动（CUDA 12.x 兼容） |
| **系统** | Linux x86-64（glibc ≥ 2.35）或 Windows 10/11 x64 |
| **运行时依赖** | **仅需 NVIDIA 驱动**——便携包已捆绑 CUDA 运行时、OpenSSL、MSVC 运行库，无需安装 CUDA Toolkit / CUTLASS / 编译器 |

### GPU profile 与推荐便携包

完整权威表见 [`GPU_PROFILES.md`](GPU_PROFILES.md)。简表：

| GPU / 架构 | 推荐包 | 状态 |
|-----|------|------|
| RTX 20 系 / Titan RTX / `sm_75` | `kan-portable-linux-x64-sm75.tar.gz` | 专用 WMMA 包，POSTCHECK ok=1（Candidate）|
| RTX 3080 / 3090 / 3080 Ti / `sm_86` | `kan-portable-linux-x64-sm86-g8.tar.gz` | **tuned production package**，GROUPM=8/KSTAGES=3 |
| RTX 4090 / L40 / `sm_89` | `kan-portable-linux-x64.tar.gz` 或 Windows zip | generic 多架构 fatbin |
| A100 / `sm_80`、H100 / `sm_90` | `kan-portable-linux-x64.tar.gz` | generic 兼容 |
| RTX 50 系 / `sm_120` | `kan-portable-linux-x64.tar.gz` | CUDA 12 PTX JIT fallback；native tuned 需 CUDA 13 |
| Windows 任意上述卡 | `kan-portable-windows-x64.zip` | sm_75 + sm_86 WMMA 路径（v1） |

> **Volta / `sm_70`（V100/V100S）不支持**当前 release——内核面向 `sm_75+` 风格 int8 Tensor-Core。

---

## 下载与快速开始

### 🔽 下载

从 GitHub Release 下载对应平台的便携包（无需编译）：

👉 **https://github.com/tvvshow/kan-mine/releases/latest**

| 文件 | 平台 | 说明 |
|---|---|---|
| `kan-portable-linux-x64.tar.gz` | Linux | **generic compatibility package**（Ampere/Ada/Hopper + Blackwell PTX）|
| `kan-portable-linux-x64-sm86-g8.tar.gz` | Linux | 3080/3090 调优包 |
| `kan-portable-linux-x64-sm75.tar.gz` | Linux | Turing(2080Ti)专用 |
| `kan-portable-windows-x64.zip` | **Windows** | 自包含（含运行期 DLL）|
| `SHA256SUMS` | — | 校验文件 |

下载后建议校验：
```bash
sha256sum -c SHA256SUMS   # Linux
# Windows: Get-FileHash kan-portable-windows-x64.zip
```

### 🐧 Linux 快速开始

```bash
tar xzf kan-portable-linux-x64-sm86-g8.tar.gz   # 3090/3080 用 sm86-g8；其他卡用 generic
cd kan-portable-linux-x64
./run.sh --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet 你的PRL地址.矿工名 \
  --batch 1000 --cfg real --tc
```

`run.sh` 会自动：检测 NVIDIA 驱动、为 sm_86 包设置 `TC_PERSIST=0`、并在 pool 模式下断线自动重启。

### 🪟 Windows 快速开始

解压 `kan-portable-windows-x64.zip` 到一个文件夹，在**该文件夹打开命令提示符**（cmd），运行：

```bat
run.bat --algo pearl --pool stratum+tcp://prl.kryptex.network:7048 --wallet 你的PRL地址.矿工名 --batch 1000 --cfg real --tc
```

或用 PowerShell：
```powershell
.\run.ps1 --algo pearl --pool stratum+tcp://prl.kryptex.network:7048 --wallet 你的PRL地址.矿工名 --batch 1000 --cfg real --tc
```

Windows 包已捆绑 `libssl-3-x64.dll` / `libcrypto-3-x64.dll` / `cudart64_12.dll` / `vcruntime140.dll` 等，**只需安装 NVIDIA 驱动即可运行**。

### 一键安装脚本（Linux）

```bash
VERSION=v1.2.22 ./install_kan.sh
```

`install_kan.sh` 会根据 `nvidia-smi` 自动选择合适的包（sm_86 → sm86-g8，其他 → generic），校验 SHA256 后安装到 `/opt/kan`（可用 `DEST=` 改路径）。

---

## 矿池挖矿

### 钱包地址格式

```
--wallet PRL地址.矿工名
```

- **PRL 地址**：bech32 格式，以 `prl1` 开头
- **矿工名**：可选，用 `.` 分隔（如 `prl1abc...xyz.rig01`）
- 也可分开：`--wallet prl1abc...xyz --worker rig01`

### 矿池地址

Kryptex / LuckyPool Pearl 矿池（实测可用）：

| 协议 | 地址 | 说明 |
|---|---|---|
| **明文 TCP（推荐）** | `stratum+tcp://prl.kryptex.network:7048` | 性能最佳 |
| TLS 加密 | `stratum+ssl://prl.kryptex.network:8048` | 加密传输 |

可重复 `--pool` 实现故障转移：
```bash
./run.sh --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --pool stratum+ssl://backup-pool.example:8048 \
  --wallet prl1abc...xyz.rig01 --batch 1000 --cfg real --tc
```

### 后台运行（Linux）

```bash
nohup ./run.sh --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet prl1abc...xyz.rig01 --batch 1000 --cfg real --tc \
  > miner.log 2>&1 &
tail -f miner.log
```

`run.sh` 的 `KAN_RESTART=auto`（默认）会在进程退出/断线后自动重启；想一次性调试可 `KAN_RESTART=0 ./run.sh ...`。

---

## Solo 挖矿

Solo 模式连接本地 `pearld` 节点，并将 PlainProof 转换为 ZK 证明提交。需要可选的 `zkprove` 工具（见[从源码编译](#从源码编译)）。

```bash
./build/kan --solo \
  --node 127.0.0.1:44107 \
  --rpcuser your_rpc_user \
  --rpcpass your_rpc_pass \
  --addr prl1qyour_p2tr_address \
  --zkprove /path/to/zkprove
```

> Solo 挖矿的预期收益与矿池相同（仅方差不同）。Pearl 网络难度下，单卡 solo 出块周期极长，**绝大多数矿工应使用矿池模式**。

---

## 命令行参数完整参考

### 矿池模式参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--algo pearl` | 指定算法（当前仅支持 pearl） | 必填 |
| `--pool URL` | 矿池地址。格式 `stratum+tcp://host:port` 或 `stratum+ssl://host:port`。**可重复指定**实现主/备故障转移（按顺序尝试，主池不可达自动切下一个） | 必填 |
| `--wallet ADDR[.WORKER]` | PRL 钱包地址，可带 `.矿工名` 后缀 | 必填 |
| `--worker NAME` | 矿工名（也可合并在 wallet 中用 `.` 分隔） | `pm` |
| `--devices LIST` | 选择物理 GPU 子集（如 `0,1,3`）；不设置时自动使用所有 GPU。与 `CUDA_VISIBLE_DEVICES` 互斥 | 全部 GPU |
| `--batch N` | 每轮最大 draw 数（新 job 到达前最多搜索 N 次） | `1000` |
| `--api-port N` | 开启 HTTP/JSON 监控端口（每卡 + 聚合算力、accepted/rejected、NVML 温度/转速/功耗），对接 HiveOS/mmpOS/curl | 关闭 |
| `--breakdown` | 打印每 draw 的详细计时（GPU 总时长 / kernel 时长） | 关闭 |
| `--agent STRING` | 自定义 stratum agent 标识 | `Kan/<version>` |

### Solo 模式参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--solo` | 启用 solo 模式 | — |
| `--node HOST:PORT` | pearld 节点 RPC 地址 | `127.0.0.1:44107` |
| `--rpcuser USER` | RPC 用户名 | 必填 |
| `--rpcpass PASS` | RPC 密码 | 必填 |
| `--addr ADDR` | P2TR 接收地址 | 必填 |
| `--zkprove PATH` | zkprove 工具路径（PlainProof → ZK 证明） | `./zkprove` |

### 通用参数（矿池 + Solo）

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--cfg real` | 使用真实网络配置 (m=n=131072, k=4096, rank=256) | `real` |
| `--tc` | 强制使用 Tensor-Core 搜索内核（强烈推荐） | 开启 |
| `--version` | 显示版本号 | — |
| `--help` | 显示帮助 | — |

### 钱包地址说明

- Pearl 地址为 bech32 编码，以 `prl1` 开头（类似 Bitcoin 的 bc1）
- 从 Pearl 官方钱包或交易所获取
- **矿工名**是可选标签，便于在矿池后台区分不同机器/不同卡，不影响收款地址

---

## 环境变量与运行时开关

| 变量 | 作用 | 默认 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | 限制可见 GPU（与 `--devices` 互斥，设置后禁用自动 fanout） | 不设置 |
| `TC_PERSIST` | 持久化内核启动模式（`1`=开启，`0`=关闭）。sm_86 包默认 `0`（实测更快），其他架构默认 `1` | 卡型相关 |
| `TC_TIMING` | `1` 时打印 CUTLASS 内核细分计时（prep+gather / search / 总吞吐） | `0` |
| `KAN_RESTART` | `run.sh` 的重启策略：`auto`=pool 模式断线自动重启，`0`=一次性 | `auto` |
| `KAN_RESTART_DELAY` | 重启间隔秒数 | `15` |
| `KAN_SHOW_BUILD_INFO` | `0` 时关闭 `run.sh` 启动横幅 | `1` |
| `KAN_VERSION` | （打包时注入）版本字符串 | 来自 git describe |

示例：
```bash
# 关闭持久化模式调试
TC_PERSIST=0 ./run.sh --algo pearl --pool ... --wallet ...

# 打开内核细分计时
TC_TIMING=1 ./run.sh --algo pearl --pool ... --wallet ... --breakdown
```

---

## 单机多卡（多 GPU）

pool 模式默认自动使用所有检测到的 GPU：

```text
未设置 CUDA_VISIBLE_DEVICES：
  父进程作为 supervisor，为每张物理 GPU fork 一个 per-GPU isolated lane process；
  每个 lane 各自建立一条 stratum 连接，以同一个 worker 名认证
  —— 矿池按 worker 聚合整机算力。

--devices 0,1,3：
  只在指定 GPU 子集 fanout。

CUDA_VISIBLE_DEVICES=0（外部设置）：
  miner 尊重该变量，只在它暴露的 GPU 上跑单个 lane，fanout 被禁用。
```

```bash
# 所有 GPU（默认）
./run.sh --algo pearl --pool ... --wallet addr.rig01

# 只用 GPU 0 和 2
./run.sh --algo pearl --devices 0,2 --pool ... --wallet addr.rig01

# 用环境变量固定单卡（禁用 fanout）
CUDA_VISIBLE_DEVICES=1 ./run.sh --algo pearl --pool ... --wallet addr.rig01
```

注意：
- `--devices` 与 `CUDA_VISIBLE_DEVICES` 互斥，同时设置会报错退出。
- 任意 lane 异常退出时，supervisor 停止其余 lane 并整体退出，交给 `run.sh` 自动重启（避免静默降级）。
- Windows v1 为单卡模式（多卡 supervisor 用 fork，Windows 暂未实现 CreateProcess 等价）。

---

## 生产部署

### systemd 服务（Linux，便携包内含模板）

```bash
cd /opt/kan
sudo -E ./install_service.sh
sudo editor /opt/kan/kan.env    # 设置 KAN_WALLET / KAN_WORKER / KAN_DEVICES
sudo systemctl enable --now kan
journalctl -u kan -f
```

`kan.env` 示例：
```ini
KAN_POOL=stratum+tcp://prl.kryptex.network:7048
KAN_WALLET=prl1abc...xyz
KAN_WORKER=rig01
KAN_DEVICES=0,1         # 可选；不设置则自动全卡 fanout
```

包内还含 `kan.logrotate`（日志轮转）和 `kan.service`（systemd 单元）。

### HiveOS / mmpOS

用 `--api-port 4068` 开启 HTTP 监控，矿机面板读取 `http://127.0.0.1:4068/summary` 即可拿到算力 / 份额 / 温度。或将便携包作为 Custom Miner 配置，Installation URL 指向 Release 资产。

---

## 运行时交互与日志格式

### 运行时按键

| 按键 | 功能 |
|------|------|
| `s` | 立即打印统计表（否则每 120 秒自动一次） |
| `q` | 优雅退出 |

### 统计表（首次 15 秒后显示）

```
-----prl1abc...xyz.rig01---------------stratum+tcp://prl.kryptex.network:7048-----
 DEVICE MODEL              HASHRATE  TEMP  FAN POWER      EFFIC       A    R  LAST
----------------------------------------------------------------------------------------
 GPU #0 RTX 4090           248.60 TH/s    65C   80%  449W  553.7 GH/W     300    3     -
----------------------------------------------------------------------------------------
 10s                       251.81 TH/s                0W           A: 300
 60s                       248.60 TH/s                           R: 3
 15m                       248.60 TH/s                           S: 0
[0 days 03:02:17]-------------------------------------[99.0% accept - ver. v1.2.22]
```

各列含义：
- **HASHRATE**：60 秒窗口平均算力（TH/s）
- **TEMP / FAN / POWER**：NVML 读取的温度 / 风扇转速 / 功耗
- **EFFIC**：能效比（GH/W）—— 越高越省电
- **A / R**：已接受 / 已拒绝的 share 数
- **LAST**：距上次接受 share 的时间
- **10s / 60s / 15m**：三个独立时间窗口的算力
- 末行 `[99.0% accept - ver. v1.2.22]`：接受率与版本

### 日志事件

```
23:10:26  info   stratum        authorize: ok wallet=prl1abc...xyz agent=Kan/v1.2.22
23:10:26  info   stratum        new job id=3f8888f0_2097152 height=78666 diff=2097152 seq=1
23:10:26  info   stratum        async share submit worker active gpu=0
23:25:21  info   GPU #0         share accepted submit_wait=0.797s
```

---

## 性能参考

### 官方基准对照（PearlHash 算法）

数据来自 Kryptex 官方设备页（`pool.kryptex.com/zh-cn/device/...`）与本仓库实测：

| GPU | 官方最大算力 | 官方功耗 | 本仓库实测（便携版）|
|-----|------|------|------|
| RTX 4090 | **255 TH/s** | 450W | **248 TH/s**（97%，450W）|
| RTX 3090 | 135 TH/s | 320W | 98-105 TH/s（受限功耗墙的租用卡；健康可调优卡接近 135）|
| RTX 3080 Ti | ~120 TH/s | 350W | 98-106 TH/s（sm86-g8 包）|

> 3090 的 135 TH/s 官方值是**调优后**（降压 + 显存超频）的上限。锁功耗墙 / 无调频权限的卡（如某些云容器）实际只能跑到 98-105。本仓库内核在 4090 上达到官方 97%，证明内核效率已接近上限。

### 内核优化历程

| 阶段 | 技术要点 |
|------|---------|
| 手写 WMMA | dp4a → WMMA 基础内核 |
| 手写 IMMA | mma.sync.m16n8k32 + register-only fold |
| CUTLASS 融合 | 3-stage cp.async 流水线 + FoldMmaMultistage |
| GPU-resident pipeline | RNG + blake3 树哈希 + 噪声生成全部上 GPU（CPU 1490ms → 10ms）|
| async overlap | search(N) 与 prep(N+1) 重叠执行 |
| 异步 share 提交 | submit_wait 不阻塞下一轮 mining |

可追溯基准记录见 [`bench/results/`](bench/results/)。

---

## 从源码编译

> 绝大多数用户**不需要**编译——直接用 [Release](https://github.com/tvvshow/kan-mine/releases/latest) 的便携包即可。以下仅适合开发者/自编译。

### Linux

依赖：CUDA Toolkit 12.x、GCC 9+、OpenSSL、CUTLASS 3.5.1（header-only）。

```bash
# 1. 依赖
sudo apt install -y libssl-dev g++ make git curl

# 2. CUTLASS（header-only，必需，否则回退低速 WMMA）
git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass

# 3. 克隆
git clone https://github.com/tvvshow/kan-mine.git kan
cd kan

# 4. 编译
export CUTLASS_HOME=~/cutlass
./build.sh
```

产物：`build/kan`（矿池+solo 二进制）、`build/plainproof_gen`（proof CLI）。

### Windows

依赖：Visual Studio 2022（MSVC）、CUDA Toolkit 12.5、vcpkg（OpenSSL）。

```powershell
git clone https://github.com/tvvshow/kan-mine.git kan
cd kan
# 在 Developer Command Prompt 或设置好 MSVC 环境后：
vcpkg install openssl:x64-windows
.\build_windows.ps1
```

产物：`build\kan.exe`、`build\plainproof_gen.exe`、`build\pearl-miner.exe`。

### 编译选项

| 环境变量 | 说明 | 默认 |
|---------|------|------|
| `CUTLASS_HOME` | CUTLASS 源码路径 | `~/cutlass` |
| `ARCH` | nvcc 目标架构（如 `sm_86`） | 自动从 nvidia-smi 检测 |
| `KERNEL` | `cutlass`（默认高速）或 `wmma`（Turing/低速回退） | cutlass（CUTLASS 存在时）|

---

## 项目结构

```
kan-mine/
├── src/
│   ├── miner_main.cpp      # 主程序（矿池 + Solo + 日志 + NVML + 多卡 supervisor）
│   ├── plainproof_gen.cpp  # PlainProof 生成器（挖矿核心循环）
│   ├── tc_cutlass_v2.cu    # CUTLASS 融合 Tensor-Core 搜索内核（Ampere+）
│   ├── tc_block.cu         # WMMA 内核（Turing / CUTLASS 不可用时回退）
│   ├── gpu_prep.cu         # GPU 端 RNG + blake3 + 噪声生成
│   ├── prover.h            # MineParams / MineResult API
│   └── platform.h          # 跨平台 shim（Linux/Windows + MSVC 兼容）
├── blake3/                 # BLAKE3 哈希库（vendored，含 SIMD 加速）
├── oracle/                 # 离线测试向量（golden header/config）
├── build.sh                # Linux 一键编译
├── build_windows.ps1       # Windows 编译（MSVC + nvcc）
├── package_portable.sh     # Linux 便携包打包
├── package_windows.ps1     # Windows 便携包打包（DLL + zip）
├── install_kan.sh          # GPU 自检测一键安装
├── install_service.sh      # systemd 服务模板
├── .github/workflows/      # GitHub Actions（linux.yml + windows.yml）
├── .cnb.yml                # cnb.cool 备用 CI
├── GPU_PROFILES.md         # GPU profile 权威表
├── CHANGELOG.md            # 变更记录
├── LICENSE                 # 许可条款（必读）
├── README.md               # 本文件（中文）
└── README.en.md            # 英文版 README（English）
```

---

## 常见问题

### 便携包无法运行："libnvidia-ml.so.1 not found" / "cudart64_12.dll not found"
- **Linux**：需安装 NVIDIA 驱动（`libnvidia-ml.so.1` 随驱动提供）。CUDA 运行时已捆绑在包内，无需装 CUDA Toolkit。
- **Windows**：需安装 NVIDIA 驱动；其余 DLL（OpenSSL/CUDA runtime/MSVC CRT）已捆绑在 zip 内。

### NVML 温度/风扇/功耗显示为 `--`
非致命。确保 `libnvidia-ml.so.1`（Linux）或 `nvml.dll`（Windows）存在，它们随 NVIDIA 驱动安装。

### 编译报错 "CUTLASS not found"
```bash
git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass
./build.sh
```
不装 CUTLASS 也能编译，但回退到 WMMA 内核，算力仅 ~30 TH/s。

### 矿池拒绝所有 share
1. 检查钱包地址格式：PRL bech32，以 `prl1` 开头
2. 确认矿池可达：`telnet prl.kryptex.network 7048`
3. TLS 失败时改用明文 7048
4. 查看日志 `share rejected` 原因

### 算力偏低
1. `nvidia-smi` 查 GPU 时钟是否正常（核心 1500MHz+），撞功耗墙会降频
2. 杀掉残留 profiler：`sudo pkill ncu`（ncu 会限制 SM 频率）
3. 3090/3080 用户确认用的是 `sm86-g8` 调优包而非 generic
4. 用 `TC_TIMING=1 --breakdown` 看每 draw 计时定位瓶颈

### Windows 上 `cl.exe is not recognized`
在 Developer Command Prompt for VS 2022 里编译，或 CI 已用 `ilammy/msvc-dev-cmd` 自动处理。

---

## 许可与出处

本项目按 [LICENSE](LICENSE) 文件条款发布。**核心要点：**

| ✅ 允许 | ❌ 禁止 |
|---|---|
| 个人非商业挖矿使用 | 商业使用（收费托管 / 云算力 / 收费套件）|
| 审阅、学习、研究源码 | 加入任何抽水 / devfee / 算力分流 |
| 学习性 fork 与修改 | 闭源魔改、删除 LICENSE、弱化禁商用条款 |
| 在矿池为本人地址挖矿 | 篡改 wallet/worker 窃取他人算力 |

**二次开发必须：**
1. 在用户可见处（README / `--version` / 启动横幅）标注 `Based on Kan by tvvshow/kan-mine (https://github.com/tvvshow/kan-mine)`
2. 保留 [LICENSE](LICENSE) 全文
3. 以相同条款（禁商用 + 禁抽水 + 标出处）发布
4. 公开衍生作品的完整源代码

违反任一条款 = 自动失去授权，且原作者保留追究权利。

---

## 致谢

- **Pearl 区块链**：https://github.com/pearl-network/pearl
- **NVIDIA CUTLASS**：https://github.com/NVIDIA/cutlass
- **BLAKE3**：https://github.com/BLAKE3-team/BLAKE3

---

*RTX 4090: 248 TH/s（官方 255 的 97%）· RTX 3090: 98-135 TH/s（取决于功耗墙/调优）· 100% 开源 · 零抽水 · 禁商用*
