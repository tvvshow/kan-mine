# Kan — High-Performance Pearl (PRL) PoUW GPU Miner

**English** | [中文](README.md)

[![Release](https://img.shields.io/badge/release-v1.2.22-blue)](https://github.com/tvvshow/kan-mine/releases/tag/v1.2.22)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-green)](#download--quick-start)
[![License](https://img.shields.io/badge/license-NonCommercial%20%7C%20NoFee%20%7C%20Attribution-important)](LICENSE)

> **🔴 Important — please read**
>
> This project is **100% open-source, with ZERO dev fee / ZERO hashrate skimming**.
>
> - Every valid share you submit to the pool is credited to **your wallet** — the source contains **no** hashrate diversion, share redirection, or hidden devfee. Audit it freely.
> - **Commercial use is prohibited** (no paid hosting / cloud-hashrate / paid suites).
> - **Derivative works must NOT add any skimming mechanism** (no devfee, no share diversion, no wallet/worker tampering).
> - **Derivatives must attribute the source** (keep the `Based on Kan by tvvshow/kan-mine` notice) and retain the [LICENSE](LICENSE).
>
> Full legal terms are in [LICENSE](LICENSE). Violating any clause = automatic termination of your license.

---

## Table of Contents

- [Features](#features)
- [Hardware Requirements & GPU Profiles](#hardware-requirements--gpu-profiles)
- [Download & Quick Start](#download--quick-start)
- [Pool Mining](#pool-mining)
- [Solo Mining](#solo-mining)
- [Full Command-Line Reference](#full-command-line-reference)
- [Environment Variables & Runtime Switches](#environment-variables--runtime-switches)
- [Multi-GPU (Single Machine)](#multi-gpu-single-machine)
- [Production Deployment (systemd / HiveOS)](#production-deployment)
- [Runtime Interaction & Log Format](#runtime-interaction--log-format)
- [Performance](#performance)
- [Building from Source](#building-from-source)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License & Attribution](#license--attribution)

---

## Features

- **Pool mining** (Stratum V1 — LuckyPool / Kryptex / etc.) and **solo mining** (pearld RPC)
- **Cross-platform**: Linux x86-64 + **Windows x64**, both with ready-to-run portable packages
- **High-performance CUTLASS int8 Tensor-Core kernel** (see [Performance](#performance))
- **Fully GPU-resident pipeline**: RNG + blake3 tree hash + noise generation + jackpot search all on GPU
- **Multi-GPU**: pool mode auto-fanout to all detected GPUs (one isolated lane process per GPU, shared worker name)
- **Real-time hashrate**: first table at 15s, 500ms sampling; 10s/60s/15m windows
- **NVML hardware monitoring**: temperature / fan / power / efficiency
- **Async share submission**: network `submit_wait` never blocks the next mining attempt
- **Stale-proof early-abort**: skip un-submittable stale proofs when a new job arrives
- **Auto-restart on disconnect**: `run.sh` / `run.bat` ship with a pool-mode restart loop
- **Multi-pool failover**: repeat `--pool` for primary/backup (auto-failover on unreachable primary)
- **HTTP/JSON monitoring API**: for HiveOS / mmpOS / curl (`--api-port`)
- **Zero dev fee / zero hashrate skimming**

---

## Hardware Requirements & GPU Profiles

| Item | Requirement |
|------|------|
| **GPU** | NVIDIA Tensor-Core GPU (Ampere / Turing or newer recommended — see table below) |
| **VRAM** | ≥ 4 GB (uses ~2 GB in practice) |
| **Driver** | A recent NVIDIA driver (CUDA 12.x compatible) |
| **OS** | Linux x86-64 (glibc ≥ 2.35) or Windows 10/11 x64 |
| **Runtime deps** | **Only the NVIDIA driver** — portable packages bundle the CUDA runtime, OpenSSL, and MSVC runtime. No CUDA Toolkit / CUTLASS / compiler needed on the target machine. |

### GPU profiles & recommended package

Full authoritative table in [`GPU_PROFILES.md`](GPU_PROFILES.md). Summary:

| GPU / Arch | Recommended Package | Status |
|-----|------|------|
| RTX 20 series / Titan RTX / `sm_75` | `kan-portable-linux-x64-sm75.tar.gz` | Dedicated WMMA package, POSTCHECK ok=1 (Candidate) |
| RTX 3080 / 3090 / 3080 Ti / `sm_86` | `kan-portable-linux-x64-sm86-g8.tar.gz` | **tuned production package**, GROUPM=8/KSTAGES=3 |
| RTX 4090 / L40 / `sm_89` | `kan-portable-linux-x64.tar.gz` or Windows zip | generic multi-arch fatbin |
| A100 / `sm_80`, H100 / `sm_90` | `kan-portable-linux-x64.tar.gz` | generic compatible |
| RTX 50 series / `sm_120` | `kan-portable-linux-x64.tar.gz` | CUDA 12 PTX JIT fallback; native tuned needs CUDA 13 |
| Windows, any of the above | `kan-portable-windows-x64.zip` | sm_75 + sm_86 WMMA path (v1) |

> **Volta / `sm_70` (V100/V100S) is not supported** by the current release — the kernel targets the `sm_75+` style int8 Tensor-Core path.

---

## Download & Quick Start

### 🔽 Download

Grab the portable package for your platform from the GitHub Release (no compilation needed):

👉 **https://github.com/tvvshow/kan-mine/releases/latest**

| File | Platform | Notes |
|---|---|---|
| `kan-portable-linux-x64.tar.gz` | Linux | **generic compatibility package** (Ampere/Ada/Hopper + Blackwell PTX) |
| `kan-portable-linux-x64-sm86-g8.tar.gz` | Linux | 3080/3090 tuned |
| `kan-portable-linux-x64-sm75.tar.gz` | Linux | Turing (2080Ti) dedicated |
| `kan-portable-windows-x64.zip` | **Windows** | self-contained (DLLs bundled) |
| `SHA256SUMS` | — | checksums |

Verify after download:
```bash
sha256sum -c SHA256SUMS        # Linux
# Windows: Get-FileHash kan-portable-windows-x64.zip
```

### 🐧 Linux Quick Start

```bash
tar xzf kan-portable-linux-x64-sm86-g8.tar.gz   # 3090/3080 use sm86-g8; other cards use generic
cd kan-portable-linux-x64
./run.sh --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet YOUR_PRL_ADDRESS.WORKER \
  --batch 1000 --cfg real --tc
```

`run.sh` auto-detects the NVIDIA driver, sets `TC_PERSIST=0` for sm_86 packages, and restarts on disconnect in pool mode.

### 🪟 Windows Quick Start

Unzip `kan-portable-windows-x64.zip` into a folder, open a **Command Prompt in that folder**, and run:

```bat
run.bat --algo pearl --pool stratum+tcp://prl.kryptex.network:7048 --wallet YOUR_PRL_ADDRESS.WORKER --batch 1000 --cfg real --tc
```

or in PowerShell:
```powershell
.\run.ps1 --algo pearl --pool stratum+tcp://prl.kryptex.network:7048 --wallet YOUR_PRL_ADDRESS.WORKER --batch 1000 --cfg real --tc
```

The Windows zip bundles `libssl-3-x64.dll` / `libcrypto-3-x64.dll` / `cudart64_12.dll` / `vcruntime140.dll` — **only the NVIDIA driver is required on the host**.

### One-line installer (Linux)

```bash
VERSION=v1.2.22 ./install_kan.sh
```

`install_kan.sh` auto-selects the right package via `nvidia-smi` (sm_86 → sm86-g8, others → generic), verifies SHA256, and installs to `/opt/kan` (override with `DEST=`).

---

## Pool Mining

### Wallet address format

```
--wallet PRL_ADDRESS.WORKER
```

- **PRL address**: bech32, starting with `prl1`
- **Worker name**: optional, separated by `.` (e.g. `prl1abc...xyz.rig01`)
- Or specify separately: `--wallet prl1abc...xyz --worker rig01`

### Pool endpoints

Kryptex / LuckyPool Pearl pools (tested working):

| Protocol | Endpoint | Notes |
|---|---|---|
| **Plain TCP (recommended)** | `stratum+tcp://prl.kryptex.network:7048` | best performance |
| TLS encrypted | `stratum+ssl://prl.kryptex.network:8048` | encrypted transport |

Repeat `--pool` for failover:
```bash
./run.sh --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --pool stratum+ssl://backup-pool.example:8048 \
  --wallet prl1abc...xyz.rig01 --batch 1000 --cfg real --tc
```

### Background run (Linux)

```bash
nohup ./run.sh --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet prl1abc...xyz.rig01 --batch 1000 --cfg real --tc \
  > miner.log 2>&1 &
tail -f miner.log
```

`run.sh`'s `KAN_RESTART=auto` (default) restarts on exit/disconnect; use `KAN_RESTART=0 ./run.sh ...` for a one-shot debug run.

---

## Solo Mining

Solo mode connects to a local `pearld` node and converts the PlainProof into a ZK proof for submission. Requires the optional `zkprove` tool (see [Building from Source](#building-from-source)).

```bash
./build/kan --solo \
  --node 127.0.0.1:44107 \
  --rpcuser your_rpc_user \
  --rpcpass your_rpc_pass \
  --addr prl1qyour_p2tr_address \
  --zkprove /path/to/zkprove
```

> Solo mining has the same expected yield as pool mining (only the variance differs). At Pearl's current network difficulty, a single card's time-to-block solo is extremely long — **the vast majority of miners should use pool mode**.

---

## Full Command-Line Reference

### Pool-mode options

| Option | Description | Default |
|------|------|-------|
| `--algo pearl` | Algorithm (only `pearl` currently supported) | required |
| `--pool URL` | Pool URL, `stratum+tcp://host:port` or `stratum+ssl://host:port`. **Repeatable** for primary/backup failover (tried in order; auto-advances when unreachable) | required |
| `--wallet ADDR[.WORKER]` | PRL wallet address, optionally with `.WORKER` suffix | required |
| `--worker NAME` | Worker name (or embed in wallet with `.`) | `pm` |
| `--devices LIST` | Physical GPU subset (e.g. `0,1,3`); default uses all GPUs. Mutually exclusive with `CUDA_VISIBLE_DEVICES` | all GPUs |
| `--batch N` | Max draws per attempt before re-checking for a new job | `1000` |
| `--api-port N` | HTTP/JSON stats port (per-GPU + aggregated hashrate, accepted/rejected, NVML temp/fan/power) for HiveOS/mmpOS/curl | off |
| `--breakdown` | Print per-draw timing (GPU total / kernel time) | off |
| `--agent STRING` | Custom stratum agent string | `Kan/<version>` |

### Solo-mode options

| Option | Description | Default |
|------|------|-------|
| `--solo` | Enable solo mode | — |
| `--node HOST:PORT` | pearld node RPC endpoint | `127.0.0.1:44107` |
| `--rpcuser USER` | RPC username | required |
| `--rpcpass PASS` | RPC password | required |
| `--addr ADDR` | P2TR payout address | required |
| `--zkprove PATH` | Path to the zkprove tool (PlainProof → ZK proof) | `./zkprove` |

### Universal options (pool + solo)

| Option | Description | Default |
|------|------|-------|
| `--cfg real` | Use the real network config (m=n=131072, k=4096, rank=256) | `real` |
| `--tc` | Force the Tensor-Core search kernel (strongly recommended) | on |
| `--version` | Print version | — |
| `--help` | Show help | — |

### About wallet addresses

- Pearl addresses are bech32-encoded, starting with `prl1` (similar to Bitcoin's bc1)
- Obtain one from the official Pearl wallet or an exchange
- The **worker name** is an optional tag to distinguish machines/cards in the pool dashboard; it does not affect the payout address

---

## Environment Variables & Runtime Switches

| Variable | Effect | Default |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs (mutually exclusive with `--devices`; disables auto-fanout when set) | unset |
| `TC_PERSIST` | Persistent kernel-launch mode (`1`=on, `0`=off). sm_86 packages default to `0` (measured faster); others default to `1` | card-dependent |
| `TC_TIMING` | `1` prints CUTLASS kernel sub-timings (prep+gather / search / total throughput) | `0` |
| `KAN_RESTART` | `run.sh` restart policy: `auto`=restart on disconnect in pool mode, `0`=one-shot | `auto` |
| `KAN_RESTART_DELAY` | Seconds between restarts | `15` |
| `KAN_SHOW_BUILD_INFO` | `0` silences the `run.sh` startup banner | `1` |
| `KAN_VERSION` | (injected at packaging) version string | from git describe |

Examples:
```bash
# Disable persistent mode for debugging
TC_PERSIST=0 ./run.sh --algo pearl --pool ... --wallet ...

# Enable kernel sub-timings
TC_TIMING=1 ./run.sh --algo pearl --pool ... --wallet ... --breakdown
```

---

## Multi-GPU (Single Machine)

Pool mode auto-uses all detected GPUs by default:

```text
Without CUDA_VISIBLE_DEVICES set:
  The parent process acts as a supervisor, forking one per-GPU isolated lane process
  per physical GPU. Each lane opens its own stratum connection and authenticates
  with the same worker name — the pool aggregates the whole machine under that worker.

--devices 0,1,3:
  Fan out only on the specified GPU subset.

CUDA_VISIBLE_DEVICES=0 (set externally):
  The miner respects it and runs a single lane on whatever it exposes; auto-fanout
  is disabled.
```

```bash
# All GPUs (default)
./run.sh --algo pearl --pool ... --wallet addr.rig01

# Only GPUs 0 and 2
./run.sh --algo pearl --devices 0,2 --pool ... --wallet addr.rig01

# Pin to one card via env (disables fanout)
CUDA_VISIBLE_DEVICES=1 ./run.sh --algo pearl --pool ... --wallet addr.rig01
```

Notes:
- `--devices` and `CUDA_VISIBLE_DEVICES` are mutually exclusive; setting both exits with an error.
- If any lane exits abnormally, the supervisor stops the rest and exits as a whole, letting `run.sh` auto-restart (no silent degraded operation).
- Windows v1 is single-GPU (the multi-GPU supervisor uses fork; the Windows CreateProcess equivalent is not yet implemented).

---

## Production Deployment

### systemd service (Linux; template included in the portable package)

```bash
cd /opt/kan
sudo -E ./install_service.sh
sudo editor /opt/kan/kan.env    # set KAN_WALLET / KAN_WORKER / KAN_DEVICES
sudo systemctl enable --now kan
journalctl -u kan -f
```

Example `kan.env`:
```ini
KAN_POOL=stratum+tcp://prl.kryptex.network:7048
KAN_WALLET=prl1abc...xyz
KAN_WORKER=rig01
KAN_DEVICES=0,1         # optional; omit to auto-fanout all GPUs
```

The package also ships `kan.logrotate` (log rotation) and `kan.service` (systemd unit).

### HiveOS / mmpOS

Open HTTP monitoring with `--api-port 4068`; the dashboard reads `http://127.0.0.1:4068/summary` for hashrate / shares / temperatures. Or use the portable package as a Custom Miner with the Installation URL pointing at the Release asset.

---

## Runtime Interaction & Log Format

### Runtime keys

| Key | Action |
|------|------|
| `s` | Print the stats table immediately (otherwise every 120s) |
| `q` | Graceful exit |

### Stats table (first shown after 15s)

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

Column meanings:
- **HASHRATE**: 60-second window average (TH/s)
- **TEMP / FAN / POWER**: NVML temperature / fan speed / power draw
- **EFFIC**: efficiency (GH/W) — higher is more power-efficient
- **A / R**: accepted / rejected share counts
- **LAST**: time since the last accepted share
- **10s / 60s / 15m**: independent windowed hashrates
- Footer `[99.0% accept - ver. v1.2.22]`: accept rate and version

### Log events

```
23:10:26  info   stratum        authorize: ok wallet=prl1abc...xyz agent=Kan/v1.2.22
23:10:26  info   stratum        new job id=3f8888f0_2097152 height=78666 diff=2097152 seq=1
23:10:26  info   stratum        async share submit worker active gpu=0
23:25:21  info   GPU #0         share accepted submit_wait=0.797s
```

---

## Performance

### Official benchmark comparison (PearlHash)

Data from the Kryptex official device pages (`pool.kryptex.com/zh-cn/device/...`) and this project's measurements:

| GPU | Official Max Hashrate | Official Power | Our Measurement (portable) |
|-----|------|------|------|
| RTX 4090 | **255 TH/s** | 450W | **248 TH/s** (97%, 450W) |
| RTX 3090 | 135 TH/s | 320W | 98–105 TH/s (on a power-locked rental card; a healthy tunable card approaches 135) |
| RTX 3080 Ti | ~120 TH/s | 350W | 98–106 TH/s (sm86-g8 package) |

> The 3090's official 135 TH/s is the **tuned** maximum (undervolt + memory OC). Cards with a locked power cap or no tuning permissions (e.g. some cloud containers) reach only 98–105. Our kernel reaches 97% of official on the 4090, confirming the kernel efficiency is near the ceiling.

### Kernel optimization history

| Stage | Key technique |
|------|---------|
| Hand-written WMMA | dp4a → basic WMMA kernel |
| Hand-written IMMA | mma.sync.m16n8k32 + register-only fold |
| CUTLASS fused | 3-stage cp.async pipeline + FoldMmaMultistage |
| GPU-resident pipeline | RNG + blake3 tree hash + noise generation fully on GPU (CPU 1490ms → 10ms) |
| Async overlap | search(N) overlapped with prep(N+1) |
| Async share submit | submit_wait never blocks the next mining attempt |

Traceable benchmark records in [`bench/results/`](bench/results/).

---

## Building from Source

> Most users do **not** need to build — use the [Release](https://github.com/tvvshow/kan-mine/releases/latest) portable packages. The following is for developers / self-builders.

### Linux

Dependencies: CUDA Toolkit 12.x, GCC 9+, OpenSSL, CUTLASS 3.5.1 (header-only).

```bash
# 1. Deps
sudo apt install -y libssl-dev g++ make git curl

# 2. CUTLASS (header-only, required — otherwise falls back to the slow WMMA kernel)
git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass

# 3. Clone
git clone https://github.com/tvvshow/kan-mine.git kan
cd kan

# 4. Build
export CUTLASS_HOME=~/cutlass
./build.sh
```

Outputs: `build/kan` (pool+solo binary), `build/plainproof_gen` (proof CLI).

### Windows

Dependencies: Visual Studio 2022 (MSVC), CUDA Toolkit 12.5, vcpkg (OpenSSL).

```powershell
git clone https://github.com/tvvshow/kan-mine.git kan
cd kan
# From a Developer Command Prompt (or with the MSVC env set up):
vcpkg install openssl:x64-windows
.\build_windows.ps1
```

Outputs: `build\kan.exe`, `build\plainproof_gen.exe`, `build\pearl-miner.exe`.

### Build options

| Env var | Description | Default |
|---------|------|------|
| `CUTLASS_HOME` | CUTLASS source path | `~/cutlass` |
| `ARCH` | nvcc target arch (e.g. `sm_86`) | auto-detected from nvidia-smi |
| `KERNEL` | `cutlass` (default fast) or `wmma` (Turing / slow fallback) | cutlass (when CUTLASS is present) |

---

## Project Structure

```
kan-mine/
├── src/
│   ├── miner_main.cpp      # main binary (pool + solo + logging + NVML + multi-GPU supervisor)
│   ├── plainproof_gen.cpp  # PlainProof generator (core mining loop)
│   ├── tc_cutlass_v2.cu    # CUTLASS fused Tensor-Core search kernel (Ampere+)
│   ├── tc_block.cu         # WMMA kernel (Turing / fallback when CUTLASS unavailable)
│   ├── gpu_prep.cu         # GPU-side RNG + blake3 + noise generation
│   ├── prover.h            # MineParams / MineResult API
│   └── platform.h          # cross-platform shim (Linux/Windows + MSVC compat)
├── blake3/                 # BLAKE3 hash library (vendored, SIMD-accelerated)
├── oracle/                 # offline test vectors (golden header/config)
├── build.sh                # one-line Linux build
├── build_windows.ps1       # Windows build (MSVC + nvcc)
├── package_portable.sh     # Linux portable packager
├── package_windows.ps1     # Windows portable packager (DLLs + zip)
├── install_kan.sh          # GPU-autodetect one-line installer
├── install_service.sh      # systemd service template
├── .github/workflows/      # GitHub Actions (linux.yml + windows.yml)
├── .cnb.yml                # cnb.cool backup CI
├── GPU_PROFILES.md         # authoritative GPU profile table
├── CHANGELOG.md            # changelog
├── LICENSE                 # license terms (must read)
└── README.md               # this file (Chinese) / README.en.md (English)
```

---

## Troubleshooting

### Portable package won't run: "libnvidia-ml.so.1 not found" / "cudart64_12.dll not found"
- **Linux**: install the NVIDIA driver (`libnvidia-ml.so.1` ships with it). The CUDA runtime is already bundled in the package — no CUDA Toolkit needed.
- **Windows**: install the NVIDIA driver; all other DLLs (OpenSSL / CUDA runtime / MSVC CRT) are bundled in the zip.

### NVML temp/fan/power show as `--`
Non-fatal. Ensure `libnvidia-ml.so.1` (Linux) or `nvml.dll` (Windows) is present — it ships with the NVIDIA driver.

### Build error: "CUTLASS not found"
```bash
git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass
./build.sh
```
You can still build without CUTLASS, but it falls back to the WMMA kernel (~30 TH/s only).

### The pool rejects all shares
1. Check the wallet format: PRL bech32, starting with `prl1`
2. Verify the pool is reachable: `telnet prl.kryptex.network 7048`
3. If TLS fails, switch to plaintext port 7048
4. Check the log for the `share rejected` reason

### Low hashrate
1. `nvidia-smi` to check the GPU clock is normal (core 1500MHz+); hitting the power cap throttles clocks
2. Kill leftover profilers: `sudo pkill ncu` (ncu limits SM frequency)
3. 3090/3080 users: confirm you're using the `sm86-g8` tuned package, not generic
4. Use `TC_TIMING=1 --breakdown` to see per-draw timings and locate the bottleneck

### Windows: "cl.exe is not recognized"
Build from the "Developer Command Prompt for VS 2022", or rely on CI which handles it via `ilammy/msvc-dev-cmd`.

---

## License & Attribution

This project is released under the terms in [LICENSE](LICENSE). **Key points:**

| ✅ Allowed | ❌ Prohibited |
|---|---|
| Personal, non-commercial mining | Commercial use (paid hosting / cloud hash / paid suites) |
| Reading, studying, researching the source | Adding any devfee / hashrate skimming / share diversion |
| Forking and modifying for learning | Closed-source mods, removing LICENSE, weakening the no-commercial clause |
| Mining to your own address on a pool | Tampering with wallet/worker to steal others' hashrate |

**Derivative works must:**
1. Prominently attribute (README / `--version` / startup banner): `Based on Kan by tvvshow/kan-mine (https://github.com/tvvshow/kan-mine)`
2. Retain the [LICENSE](LICENSE) in full
3. Release under the same terms (no-commercial + no-skimming + attribution)
4. Publish the complete source code of the derivative

Violating any clause = automatic termination of your license, and the original authors reserve the right to pursue recourse.

---

## Acknowledgements

- **Pearl blockchain**: https://github.com/pearl-network/pearl
- **NVIDIA CUTLASS**: https://github.com/NVIDIA/cutlass
- **BLAKE3**: https://github.com/BLAKE3-team/BLAKE3

---

*RTX 4090: 248 TH/s (97% of the official 255) · RTX 3090: 98–135 TH/s (depends on power cap / tuning) · 100% open-source · zero dev fee · no commercial use*
