# Pearl Miner

Self-built Pearl (PRL) Proof-of-Useful-Work (PoUW) miner with CUTLASS int8 tensor-core acceleration. Achieves **106+ TH/s** on RTX 3090 (vs lpminer reference 91 TH/s).

## Features

- **Pool mining**: LuckyPool/kryptex stratum (plaintext TCP or TLS)
- **Solo mining**: direct pearld RPC (getblocktemplate/submitblock)
- **Tensor-core acceleration**: fused CUTLASS int8 GEMM + GPU-resident draw pipeline (RNG + blake3 + noise + search all on GPU)
- **lpminer-compatible logging**: stats table every 120s with NVML hardware monitoring (temp/fan/power/efficiency)
- **Zero devfee**

## Build

### Prerequisites

- **CUDA Toolkit 12.x** (tested on 12.8)
- **CUTLASS 3.5.1**: clone to `~/cutlass` or set `CUTLASS_HOME`
- **OpenSSL**: `libssl-dev` (for pool TLS + solo HTTPS)
- **GCC/G++** with C++17 support

### Quick start

```bash
# Clone
git clone https://cnb.cool/wuyueyi/peral
cd peral

# Ensure CUTLASS is available
export CUTLASS_HOME=~/cutlass  # or wherever you cloned CUTLASS v3.5.1

# Build (auto-detects CUTLASS and links tensor-core kernel)
./build.sh

# Binary output: build/pearl-miner
```

The build script automatically:
- Links `tc_cutlass_v2.cu` (fused tensor-core kernel) + `gpu_prep.cu` (GPU-resident pipeline) when CUTLASS is found
- Falls back to CPU-only build if CUTLASS is missing
- Compiles with AVX2-accelerated blake3 when available

## Run

### Pool mining (LuckyPool / kryptex)

**Plaintext (port 7048, recommended):**
```bash
./build/pearl-miner --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv.myworker
```

**TLS (port 8048):**
```bash
./build/pearl-miner --algo pearl \
  --pool stratum+ssl://prl.kryptex.network:8048 \
  --wallet YOUR_PRL_ADDRESS.worker
```

The wallet can be specified as `ADDR.WORKER` (combined) or split with `--wallet ADDR --worker NAME`.

### Solo mining (requires pearld node)

```bash
./build/pearl-miner --solo \
  --node 127.0.0.1:44107 \
  --rpcuser your_rpc_user \
  --rpcpass your_rpc_pass \
  --addr prl1... \
  --zkprove ./zkprove
```

Solo mode requires the `zkprove` helper (from the official Pearl release) to convert PlainProof → ZK proof → block.

### Interactive commands

While running:
- **`s`** — print stats table immediately (otherwise prints every 120s)
- **`q`** — quit gracefully

### Common parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--algo pearl` | Algorithm (only pearl supported) | — |
| `--cfg real` | Use real network config (m=n=131072, k=4096, rank=256) | `real` |
| `--batch N` | Max draws per job before refetch | 1000000 |
| `--breakdown` | Print per-draw timing breakdown | off |

## Output

**Startup banner:**
```
11:09:33  info   about         pearl-miner/self-built (106+ TH/s RTX 3090)
11:09:33  info   cpu           AMD EPYC 7402 24-Core Processor (16 threads)
11:09:33  info   algo          pearl
11:09:33  info   pool          stratum+tcp://prl.kryptex.network:7048
11:09:33  info   wallet        prl1patz...apmv.myworker
11:09:33  info   worker        myworker
11:09:33  info   commands      s (stats), q (quit); table every 120s
11:09:33  info   detected      1 devices - driver 570.153.02
11:09:33  info   GPU           #0 RTX 3090        24GB sm_86 bus:00 enabled
11:09:33  info   devfee        0%
11:09:34  info   stratum       authorize: ok wallet=prl1patz...apmv.myworker agent=lpminer/0.1.9-552bdfe
11:09:34  info   stratum       new job id=5bd59536_2097152 height=71246 diff=2097152 seq=1
```

**Stats table (every 120s or on `s` keypress):**
```
-----prl1patz...apmv.myworker--------------------stratum+tcp://prl.kryptex.network:7048-----
 DEVICE MODEL              HASHRATE  TEMP  FAN POWER      EFFIC       A    R  LAST
----------------------------------------------------------------------------------------
 GPU #0 RTX 3090        106.50 TH/s   63C  37%  345W  308.7 GH/W       5    0    2m
----------------------------------------------------------------------------------------
 10s                    106.50 TH/s               345W           A: 5
 60s                    106.50 TH/s                              R: 0
 15m                    106.50 TH/s                              S: 0
[0 days 00:15:42]-----------------------------------[100.0% accept - ver. self-built]
```

**Event lines:**
```
11:09:55  info   GPU #0        share accepted
11:10:50  info   stratum       new job id=ae1bc8ff_2097152 height=71246 diff=2097152 seq=2
```

## Technical details

### Speed optimizations (Week 1)

- **Day 3**: CUTLASS threadblock-level `MmaMultistage` (int8 tensor-core) → 72-73 TH/s
- **Day 5**: Fused `FoldMmaMultistage` (fold callback every rank-256 chunk, no barriers) → 79-81 TH/s
- **Day 5b**: GPU-resident pipeline (`gpu_prep.cu`: RNG + blake3 tree + noise on GPU) → 76.6 TH/s wall
- **Day 6**: Grouped raster (column-major blockIdx inside bands, kills 525GB/draw re-read) → 102.5 TH/s
- **Day 6c**: Lane-distributed fold v3 (warp's 32 lanes ↔ 32 jackpot tiles 1:1, one parallel RMW) + async search/prep overlap → **106.5 TH/s wall, 100+ shares accepted, 0 rejects**

Kernel: 629.5ms vs 522ms pure-GEMM roofline (~17% gap, occupancy locked at 1 TB/SM by smem 89KB of 100KB cap).

### CUTLASS recipe

- DefaultMma `<int8 RM, int8 CM, int32 RM, TensorOp, Sm80, TB 128×256×64, W 64×64×64, IMMA 16×8×32, stages=3, OpMultiplyAddSaturate>`
- Grouped raster `GROUPM=8` (compile knob `-DGROUPM=N`)
- Fold every 4 mac_loop_iter (rank/64) via `gemm_iters_fold()` callback

## Troubleshooting

**Build fails "CUTLASS not found":**
```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass
export CUTLASS_HOME=~/cutlass
./build.sh
```

**NVML warnings (temp/fan/power show `--`):**
- Non-critical; stats table still works, just no hardware monitoring
- Ensure `libnvidia-ml.so.1` is available (part of NVIDIA driver)

**Low hashrate (<50 TH/s):**
- Check `nvidia-smi` clocks (should be ~1900MHz+ core for 3090)
- Kill stuck profiler processes: `sudo pkill ncu`
- Verify CUTLASS kernel linked: run with `--breakdown` and look for `tc(cutlass2):` lines

**Pool rejects all shares:**
- Verify wallet address format (PRL bech32, starts with `prl1`)
- Check pool is reachable: `telnet prl.kryptex.network 7048`
- Try plaintext 7048 if TLS 8048 fails

## License

Self-built, no license restrictions. Based on the official Pearl zk-pow reference (Apache 2.0) and NVIDIA CUTLASS (BSD 3-Clause).

## Credits

- **Pearl blockchain**: https://github.com/pearl-network/pearl
- **CUTLASS**: https://github.com/NVIDIA/cutlass
- **lpminer reference**: 91 TH/s on RTX 3090 (closed-source, used as A/B test baseline)

Developed during Week 1 (2026-06-11): +106.5 TH/s in 6 days from 15 TH/s session start. All optimizations validated live on real pool with real wallet.
