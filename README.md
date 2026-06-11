# peral — self-built, auditable GPU solver for Pearl (PRL) Proof-of-Useful-Work

From-scratch C++/CUDA solver for Pearl's int8 NoisyGEMM jackpot search. Fully
auditable, competes with closed-source `SRBMiner-MULTI` / `lpminer`, and is a
proven continuous earner on the kryptex/LuckyPool mainnet.

> The complete operations manual (install → build → run → monitor → troubleshoot,
> in Chinese) lives at **`../README.md`**. This file is the package-level summary.

## ONE fast version — no variants

There is exactly **one** GPU kernel and **one** build path. The redundant/slow
experiments (`tc_gemm` 0.74 TH/s, `tc_panel` wrong-result, `jackpot_kernel`
dp4a fallback) were discarded to `../_archive/dead-kernels/`.

## Layout
- `src/tc_block.cu` — **the one kernel**: GATHER pre-pass → shared-memory-blocked
  WMMA int8 GEMM with a STAGES=3 cp.async pipeline → fused per-rank-chunk XOR-fold
  jackpot. ~30 TH/s on RTX 3080 Ti (sm_86), using the SRBMiner-MULTI/lpminer
  TH/s formula; CPU-postcheck verified at the real 131072 config and proven on
  the live pool.
- `src/plainproof_gen.cpp` — CPU pipeline + driver: fast splitmix64 A/B fill →
  BLAKE3 commitments → noise → jackpot search orchestration → Merkle proof +
  bincode PlainProof. Compiles two ways: with `main()` (the `plainproof_gen` CLI)
  and with `-DPROVER_LIB` (the `prover_lib.o` linked into `pearl-miner`).
- `src/miner_main.cpp` — unified `pearl-miner` driver: `--pool` (kryptex) / `--solo`.
- `src/prover.h` — the `mine_plain_proof()` entry point.
- `blake3/` — vendored BLAKE3 C; `build.sh` adds the SIMD `.S` (AVX2, ~6× the
  scalar path), auto-fetching them if absent.
- `zk-pow/` — Rust `zkprove` helper, **`--solo` only** (needs `cargo`).

## Build (one command, GPU not required to compile)
```bash
cd peral
bash build.sh                 # arch auto-detect; override: ARCH=sm_86 bash build.sh
```
Outputs in `build/`: `pearl-miner` (the miner), `plainproof_gen` (self-test CLI),
and — only if `cargo` is present — `zkprove` (solo). `build.sh` is the live fast
path: it compiles the one kernel `tc_block.cu` and links AVX2 BLAKE3 by default.

## Run
```bash
# pool (the earning path)
./build/pearl-miner --pool --wallet <addr> --worker <name>

# correctness self-test (expect: tc(block) … TH/s + POSTCHECK … ok=1)
OMP_NUM_THREADS=$(nproc) ./build/plainproof_gen --mine 3 --cfg real
```
Pool contract: plaintext TCP `prl.kryptex.network:7048`, `mining.authorize` with
`wallet="<addr>.<worker>"`, `mining.submit` `{job_id, plain_proof: <base64 bincode
PlainProof>, hs}`. `plainproof_gen` CPU-postchecks every GPU-reported win against
the active target before emitting; `pearl-miner` drops wins mined for a stale job.

Helper scripts: `cpu_test.sh` (CPU-only correctness), `run_test.sh` (GPU smoke),
`start_pool.sh` (persistent pool launcher), `verify_run.sh`.
