# pearl-miner — runbook (build · pool · solo)

Self-contained Pearl(PRL) PoUW miner. One `build.sh` produces three artifacts in
`build/`:

| artifact          | role                                                            |
|-------------------|-----------------------------------------------------------------|
| `pearl-miner`     | unified miner — `--pool` (kryptex/LuckyPool) and `--solo` (pearld) |
| `plainproof_gen`  | standalone CLI PlainProof generator (CPU/dp4a/tensor-core)      |
| `zkprove`         | SOLO-only: PlainProof → plonky2 ZK proof → assembled block_hex  |

## Build (standalone, one command)

```bash
cd peral
bash build.sh
```

- Needs: `nvcc` (CUDA toolkit), `g++`, `libssl-dev` (OpenSSL), and — for `zkprove`
  / solo — a Rust toolchain (`cargo`). CUDA **compiles without a GPU**; only running
  the kernels needs one.
- Arch: `ARCH=sm_90a bash build.sh` to pin; otherwise auto-detects via `nvidia-smi`
  or emits a portable sm_75..sm_90 fatbin (+PTX for Blackwell JIT).
- Pool-only host without Rust: `pearl-miner --pool` still builds; `zkprove` is
  skipped (only `--solo` needs it).

Verified end-to-end on a clean `nvidia/cuda:12.4.1-devel-ubuntu22.04` runner
(cnb `unified` pipeline): all three artifacts build, `cpu_test.sh` passes, and
`zkprove selftest` matches the node-golden coinbase byte-for-byte.

## Pool mining (the earning path)

```bash
./build/pearl-miner --pool \
  --wallet prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv \
  --worker w1
```

Defaults: `prl.kryptex.network:7048` plaintext, real network config, fused
tensor-core search, `agent=lpminer/0.1.9-552bdfe`. The miner submits the base64
PlainProof; the pool builds the ZK proof. On a new job it aborts the stale draw and
restarts; a win mined for an old job is dropped. Tunables: `--pool-host --pool-port
--agent --hs --batch N --cfg golden|real --gpu` (dp4a).

Cheap liveness check (no mining): `--net-probe` connects, authorizes, awaits one
job, then exits — proves the wire contract end-to-end.

## Solo mining (against your own pearld)

```bash
./build/pearl-miner --solo \
  --node 127.0.0.1:44107 --rpcuser USER --rpcpass PASS \
  --addr <p2tr taproot payout address> \
  --zkprove ./build/zkprove
```

Flow per round: `getblocktemplate` → `zkprove header` (build coinbase + merkle →
incomplete header + target) → mine → `zkprove block` (plonky2 ZK proof + assemble
`ZK_CERT|HEADER|TXCOUNT|TXNS`) → `submitblock`. A background poller aborts the
current draw when a new block arrives. The payout address MUST be Taproot (P2TR,
bech32m). pearld needs `getblocktemplate` enabled (`miningaddr`, RPC user/pass; the
miner tolerates the self-signed RPC cert).

> Note: solo at mainnet difficulty effectively never wins on one GPU (variance is
> astronomical) — solo is for correctness/end-to-end validation against a real node
> (regtest/testnet recommended). The pool is the earning path.

## CI (cnb.cool) pipelines (by branch)

| branch     | runner        | what                                                   |
|------------|---------------|--------------------------------------------------------|
| `main`     | free (no GPU) | build (pool binary) + cpu-verify                       |
| `unified`  | free (no GPU) | **full** build (incl. Rust `zkprove`) + selftest + smoke |
| `netprobe` | GPU (egress)  | build + `--pool --net-probe` live stratum validation   |
| `gpu`      | GPU (metered) | build + cpu-verify + dp4a kernel smoke (`run_test.sh`)  |
