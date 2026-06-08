# peral — self-built open-source GPU solver for Pearl (PRL) Proof-of-Useful-Work

Trusted, from-scratch CUDA solver for Pearl's int7×int7→int32 NoisyGEMM jackpot
search. Goal: match the closed-source reference solver's throughput on consumer
Blackwell (RTX 5090, sm_120) while being fully auditable.

## Layout
- `src/plainproof_gen.cpp` — CPU pipeline + driver: A/B generation, BLAKE3
  commitments, noise, jackpot search orchestration, Merkle proof + bincode
  PlainProof serialization. Flags: `--mine N --header <152hex> --target <64hex>`,
  `--tc` (tensor-core path), `--gpu` (dp4a single-shot), default = golden self-test.
- `src/jackpot_kernel.cu` — dp4a (CUDA-core) per-tile jackpot fold + search
  (`gpu_mine_init/draw/free`, `gpu_jackpot_search`). Runs on sm_61+.
- `src/tc_gemm.cu` — tensor-core path (CUTLASS legacy Sm80 int8 GEMM, chunked over
  k with cumulative accumulation so the jackpot can snapshot every `rank` steps).
  Needs sm_80+ and CUTLASS headers; otherwise `src/tc_stub.cpp` is linked instead.
- `blake3/` — vendored BLAKE3 C (portable only; SIMD sources not included).

## Build
```bash
bash build.sh          # GPU NOT required to compile; outputs build/plainproof_gen
bash cpu_test.sh       # CPU-only correctness check: reference search -> PlainProof (no GPU)
bash run_test.sh       # GPU smoke test: --mine 20 at golden nbits (needs a GPU)
```
CUDA compiles fine on a GPU-less host. `build.sh` picks the arch as
`ARCH` env override → `nvidia-smi` detection (if a GPU is present) → a portable
multi-arch fatbin (`sm_75..sm_90` SASS + Hopper PTX for Blackwell JIT). The
tensor-core path is built only when `CUTLASS_DIR/include` exists **and** an
explicit `sm_80+` `ARCH` is set (e.g. `ARCH=sm_90a`); otherwise a stub links.

## CI (cnb.cool)
`.cnb.yml` builds and **correctness-verifies on CPU** on push to `main` — no GPU
needed for that. The GPU kernel smoke test (`run_test.sh`) is added once a GPU
runner is requested.

## Notes
- int7 base [-64,64] + bounded noise keeps values in int8 [-128,126], so int8 MMA is
  bit-exact to the int32 reference.
- Perf gotcha on throttled containers: set `OMP_NUM_THREADS` to the real CPU quota
  (oversubscription can cost ~10-16×).


## Kryptex/LuckyPool live mining contract

Use the maintained live-pool path only:

```bash
GEN_EXTRA="--tc --cfg real" python3 m1c_pool_hunt.py
```

The captured lpminer-compatible contract is plaintext TCP `prl.kryptex.network:7048`,
`mining.authorize` with `wallet="<addr>.<worker>"`, and `mining.submit` params
`{"job_id": "...", "plain_proof": "<base64 bincode PlainProof>", "hs": <int>}`.
`plainproof_gen` performs a CPU postcheck of each GPU-reported win against the active
pool target before emitting the proof, and `m1c_pool_hunt.py` drops stale wins instead
of submitting a proof mined for a previous job.
