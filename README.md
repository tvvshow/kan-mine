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
bash build.sh          # auto-detects GPU arch via nvidia-smi; outputs build/plainproof_gen
bash run_test.sh       # smoke test: --mine 20 at golden nbits -> expect a win + proof
```
The tensor-core path is built only when `CC >= 80` and `CUTLASS_DIR/include` exists
(default `./cutlass`; override with `CUTLASS_DIR=...`).

## CI (cnb.cool)
`.cnb.yml` runs build + smoke-test on a CUDA GPU runner on push to `main`.

## Notes
- int7 base [-64,64] + bounded noise keeps values in int8 [-128,126], so int8 MMA is
  bit-exact to the int32 reference.
- Perf gotcha on throttled containers: set `OMP_NUM_THREADS` to the real CPU quota
  (oversubscription can cost ~10-16×).
