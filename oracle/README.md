# Pearl M2a validation oracle — REAL network config (m=n=131072)

Captured 2026-06-07 from the public closed miner `lpminer 0.1.9` via
`lpminer --pearl-share-dump` at the REAL network dims
(`--pearl-m 131072 --pearl-n 131072 --pearl-k 4096 --pearl-r 256`).
This is the offline ground-truth a fast kernel (M2a) must reproduce bit-for-bit
*before* it is trusted to hunt real shares. See memory `reference_kryptex_stratum_wire`
and `reference_pearl_gap_analysis`.

## Files
- `header.hex`            — 76-byte block header (version 1 | prev_hash 32B | merkle 32B | time | nbits=207fffff easy).
- `mining_config.hex`     — 52-byte network config blob; head u32 LE: [0]=0, [1]=4096 (k/common_dim), [2]=256 (rank), then the signal-pattern generator params (period-32; row offsets {0,8}; col offsets {0,1}). job_key = blake3(header ‖ this 52B).
- `meta.txt`              — full intermediate values for the winning easy share.
- (on box `/root/oracle/proof.bin`, 102736 B) — the bincode PlainProof itself; head u64 LE = m,n,k,noise_rank. Too big to keep here; regenerate with lpminer if needed.

## The contract a correct kernel must satisfy (from meta.txt)
- dims: m=n=131072, k=4096, r=256(noise_rank).
- signal selection: rows_pattern = 0,8,32,40,64,72,96,104 (h=8); cols_pattern = 0,1,32,33,…,224,225 (w=16) ⇒ 128 signal pairs (listed verbatim).
- job_key   = ebcbd9ae056ce80d1317c48889224ceb72533d2e36019a351454b3daa11e07ab
- hash_a    = ee91778acf45c76952b5b99422ad1b5c93ed70e520d7af30d7321ded6bc5cdd6
- hash_b    = 6abf70579c6cfeecc83cc2a346c74c597784ba6d3192b0fec39280035a726569
- a_noise_seed = fee4cec5f030e759a7d01f2ae6dde7df5d2fd2de4e8b444fff5b2a8d2ce77fcc
- b_noise_seed = 1cbb894eaa0200d3c23cbd256abc48c1c0c7738e1a84ad1c3683056726bb35fb
- winning tile = (tile_m=5, tile_n=0), chain_id=0, 1 hash/thread.
- EXPECTED 16-word transcript (CPU == GPU, both agree):
  fffcb9f4,fff9f2ad,00013a4c,ffff97d4,ffff02c8,fffdecdd,000153c8,0018e010,
  001a31a8,0006f41d,ffff0ac1,00080ce8,fff9b087,ffff873b,fffbbeb3,fffb83f8

## How to use
A reference (CPU) or fast (GPU) kernel that, given `header.hex` + `mining_config.hex`,
derives job_key → A,B → adds noise (seeds above) → does the int8 GEMM → folds the
signal rows×cols → MUST produce exactly the transcript above for tile (5,0). Matching
it offline = the kernel is correct at REAL config; only then is it worth pointing at
the live pool (where the same machinery must find a tile meeting diff 2^20).

Regenerate a fresh oracle anytime (different job_key/seeds/transcript):
`/root/lp/lpminer --pearl-share-dump DIR --pearl-m 131072 --pearl-n 131072 --pearl-k 4096 --pearl-r 256`
