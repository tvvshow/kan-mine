# Pearl(PRL) 独立挖矿客户端：官方源码方案 + 与本实现比对

> 依据：本地官方源码 `D:\mybitcoin\3\pearl`（Go 节点 + Rust `zk-pow` + Python `miner-base`）。
> 本实现：`D:\mybitcoin\3\peral`（`plainproof_gen.cpp` CPU 参考 + `tc_block.cu` GPU kernel + `miner_main.cpp` 网络层）。
> 结论：算法与 PlainProof 序列化每一环都与官方一致，且已被官方源码编译出的验证器实证为 VALID；唯一非算法差异是矿池 stratum 协议（官方仓库不含）。剩余短板是 GPU kernel 速度（见 `DESIGN_speedup.md`）。

## 1. 挖矿核心在哪（权威实现位置）

挖矿核心**不在** Go 节点里，而在三处：

| 角色 | 路径 | 作用 |
|---|---|---|
| 算法权威 | `zk-pow/src/ffi/mine.rs`、`circuit/pearl_noise.rs`、`api/verify.rs`、`api/sanity_checks.rs`、`ffi/plain_proof.rs` | 出题/加噪/jackpot/目标/PlainProof/**验证器** |
| 可读参考 | `miner/miner-base/src/miner_base/*.py`（noisy_gemm/noise_generation/inner_hash/commitment_hash/matrix_merkle_tree） | 同一算法的 Python 镜像 |
| 客户端外壳 | `miner/pearl-gateway/...` + `gateway_client.py` | miner↔gateway 的 JSON-RPC（`getMiningInfo`/`submitPlainProof`） |

关键认知：**A、B 矩阵由矿工本地随机生成**（`mine.rs:33-39`，`signal ∈ [-64,64]` 含端点），不是外部下发的真实算力任务。所以"随机 A/B"在协议层就是正确做法。

## 2. 一次 draw 的算法（`mine.rs::try_mine_one`）

```
输入：incomplete_header(76B)
      MiningConfiguration(common_dim=k, rank, mma_type=Int7xInt7ToInt32,
                          rows_pattern, cols_pattern, reserved[32]=0)

1. job_key   = blake3( header.to_bytes() || config.to_bytes() )            # 无 key
2. 随机生成   A(m×k), B(k×n) ∈ [-64,64]，Bt = Bᵀ
3. 承诺链：
     hash_a = blake3(pad1024(A_rowmajor),  key=job_key)
     hash_b = blake3(pad1024(Bt_colmajor), key=job_key)
     b_noise_seed = blake3( job_key      || hash_b )                       # 无 key
     a_noise_seed = blake3( b_noise_seed  || hash_a )                      # 无 key
4. 噪声（pearl_noise.rs；NOISE_RANGE=128 → UNIFORM=64, RANGE_MASK=63, ZERO_POINT=32）：
     E_AL = uniform(label"A_tensor", a_noise_seed)        # byte&63 - 32
     E_AR = perm  (label"A_tensor", a_noise_seed)         # k 对 [first, second]
     E_BL = perm  (label"B_tensor", b_noise_seed)
     E_BR = uniform(label"B_tensor", b_noise_seed)
       get_random_hash: msg = i32(1+index)@(prepend_index*4) ‖ seed(32B)，keyed blake3
       perm: first = ru & (r-1); second = first ^ (1 + mulhi_u32(r-1, ru))
     noise_a[row]  = matvec_sparse_perm(E_AR, E_AL[row])   # v[first]-v[second]
     noise_bt[col] = matvec_sparse_perm(E_BL, E_BR[col])
     A' = A + noise_a ;  Bt' = Bt + noise_bt               # int8
5. tile 遍历：a_rows = threads_partition(rows_pattern, m)；b_cols 同理
   每个 tile（h×w）：
     jackpot[16] = 0;  dot_len = k - (k % rank)
     for ll in (rank..=k step rank):
        tile[u][v] += Σ_{l∈[ll-rank,ll)} A'[a_idx][l] * Bt'[b_idx][l]
        xored = XOR(所有 tile cell 当作 u32)
        tid = (ll/rank - 1) % 16
        jackpot[tid] = rotl(jackpot[tid], 13) ^ xored
     jackpot_hash = blake3( jackpot(16×u32 LE = 64B), key=a_noise_seed )   # 单块 keyed root, flags=27
     命中 ⇔ u256_LE(jackpot_hash) <= bound
   bound = nbits_to_difficulty(nbits) * (h*w*dot_len)，溢出饱和到 U256::MAX
6. PlainProof = { m, n, k, noise_rank, a:MerkleProof(行), bt:MerkleProof(列) }
   bincode → base64 → 提交
```

验证器 `verify_plain_proof` 反向：从 proof 的 `row_indices` **重建 config** → 范围检查 strip∈[-64,64] → 重算 noise/jackpot/hash → 检查 `≤ bound`，并 `ensure_eq!(重算 hash_a/b, proof.root)` 绑定 Merkle。**矿池用 `nbits_override` 决定真实难度，忽略下发 target**。

## 3. 与本实现逐项比对

| 组件 | 官方源码 | 本实现 | 结论 |
|---|---|---|---|
| job_key | `blake3(header‖config)` | `blake3_digest(jk_input)` | ✅ |
| 承诺链 seeds | `mine.rs:163-178` | `produce_draw`(b) 同序 | ✅ |
| 噪声 uniform | `byte&63 - 32` | `RANGE_MASK=63, ZERO_POINT=32` | ✅ |
| 噪声 perm | `first^(1+mulhi(r-1,ru))` | `mul_hi_u32` 同式 | ✅ |
| seed/label 路由 | A↔a_noise_seed，B↔b_noise_seed | 同 | ✅ |
| jackpot 累加 | `rotl(·,13)^xored`, tid=%16 | kernel `rotl32d(·,13)^x`, `c%16` | ✅ |
| jackpot hash | keyed blake3 单块 flags=27 | `jackpot_blake3` v[15]=27 | ✅ |
| bound | `diff*(h*w*dot_len)` 饱和 | `mul_u256_u64_saturate` | ✅ |
| 信号范围 | `[-64,64]` 含端点 | RNG → `[-64,64]` | ✅（+64 合法） |
| Merkle | KEYED, 1024B chunk | `chunk_cv/parent_cv/root_cv` 同 flags | ✅ |
| bincode leaf_data | `serde_chunk_vec`→`Vec<&[u8]>`（每片 u64长度+字节） | 每片 `u64(1024)`+字节 | ✅ 逐字节 |
| PlainProof 字段序 | m,n,k,noise_rank,a,bt | 同 | ✅ |
| 真实网络 config | m=n=131072,k=4096,r=256 | `--cfg real` + 抓包 pattern | ✅ |

## 4. 两处"差异"（均非 bug）

1. **RNG 分布不同**：官方 `rand` 均匀；本实现 splitmix64 + `(byte*129>>8)-64`。对正确性无影响——验证器只从提交的 A/B 叶子重算 jackpot，不关心抽样方式，只要值域 `[-64,64]`。本映射值域正是 `[-64,64]`。
2. **网络协议不同（架构层）**：官方源码只有 gateway JSON-RPC（`getMiningInfo`/`submitPlainProof`）；真实矿池（Kryptex/LuckyPool）用 stratum（7048 明文 `mining.authorize/notify/submit`），**官方仓库不含**。本实现 `miner_main.cpp` 按 MITM 抓包实现该 stratum（`--pool`）+ pearld 直连（`--solo`），线上验证通过。

## 5. 实证

- 官方源码编译出的验证器对本实现 proof（CPU 路径 + GPU `--mine` 重抽路径）跑出 **ALL-VALID**。
- 线上矿池：153 个 share 全部接受、0 拒绝。

**结论：作为独立挖矿客户端，本实现与官方完全等价，正确性无缺口。唯一短板是 GPU kernel 速度——属性能问题，见 `DESIGN_speedup.md`。**
