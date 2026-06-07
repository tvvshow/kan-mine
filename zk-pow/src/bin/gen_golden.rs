//! Generate a known-good "golden" Pearl PlainProof via the CPU mining path.
//!
//! Mirrors the config/header setup of `prove_verify.rs`
//! (`test_ffi_mine_prove_verify_with_rank`) but stops after CPU `mine(...)`
//! returns a PlainProof (no STARK prover is run).
//!
//! stdout : base64(STANDARD) of bincode(&PlainProof)   -> feed to verify_plain stdin
//! stderr : the exact header hex (152 chars) + chosen (m,n,k,rank,rows_pattern,cols_pattern)

use base64::{Engine as _, engine::general_purpose::STANDARD};
use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};
use zk_pow::ffi::mine::mine;

fn main() -> anyhow::Result<()> {
    // --- Header (same as prove_verify::test_block_header) ---
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x66666666,
        nbits: 0x1D2FFFFF,
    };

    // --- Mining configuration (same as prove_verify::default_mining_config) ---
    let rank: u16 = 128;
    let m: usize = 6144;
    let n: usize = 4096;
    // matches prove_verify: k = (16*rank).max(1024) + 192  -> 2240 for rank=128
    let k: usize = (16 * rank as usize).max(1024) + 192;

    let rows_pattern = PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73])?;
    let cols_pattern = PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73])?;

    let config = MiningConfiguration {
        common_dim: k as u32,
        rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern,
        cols_pattern,
        reserved: MiningConfiguration::RESERVED_VALUE,
    };

    eprintln!("mining: m={m} n={n} k={k} rank={rank}");
    eprintln!("rows_pattern={:?}", config.rows_pattern.to_list());
    eprintln!("cols_pattern={:?}", config.cols_pattern.to_list());

    // --- CPU mine: returns a PlainProof (no GPU, no STARK) ---
    let plain_proof = mine(m, n, k, header, config, None, false)?;

    eprintln!(
        "PlainProof: m={} n={} k={} noise_rank={}",
        plain_proof.m, plain_proof.n, plain_proof.k, plain_proof.noise_rank
    );
    eprintln!("a.row_indices={:?}", plain_proof.a.row_indices);
    eprintln!("bt.row_indices={:?}", plain_proof.bt.row_indices);

    // EXACT header used (so it can be fed verbatim to verify_plain)
    let header_hex = hex::encode(header.to_bytes());
    eprintln!("HEADER_HEX={header_hex}");

    // stdout: base64(bincode(&plain_proof))
    let bytes = bincode::serialize(&plain_proof)?;
    println!("{}", STANDARD.encode(bytes));

    Ok(())
}
