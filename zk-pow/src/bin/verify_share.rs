//! CPU verifier for pool-share PlainProofs.
//!
//! Usage:
//!   verify_share <header152hex> <pool_target64hex_be> < share.b64
//!
//! `verify_plain` checks against the block header's nbits.  Pool shares are
//! intentionally easier: LuckyPool/Kryptex sends a per-share `target` in
//! mining.notify, and the miner/verifier must compare
//!   blake3(jackpot, key=a_noise_seed) <= target * h * w * rounded_common_dim.
//! This binary mirrors the official verifier path but swaps in that pool target,
//! making it suitable as a pre-submit/debug gate for rejected shares.

use std::io::Read as _;

use anyhow::{Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use primitive_types::U256;
use zk_pow::{
    api::{
        proof::IncompleteBlockHeader,
        proof_utils::{CompiledPublicParams, compute_jackpot_hash},
        sanity_checks::public_params_sanity_check,
    },
    circuit::{chip::compute_jackpot, pearl_noise::compute_noise},
    ffi::plain_proof::{PlainProof, parse_plain_proof},
};

fn share_bound_from_target(target_hex_be: &str, config: &zk_pow::api::proof::MiningConfiguration) -> Result<U256> {
    let target_bytes = hex::decode(target_hex_be.trim())?;
    ensure!(target_bytes.len() == 32, "pool target must be 32 bytes / 64 hex chars");
    let target = U256::from_big_endian(&target_bytes);
    let factor = U256::from(
        config.rows_pattern.size() as usize
            * config.cols_pattern.size() as usize
            * config.dot_product_length(),
    );
    ensure!(factor > U256::zero(), "invalid zero difficulty adjustment factor");
    Ok(if target > U256::MAX / factor {
        U256::MAX
    } else {
        target * factor
    })
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        anyhow::bail!("usage: verify_share <header152hex> <pool_target64hex_be>  (base64 PlainProof on stdin)");
    }

    let header_bytes = hex::decode(args[1].trim())?;
    let header = IncompleteBlockHeader::from_bytes(&header_bytes)?;

    let mut b64 = String::new();
    std::io::stdin().read_to_string(&mut b64)?;
    let raw = STANDARD.decode(b64.trim().as_bytes())?;
    let plain_proof: PlainProof = bincode::deserialize(&raw)?;

    let (private_params, mut public_params) = parse_plain_proof(header, &plain_proof)?;
    public_params_sanity_check(&public_params)?;
    for strip in private_params.s_a.iter().chain(private_params.s_b.iter()) {
        for &val in strip {
            ensure!((-64..=64).contains(&val), "Matrix value {val} out of range [-64, 64]");
        }
    }

    let compiled = CompiledPublicParams::from(&public_params);
    let noise = compute_noise(&compiled);
    let jackpot = compute_jackpot(&compiled, &private_params.s_a, &private_params.s_b, &noise);
    public_params.hash_jackpot = compute_jackpot_hash(&jackpot, compiled.a_noise_seed());

    let bound = share_bound_from_target(&args[2], &public_params.mining_config)?;
    let hash_v = U256::from_little_endian(&public_params.hash_jackpot);
    ensure!(
        hash_v <= bound,
        "Share target not satisfied: hash={hash_v:#x} bound={bound:#x}"
    );

    println!(
        "VALID_SHARE hash={hash_v:#x} bound={bound:#x} m={} n={} k={} r={} h={} w={}",
        public_params.m,
        public_params.n,
        public_params.common_dim(),
        public_params.rank(),
        public_params.h(),
        public_params.w()
    );
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("INVALID_SHARE: {e:#}");
            std::process::exit(1);
        }
    }
}
