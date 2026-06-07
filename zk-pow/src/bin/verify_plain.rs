//! CPU ground-truth verifier for Pearl PlainProofs.
//!
//! Usage:
//!   verify_plain <header152hex>   < golden.b64
//!
//! argv[1]      : 152 hex chars = 76-byte serialized IncompleteBlockHeader.
//! stdin        : base64 (STANDARD) of bincode(&PlainProof).
//! Prints "VALID" to stdout on success; "INVALID: {e:#}" to stderr + exit 1 on failure.

use std::io::Read as _;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use zk_pow::api::proof::IncompleteBlockHeader;
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::plain_proof::PlainProof;

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("usage: verify_plain <header152hex>  (base64 PlainProof on stdin)");
    }

    // argv[1] -> 76-byte header
    let header_bytes = hex::decode(args[1].trim())?;
    let header = IncompleteBlockHeader::from_bytes(&header_bytes)?;

    // stdin -> base64 -> bincode -> PlainProof
    let mut b64 = String::new();
    std::io::stdin().read_to_string(&mut b64)?;
    let raw = STANDARD.decode(b64.trim().as_bytes())?;
    let plain_proof: PlainProof = bincode::deserialize(&raw)?;

    verify_plain_proof(&header, &plain_proof)
}

fn main() {
    match run() {
        Ok(()) => {
            println!("VALID");
        }
        Err(e) => {
            eprintln!("INVALID: {e:#}");
            std::process::exit(1);
        }
    }
}
