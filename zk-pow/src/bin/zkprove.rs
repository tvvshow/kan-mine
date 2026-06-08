//! zkprove — SOLO-mode helper for the self-built Pearl miner.
//!
//! The C++ `pearl-miner` owns networking (pearld JSON-RPC) and the fast CUDA
//! PlainProof search.  This Rust binary owns the two pieces that MUST reuse the
//! vendored zk-pow crate / standard Bitcoin block construction and would be
//! error-prone to re-implement in C++:
//!
//!   1. building the block header to mine against (coinbase + merkle_root), and
//!   2. turning a winning PlainProof into a full, submittable block:
//!         PlainProof --zk_prove_plain_proof--> ZKProof --assemble--> block_hex
//!
//! Subcommands (block template JSON is read from stdin in every case):
//!   zkprove header --addr <p2tr>                  -> JSON {incomplete_header, target, height}
//!   zkprove block  --addr <p2tr> --pp <b64>       -> hex block for `submitblock`
//!   zkprove selftest                              -> validates coinbase vs a node-golden vector
//!
//! The coinbase/merkle/cert/header byte layout is a faithful port of the official
//! pearl-gateway `blockchain_utils.py` + `pearl_block.py` (validated byte-for-byte
//! by `selftest` against the node coinbase vector in test_blockchain_utils.py).
//! No new crate dependencies: SHA-256 and bech32m are hand-rolled here so the
//! build stays hermetic over the already-vendored lockfile.

#[cfg(unix)]
use tikv_jemallocator::Jemalloc;
#[cfg(unix)]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use std::io::Read as _;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::Value;

use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams,
};
use zk_pow::api::prove::zk_prove_plain_proof;
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::ffi::mine::mine as cpu_mine;
use zk_pow::ffi::plain_proof::PlainProof;
const PUBLICDATA_SIZE: usize = PublicProofParams::PUBLICDATA_SIZE; // 164
const ZK_CERTIFICATE_VERSION: u32 = 1;

// ============================================================================
// SHA-256 (FIPS 180-4) — hand-rolled, no external crate.
// ============================================================================
const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bitlen = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K256[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

fn dsha256(data: &[u8]) -> [u8; 32] {
    sha256(&sha256(data))
}

// ============================================================================
// bech32 / bech32m decode (BIP173/BIP350) — extract the 32-byte P2TR program.
// ============================================================================
const BECH32_CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
const BECH32M_CONST: u32 = 0x2bc830a3;

fn bech32_polymod(values: &[u8]) -> u32 {
    const GEN: [u32; 5] = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
    let mut chk: u32 = 1;
    for &v in values {
        let b = chk >> 25;
        chk = ((chk & 0x1ffffff) << 5) ^ (v as u32);
        for (i, g) in GEN.iter().enumerate() {
            if (b >> i) & 1 == 1 {
                chk ^= g;
            }
        }
    }
    chk
}

fn hrp_expand(hrp: &[u8]) -> Vec<u8> {
    let mut v: Vec<u8> = hrp.iter().map(|c| c >> 5).collect();
    v.push(0);
    v.extend(hrp.iter().map(|c| c & 31));
    v
}

/// convertbits(from->to). Returns None on invalid padding (pad=false).
fn convertbits(data: &[u8], from: u32, to: u32, pad: bool) -> Option<Vec<u8>> {
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    let mut out = Vec::new();
    let maxv = (1u32 << to) - 1;
    let max_acc = (1u32 << (from + to - 1)) - 1;
    for &value in data {
        let value = value as u32;
        if (value >> from) != 0 {
            return None;
        }
        acc = ((acc << from) | value) & max_acc;
        bits += from;
        while bits >= to {
            bits -= to;
            out.push(((acc >> bits) & maxv) as u8);
        }
    }
    if pad {
        if bits > 0 {
            out.push(((acc << (to - bits)) & maxv) as u8);
        }
    } else if bits >= from || ((acc << (to - bits)) & maxv) != 0 {
        return None;
    }
    Some(out)
}

/// Decode a P2TR (Taproot) address of ANY hrp; enforce witness v1 + bech32m +
/// 32-byte program. Returns the 32-byte witness program.
fn p2tr_program(addr: &str) -> anyhow::Result<[u8; 32]> {
    let addr = addr.trim();
    let lower = addr.to_lowercase();
    if addr.chars().any(|c| c.is_ascii_uppercase()) && addr.chars().any(|c| c.is_ascii_lowercase())
    {
        anyhow::bail!("mixed-case bech32 address");
    }
    let pos = lower.rfind('1').ok_or_else(|| anyhow::anyhow!("no separator '1' in address"))?;
    if pos == 0 || pos + 7 > lower.len() {
        anyhow::bail!("invalid bech32 separator position");
    }
    let hrp = &lower.as_bytes()[..pos];
    let data_part = &lower.as_bytes()[pos + 1..];
    let mut data = Vec::with_capacity(data_part.len());
    for &c in data_part {
        let idx = BECH32_CHARSET
            .iter()
            .position(|&x| x == c)
            .ok_or_else(|| anyhow::anyhow!("invalid bech32 char"))?;
        data.push(idx as u8);
    }
    // checksum must be bech32m for taproot
    let mut chk_input = hrp_expand(hrp);
    chk_input.extend_from_slice(&data);
    if bech32_polymod(&chk_input) != BECH32M_CONST {
        anyhow::bail!("address is not valid bech32m (taproot requires bech32m)");
    }
    let payload = &data[..data.len() - 6];
    if payload.is_empty() {
        anyhow::bail!("empty witness program");
    }
    let witver = payload[0];
    if witver != 1 {
        anyhow::bail!("expected taproot witness version 1, got {witver}");
    }
    let program = convertbits(&payload[1..], 5, 8, false)
        .ok_or_else(|| anyhow::anyhow!("convertbits failed"))?;
    if program.len() != 32 {
        anyhow::bail!("taproot program must be 32 bytes, got {}", program.len());
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&program);
    Ok(out)
}

// ============================================================================
// Bitcoin serialization primitives.
// ============================================================================
fn varint(n: u64) -> Vec<u8> {
    if n < 0xfd {
        vec![n as u8]
    } else if n <= 0xffff {
        let mut v = vec![0xfd];
        v.extend_from_slice(&(n as u16).to_le_bytes());
        v
    } else if n <= 0xffff_ffff {
        let mut v = vec![0xfe];
        v.extend_from_slice(&(n as u32).to_le_bytes());
        v
    } else {
        let mut v = vec![0xff];
        v.extend_from_slice(&n.to_le_bytes());
        v
    }
}

/// Encode a single data push (script): opcode + bytes (OP_PUSHBYTES / PUSHDATA1/2).
fn push_data(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let n = data.len();
    if n < 0x4c {
        out.push(n as u8);
    } else if n <= 0xff {
        out.push(0x4c);
        out.push(n as u8);
    } else if n <= 0xffff {
        out.push(0x4d);
        out.extend_from_slice(&(n as u16).to_le_bytes());
    } else {
        out.push(0x4e);
        out.extend_from_slice(&(n as u32).to_le_bytes());
    }
    out.extend_from_slice(data);
    out
}

/// BIP34 height as a minimally-encoded script number push (matches
/// bitcoinutils `Script([height]).to_hex()`): OP_0 / OP_1..16 / pushdata(CScriptNum).
fn bip34_height_script(height: i64) -> Vec<u8> {
    if height == 0 {
        return vec![0x00]; // OP_0
    }
    if (1..=16).contains(&height) {
        return vec![0x50 + height as u8]; // OP_1..OP_16
    }
    // CScriptNum minimal little-endian encoding (heights are positive).
    let mut n = height as u64;
    let mut bytes = Vec::new();
    while n > 0 {
        bytes.push((n & 0xff) as u8);
        n >>= 8;
    }
    if bytes.last().map_or(false, |&b| b & 0x80 != 0) {
        bytes.push(0x00); // keep sign positive
    }
    push_data(&bytes)
}

/// Result of building the coinbase: full (segwit) serialization for the block body,
/// and the txid (double-sha256 of the non-witness serialization), big-endian/display.
struct Coinbase {
    full_bytes: Vec<u8>,
    txid_be: [u8; 32],
}

#[allow(clippy::too_many_arguments)]
fn build_coinbase(
    height: i64,
    coinbase_value: u64,
    p2tr_program: &[u8; 32],
    aux_flags: &[u8],
    witness_commitment: Option<&[u8]>,
) -> Coinbase {
    // scriptSig = BIP34_height_push || 0x00 (extra nonce) || aux_flags (raw script).
    // The node-golden vector proves these are concatenated RAW (the aux flags from
    // getblocktemplate are themselves already a compiled script fragment, e.g.
    // "0b2f..." = OP_PUSHBYTES_11 "/P2SH/btcd/"); there is NO extra outer push.
    let mut script_sig = bip34_height_script(height);
    script_sig.push(0x00);
    script_sig.extend_from_slice(aux_flags);

    // outputs: payout (P2TR) [+ witness-commitment OP_RETURN]
    let mut spk = vec![0x51u8, 0x20]; // OP_1 PUSH32
    spk.extend_from_slice(p2tr_program);
    let mut outputs: Vec<(u64, Vec<u8>)> = vec![(coinbase_value, spk)];
    let has_witness = witness_commitment.is_some();
    if let Some(wc) = witness_commitment {
        // OP_RETURN push( 0xaa21a9ed || commitment )
        let mut payload = vec![0xaa, 0x21, 0xa9, 0xedu8];
        payload.extend_from_slice(wc);
        let mut wscript = vec![0x6au8]; // OP_RETURN
        wscript.extend_from_slice(&push_data(&payload));
        outputs.push((0, wscript));
    }

    // --- non-witness serialization (for txid) ---
    let mut base = Vec::new();
    base.extend_from_slice(&1u32.to_le_bytes()); // version
    base.extend_from_slice(&varint(1)); // vin count
    base.extend_from_slice(&[0u8; 32]); // null prevout hash
    base.extend_from_slice(&0xffff_ffffu32.to_le_bytes()); // prevout index
    base.extend_from_slice(&varint(script_sig.len() as u64));
    base.extend_from_slice(&script_sig);
    base.extend_from_slice(&0xffff_ffffu32.to_le_bytes()); // sequence
    base.extend_from_slice(&varint(outputs.len() as u64));
    for (val, spk) in &outputs {
        base.extend_from_slice(&val.to_le_bytes());
        base.extend_from_slice(&varint(spk.len() as u64));
        base.extend_from_slice(spk);
    }
    base.extend_from_slice(&0u32.to_le_bytes()); // locktime

    let txid = dsha256(&base);
    let mut txid_be = txid;
    txid_be.reverse(); // display/big-endian

    // --- full serialization for the block body ---
    let full_bytes = if has_witness {
        // version | 00 (marker) | 01 (flag) | vin | vout | witness | locktime
        let mut out = Vec::new();
        out.extend_from_slice(&1u32.to_le_bytes());
        out.push(0x00);
        out.push(0x01);
        out.extend_from_slice(&varint(1));
        out.extend_from_slice(&[0u8; 32]);
        out.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        out.extend_from_slice(&varint(script_sig.len() as u64));
        out.extend_from_slice(&script_sig);
        out.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        out.extend_from_slice(&varint(outputs.len() as u64));
        for (val, spk) in &outputs {
            out.extend_from_slice(&val.to_le_bytes());
            out.extend_from_slice(&varint(spk.len() as u64));
            out.extend_from_slice(spk);
        }
        // witness: 1 stack item of 32 zero bytes (reserved value)
        out.extend_from_slice(&varint(1)); // stack item count for the 1 input
        out.extend_from_slice(&varint(32));
        out.extend_from_slice(&[0u8; 32]);
        out.extend_from_slice(&0u32.to_le_bytes()); // locktime
        out
    } else {
        base
    };

    Coinbase { full_bytes, txid_be }
}

/// Merkle root from display-order (big-endian) txid hex strings, coinbase first.
/// Returns the 32-byte root in display/big-endian order (as the header stores it).
fn calculate_merkle_root(txids_be: &[[u8; 32]]) -> [u8; 32] {
    assert!(!txids_be.is_empty());
    // reverse each to little-endian (internal) order
    let mut level: Vec<[u8; 32]> = txids_be
        .iter()
        .map(|h| {
            let mut x = *h;
            x.reverse();
            x
        })
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let mut combined = Vec::with_capacity(64);
            combined.extend_from_slice(&level[i]);
            if i + 1 < level.len() {
                combined.extend_from_slice(&level[i + 1]);
            } else {
                combined.extend_from_slice(&level[i]); // duplicate last
            }
            next.push(dsha256(&combined));
            i += 2;
        }
        level = next;
    }
    let mut root = level[0];
    root.reverse(); // back to display/big-endian
    root
}

fn bits_to_target(bits: u32) -> primitive_types::U256 {
    let exponent = (bits >> 24) & 0xff;
    let mantissa = bits & 0x00ff_ffff;
    let m = primitive_types::U256::from(mantissa);
    if exponent <= 3 {
        m >> (8 * (3 - exponent))
    } else {
        m << (8 * (exponent - 3))
    }
}

// ============================================================================
// Template parsing.
// ============================================================================
struct Template {
    version: u32,
    prev_block: [u8; 32], // display/big-endian, as in getblocktemplate
    bits: u32,
    curtime: u32,
    height: i64,
    coinbase_value: u64,
    aux_flags: Vec<u8>,
    witness_commitment: Option<Vec<u8>>,
    tx_data: Vec<Vec<u8>>,  // raw tx bytes, in template order (no coinbase)
    tx_txids: Vec<[u8; 32]>, // display/big-endian txids, matching tx_data order
}

fn hex_to_bytes(s: &str) -> anyhow::Result<Vec<u8>> {
    Ok(hex::decode(s.trim())?)
}

fn hex_to_32_be(s: &str) -> anyhow::Result<[u8; 32]> {
    let b = hex_to_bytes(s)?;
    if b.len() != 32 {
        anyhow::bail!("expected 32-byte hex, got {} bytes", b.len());
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&b);
    Ok(out)
}

fn parse_template(v: &Value) -> anyhow::Result<Template> {
    // Accept either the full RPC envelope {"result": {...}} or the bare result.
    let t = v.get("result").unwrap_or(v);
    let version = t["version"].as_u64().ok_or_else(|| anyhow::anyhow!("no version"))? as u32;
    let prev_block = hex_to_32_be(
        t["previousblockhash"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("no previousblockhash"))?,
    )?;
    let bits_str = t["bits"].as_str().ok_or_else(|| anyhow::anyhow!("no bits"))?;
    let bits = u32::from_str_radix(bits_str.trim_start_matches("0x"), 16)?;
    let curtime = t["curtime"].as_u64().ok_or_else(|| anyhow::anyhow!("no curtime"))? as u32;
    let height = t["height"].as_i64().ok_or_else(|| anyhow::anyhow!("no height"))?;
    let coinbase_value =
        t["coinbasevalue"].as_u64().ok_or_else(|| anyhow::anyhow!("no coinbasevalue"))?;

    let aux_flags = t
        .get("coinbaseaux")
        .and_then(|a| a.get("flags"))
        .and_then(|f| f.as_str())
        .map(hex_to_bytes)
        .transpose()?
        .unwrap_or_default();

    let witness_commitment = t
        .get("default_witness_commitment")
        .and_then(|w| w.as_str())
        .filter(|s| !s.is_empty())
        .map(hex_to_bytes)
        .transpose()?;

    let mut tx_data = Vec::new();
    let mut tx_txids = Vec::new();
    if let Some(txs) = t.get("transactions").and_then(|x| x.as_array()) {
        for tx in txs {
            let data = tx["data"].as_str().ok_or_else(|| anyhow::anyhow!("tx without data"))?;
            tx_data.push(hex_to_bytes(data)?);
            // prefer explicit txid; fall back to hash; else compute from data
            let txid = tx
                .get("txid")
                .or_else(|| tx.get("hash"))
                .and_then(|x| x.as_str());
            let txid_be = match txid {
                Some(s) => hex_to_32_be(s)?,
                None => {
                    // compute txid from non-witness data (best effort; node normally provides txid)
                    let mut h = dsha256(&hex_to_bytes(data)?);
                    h.reverse();
                    h
                }
            };
            tx_txids.push(txid_be);
        }
    }

    Ok(Template {
        version,
        prev_block,
        bits,
        curtime,
        height,
        coinbase_value,
        aux_flags,
        witness_commitment,
        tx_data,
        tx_txids,
    })
}

/// Build the coinbase + merkle root + incomplete header for a template + address.
fn build_header(t: &Template, addr: &str) -> anyhow::Result<(IncompleteBlockHeader, Coinbase)> {
    let prog = p2tr_program(addr)?;
    let coinbase = build_coinbase(
        t.height,
        t.coinbase_value,
        &prog,
        &t.aux_flags,
        t.witness_commitment.as_deref(),
    );
    let mut txids = Vec::with_capacity(1 + t.tx_txids.len());
    txids.push(coinbase.txid_be);
    txids.extend_from_slice(&t.tx_txids);
    let merkle_root = calculate_merkle_root(&txids);

    let header = IncompleteBlockHeader {
        version: t.version,
        prev_block: t.prev_block,
        merkle_root,
        timestamp: t.curtime,
        nbits: t.bits,
    };
    Ok((header, coinbase))
}

// ============================================================================
// Block assembly (port of pearl_block.py / zk_certificate.py / pearl_header.py).
// ============================================================================
fn proof_commitment(public_data: &[u8]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(4 + public_data.len());
    buf.extend_from_slice(&ZK_CERTIFICATE_VERSION.to_le_bytes());
    buf.extend_from_slice(public_data);
    dsha256(&buf)
}

/// PearlHeader.serialize(): incomplete_header(76) || proof_commitment(32).
fn pearl_header_serialize(header_bytes: &[u8], proof_commitment: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(header_bytes.len() + 32);
    v.extend_from_slice(header_bytes);
    v.extend_from_slice(proof_commitment);
    v
}

/// ZKCertificate.serialize(): version(u32 LE) || header_hash(32) || public_data(164)
///                            || proof_data_len(u32 LE) || proof_data.
fn zk_certificate_serialize(
    header_hash: &[u8; 32],
    public_data: &[u8],
    proof_data: &[u8],
) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + 32 + public_data.len() + 4 + proof_data.len());
    v.extend_from_slice(&ZK_CERTIFICATE_VERSION.to_le_bytes());
    v.extend_from_slice(header_hash);
    v.extend_from_slice(public_data);
    v.extend_from_slice(&(proof_data.len() as u32).to_le_bytes());
    v.extend_from_slice(proof_data);
    v
}

// ============================================================================
// Subcommands.
// ============================================================================
fn read_stdin() -> anyhow::Result<String> {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    Ok(s)
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn cmd_header(args: &[String]) -> anyhow::Result<()> {
    let addr = arg_value(args, "--addr").ok_or_else(|| anyhow::anyhow!("--addr <p2tr> required"))?;
    let tpl_json: Value = serde_json::from_str(&read_template_for_block(args)?)?;
    let t = parse_template(&tpl_json)?;
    let (header, _coinbase) = build_header(&t, &addr)?;
    let header_hex = hex::encode(header.to_bytes());
    let target = bits_to_target(t.bits);
    let mut tb = [0u8; 32];
    target.to_big_endian(&mut tb);
    let out = serde_json::json!({
        "incomplete_header": header_hex,
        "target": hex::encode(tb),
        "height": t.height,
        "nbits": format!("{:08x}", t.bits),
    });
    println!("{}", out);
    Ok(())
}

fn cmd_block(args: &[String]) -> anyhow::Result<()> {
    let addr = arg_value(args, "--addr").ok_or_else(|| anyhow::anyhow!("--addr <p2tr> required"))?;
    // PlainProof is ~60-140 KB of base64 — too large for an argv string on most
    // platforms, so it is read from --ppfile (or --pp for tiny test inputs).
    // Template is small; --tpl <file> or stdin.
    let pp_b64 = read_plain_proof(args)?;
    let tpl_json: Value = serde_json::from_str(&read_template_for_block(args)?)?;
    let t = parse_template(&tpl_json)?;
    let (header, coinbase) = build_header(&t, &addr)?;

    // PlainProof
    let raw = STANDARD.decode(pp_b64.trim().as_bytes())?;
    let plain_proof: PlainProof = bincode::deserialize(&raw)?;

    // ZK proof
    let mut cache = CircuitCache::default();
    eprintln!("zkprove: generating ZK proof (plonky2)…");
    let result = zk_prove_plain_proof(header, &plain_proof, &mut cache, false)?;
    if result.public_data.len() != PUBLICDATA_SIZE {
        anyhow::bail!("public_data size {} != {}", result.public_data.len(), PUBLICDATA_SIZE);
    }
    eprintln!(
        "zkprove: ZK proof ready (public_data={} proof_data={} bytes)",
        result.public_data.len(),
        result.proof_data.len()
    );

    // Header with proof_commitment, header_hash, certificate, block.
    let commitment = proof_commitment(&result.public_data);
    let header_bytes = header.to_bytes();
    let pearl_header = pearl_header_serialize(&header_bytes, &commitment);
    let header_hash = dsha256(&pearl_header);
    let zk_cert = zk_certificate_serialize(&header_hash, &result.public_data, &result.proof_data);

    // PearlBlock.serialize(): ZK_CERT | HEADER | TX_COUNT(varint) | TXNS (coinbase first)
    let ntx = 1 + t.tx_data.len();
    let mut block = Vec::new();
    block.extend_from_slice(&zk_cert);
    block.extend_from_slice(&pearl_header);
    block.extend_from_slice(&varint(ntx as u64));
    block.extend_from_slice(&coinbase.full_bytes);
    for tx in &t.tx_data {
        block.extend_from_slice(tx);
    }

    println!("{}", hex::encode(block));
    Ok(())
}

// For `block`, the template JSON is on stdin (or --tpl <file>).
fn read_template_for_block(args: &[String]) -> anyhow::Result<String> {
    if let Some(path) = arg_value(args, "--tpl") {
        Ok(std::fs::read_to_string(path)?)
    } else {
        read_stdin()
    }
}

/// PlainProof base64 from --pp <b64> (test only), --ppfile <path>, or stdin.
fn read_plain_proof(args: &[String]) -> anyhow::Result<String> {
    if let Some(s) = arg_value(args, "--pp") {
        Ok(s)
    } else if let Some(path) = arg_value(args, "--ppfile") {
        Ok(std::fs::read_to_string(path)?)
    } else {
        read_stdin()
    }
}

/// `prove` — validation/debug helper: take a complete incomplete-header (152hex)
/// + PlainProof (base64) and run the full STARK/recursion prover, reporting the
/// proof sizes and certificate commitment. No coinbase/block assembly (no
/// template needed), so it can be driven directly by `gen_golden`.
fn cmd_prove(args: &[String]) -> anyhow::Result<()> {
    let header_hex =
        arg_value(args, "--header").ok_or_else(|| anyhow::anyhow!("--header <152hex> required"))?;
    let header_bytes = hex::decode(header_hex.trim())?;
    let header = IncompleteBlockHeader::from_bytes(&header_bytes)?;
    let pp_b64 = match arg_value(args, "--pp") {
        Some(s) => s,
        None => read_stdin()?,
    };
    let raw = STANDARD.decode(pp_b64.trim().as_bytes())?;
    let plain_proof: PlainProof = bincode::deserialize(&raw)?;

    let mut cache = CircuitCache::default();
    eprintln!("zkprove prove: generating ZK proof (plonky2)…");
    let result = zk_prove_plain_proof(header, &plain_proof, &mut cache, true)?;
    let commitment = proof_commitment(&result.public_data);
    eprintln!(
        "public_data={} proof_data={} commitment={}",
        result.public_data.len(),
        result.proof_data.len(),
        hex::encode(commitment),
    );
    if result.public_data.len() != PUBLICDATA_SIZE {
        anyhow::bail!("public_data size mismatch");
    }
    println!("PROVE OK proof_data_len={}", result.proof_data.len());
    Ok(())
}

/// `gen` — CPU-mine a golden-config PlainProof for a GIVEN incomplete header
/// (152hex). Validation/dry-run only: lets `header -> gen -> block` exercise the
/// whole SOLO assembly locally without a GPU or a live node. Uses the same
/// CPU-winnable toy config as `gen_golden` (m=6144,n=4096,k=2240,rank=128).
fn cmd_gen(args: &[String]) -> anyhow::Result<()> {
    let header_hex =
        arg_value(args, "--header").ok_or_else(|| anyhow::anyhow!("--header <152hex> required"))?;
    let header_bytes = hex::decode(header_hex.trim())?;
    let header = IncompleteBlockHeader::from_bytes(&header_bytes)?;

    let rank: u16 = 128;
    let m: usize = 6144;
    let n: usize = 4096;
    let k: usize = (16 * rank as usize).max(1024) + 192; // 2240
    let config = MiningConfiguration {
        common_dim: k as u32,
        rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73])?,
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73])?,
        reserved: MiningConfiguration::RESERVED_VALUE,
    };
    eprintln!("zkprove gen: CPU-mining golden config for the given header…");
    let plain_proof = cpu_mine(m, n, k, header, config, None, false)?;
    let bytes = bincode::serialize(&plain_proof)?;
    println!("{}", STANDARD.encode(bytes));
    Ok(())
}

/// Validate the coinbase byte layout against the node-golden vector from
/// pearl-gateway tests/test_blockchain_utils.py (no witness commitment).
fn cmd_selftest() -> anyhow::Result<()> {
    // height=2591, value=297454395584, aux flags from template, P2TR program from
    // the node coinbase's scriptPubKey, no witness commitment.
    let node_hex = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff10021f0a000b2f503253482f627463642fffffffff01c0e0a941450000002251208635cb51e0601a2f55b17b1ba41b21a511b3753a0bf4610bd52eb1a15d69a28100000000";
    let prog = hex::decode("8635cb51e0601a2f55b17b1ba41b21a511b3753a0bf4610bd52eb1a15d69a281")?;
    let mut prog32 = [0u8; 32];
    prog32.copy_from_slice(&prog);
    let aux = hex::decode("0b2f503253482f627463642f")?;
    let cb = build_coinbase(2591, 297454395584, &prog32, &aux, None);
    let got = hex::encode(&cb.full_bytes);
    if got != node_hex {
        eprintln!("EXPECT: {node_hex}");
        eprintln!("GOT   : {got}");
        anyhow::bail!("coinbase mismatch vs node-golden vector");
    }
    eprintln!("coinbase: MATCHES node-golden vector ✓");

    // SHA-256 known-answer ("abc").
    let h = hex::encode(sha256(b"abc"));
    assert_eq!(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    eprintln!("sha256(\"abc\"): KAT ✓");

    // bits_to_target sanity: difficulty-1 bits 0x1d00ffff -> 0x00000000FFFF0000...0000
    let tgt = bits_to_target(0x1d00ffff);
    let mut tb = [0u8; 32];
    tgt.to_big_endian(&mut tb);
    assert_eq!(&tb[0..8], &[0, 0, 0, 0, 0xff, 0xff, 0, 0]);
    eprintln!("bits_to_target(0x1d00ffff): ✓");

    // merkle of a single coinbase == its txid (display order).
    let root = calculate_merkle_root(&[cb.txid_be]);
    assert_eq!(root, cb.txid_be);
    eprintln!("merkle(single)==txid: ✓");

    println!("SELFTEST OK");
    Ok(())
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  zkprove header   --addr <p2tr>                 < template.json\n  zkprove block    --addr <p2tr> --pp <b64>      < template.json\n  zkprove selftest"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }
    let res = match args[1].as_str() {
        "header" => cmd_header(&args[2..]),
        "block" => cmd_block(&args[2..]),
        "prove" => cmd_prove(&args[2..]),
        "gen" => cmd_gen(&args[2..]),
        "selftest" => cmd_selftest(),
        _ => usage(),
    };
    if let Err(e) = res {
        eprintln!("zkprove error: {e:#}");
        std::process::exit(1);
    }
}
