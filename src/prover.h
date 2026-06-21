// prover.h — reusable entry point into the proven PlainProof pipeline.
//
// The full A/B -> blake3 commitments -> noise -> jackpot search (CPU /
// fused WMMA tensor-core) -> Merkle -> bincode PlainProof pipeline lives in
// plainproof_gen.cpp.  Historically it was only reachable through that file's
// main().  This header exposes it as a plain function so the unified
// `kan` / legacy `pearl-miner` driver (pool + solo) can drive it
// in-process, while the standalone `plainproof_gen` CLI keeps working unchanged.
//
// plainproof_gen.cpp is compiled with -DPROVER_LIB inside the miner (drops its
// own main); the CLI build compiles it without the flag (keeps main).
#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

struct MineParams {
  // 152 hex chars = 76-byte IncompleteBlockHeader.  Empty => the built-in golden
  // self-test header (CLI default).
  std::string header_hex;
  // false => golden toy config (CPU-winnable); true => REAL kryptex network
  // config (m=n=131072, k=4096, rank=256).
  bool real_cfg = false;
  // 64 hex chars = 32-byte target, BIG-endian (pool/solo share/block target).
  // Empty => derive the bound from the header's nbits.
  std::string target_hex;
  uint64_t maxdraws = 10000000ULL;  // redraw cap for `mine`
  bool use_tc = false;              // force tensor-core kernel for the single-shot self-test path
  bool mine = false;                // internal redraw loop (fresh A/B every draw); always tensor-core
  bool probe = false;               // emit a structurally-valid proof for tile 0 (no search)
  bool dump = false;                // write /tmp debug artifacts
  bool breakdown = false;           // print per-draw timing breakdown to stderr
  uint64_t seed = 12345;            // RNG seed (per-draw reseeded internally)
};

struct MineResult {
  bool found = false;
  std::string proof_b64;            // base64(bincode(PlainProof)) on success
  std::vector<size_t> win_rows;
  std::vector<size_t> win_cols;
  uint64_t draws = 0;               // draws performed (mine mode)
  double elapsed_s = 0.0;
  double work_per_draw = 0.0;        // SRBMiner-MULTI/lpminer-compatible PRL work units per draw
};

// Run the pipeline once.  Returns 0 on a found+postchecked win (out.proof_b64
// set), 2 if no winning tile within maxdraws, 4 if a GPU-reported win fails the
// CPU postcheck against the active target, 1 on input error.
// `stop` (optional) is polled in the redraw loop so pool/solo can abort the
// current job the instant a newer one arrives; pass nullptr to disable.
int mine_plain_proof(const MineParams& params, MineResult& out, std::atomic<bool>* stop);
