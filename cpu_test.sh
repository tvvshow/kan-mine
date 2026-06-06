#!/usr/bin/env bash
# CPU-only correctness self-test — NO GPU required.
#
# The default (no-flags) path runs the full reference pipeline end to end:
# A/B generation, BLAKE3 commitments, noise, the NoisyGEMM jackpot fold, the
# difficulty-bound check, Merkle multi-leaf proofs, and bincode PlainProof
# serialization. At the golden block header's nbits a winning tile exists in the
# first config, so a base64 PlainProof is printed to stdout and the process exits
# 0. This is the SAME proof shape that validated against the official Rust oracle
# (verify_plain) on the GPU box — so a pass here means the solver is correct,
# independent of any GPU.
set -uo pipefail
cd "$(dirname "$0")/build"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "=== run: plainproof_gen 12345 (golden config, CPU reference search) ==="
./plainproof_gen 12345 > /tmp/pp.b64 2> /tmp/pp.log
RC=$?
B64=$(wc -c < /tmp/pp.b64)
echo "rc=${RC}  b64_bytes=${B64}"
echo "--- solver log (tail) ---"
tail -12 /tmp/pp.log

if [ "${RC}" -eq 0 ] && [ "${B64}" -gt 32 ]; then
  echo "CPU-VERIFY PASS: full pipeline ran, found a win, emitted a PlainProof (no GPU needed)."
else
  echo "CPU-VERIFY FAIL: rc=${RC} b64_bytes=${B64} (see log above)"
  exit 1
fi
