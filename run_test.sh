#!/usr/bin/env bash
# Smoke test: at the golden block header's nbits a jackpot win exists within a few
# draws (the difficulty is easy), so `--mine N` should exit 0 and emit a base64
# PlainProof. This proves the dp4a path builds AND runs on the CI GPU.
# (Full oracle VALID check requires building the official Rust verifier — added later.)
set -uo pipefail
cd "$(dirname "$0")/build"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "=== run: plainproof_gen --mine 20 (golden nbits) ==="
./plainproof_gen --mine 20 > /tmp/pp.b64 2> /tmp/pp.log
RC=$?
echo "rc=${RC}  b64_bytes=$(wc -c < /tmp/pp.b64)"
echo "--- solver log (tail) ---"
tail -8 /tmp/pp.log

if [ "${RC}" -eq 0 ] && [ -s /tmp/pp.b64 ]; then
  echo "SMOKE PASS: solver ran on GPU, found a win, emitted a proof."
else
  echo "SMOKE FAIL: rc=${RC} (see log above)"
  exit 1
fi
