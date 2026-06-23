#!/usr/bin/env bash
# GPU smoke test for the production geometry.
#
# Historical versions used the tiny golden config, but the live CUTLASS v2
# production kernel is specialized for the real pool dimensions (h=8, w=16).
# Golden config is still useful for CPU/oracle tests; GPU launch correctness must
# use --cfg real plus an easy target so every supported production kernel can find
# a win quickly and exercise POSTCHECK/proof emission.
set -uo pipefail
cd "$(dirname "$0")/build"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EASY="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
SEED="${SEED:-777}"

echo "=== run: plainproof_gen --cfg real --mine 1 --target EASY ==="
./plainproof_gen --cfg real --mine 1 --target "${EASY}" "${SEED}" \
  > /tmp/pp.b64 2> /tmp/pp.log
RC=$?
echo "rc=${RC}  b64_bytes=$(wc -c < /tmp/pp.b64)"
echo "--- solver log (tail) ---"
tail -20 /tmp/pp.log

if [ "${RC}" -eq 0 ] && [ -s /tmp/pp.b64 ] && \
   grep -qE 'POSTCHECK .*ok=1|POSTCHECK.*ok=1' /tmp/pp.log; then
  echo "SMOKE PASS: real-cfg GPU path found a win, POSTCHECK ok=1, emitted a proof."
else
  echo "SMOKE FAIL: rc=${RC} (see log above; expected POSTCHECK ok=1 and non-empty proof)"
  exit 1
fi
