#!/usr/bin/env bash
# GPU validation for the grid-stride/persistent kernel rework:
#   1. REAL config + moderate-nbits header (1D2FFFFF, ~1/512 tiles win): the GPU
#      mine loop wins on draw 1 -> exercises the FULL win path incl. POSTCHECK
#      (CPU re-derivation of the GPU draw). MUST print ok=1 and exit 0.
#      (golden config is h=8,w=8 — the CUTLASS kernel only does h=8,w=16, so
#       real-config is the only GPU-searchable config; same trick verify_run.sh
#       uses for TCREAL.)
#   2. REAL config, default hard header, 3 draws, standard grid -> kernel ms.
#   3. same with TC_PERSIST=1 -> persistent-grid A/B. rc=2 (no win) expected.
set -uo pipefail
cd "$(dirname "$0")/build"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# oracle header with nbits swapped to 1D2FFFFF (same as verify_run.sh REAL_HDR_MOD)
HDR="01000000f9661239d86cd892e31455d6ad6c1a55745ab7d16a63c82143d271f417ca49994f2738ce9c121c22c08598078e168bf4e1b8167b4e6f30fe911d555492a1afacf2e3246affff2f1d"

echo "=== [1/3] real-config GPU win + POSTCHECK (moderate nbits) ==="
./plainproof_gen --cfg real --mine 3 --header "$HDR" > /tmp/g.b64 2> /tmp/g.log; RC=$?
grep -E "MINE WIN|POSTCHECK|FUSED" /tmp/g.log | tail -4
if [ $RC -ne 0 ] || ! grep -q "POSTCHECK.*ok=1" /tmp/g.log; then
  echo "FAIL: win/POSTCHECK (rc=$RC)"; tail -15 /tmp/g.log; exit 1
fi
echo "PASS: POSTCHECK ok=1 (GPU draw == CPU re-derivation)"

run_real() { # $1=TC_PERSIST $2=log ; rc 0 (win) or 2 (no win in 3 draws) both OK
  TC_PERSIST=$1 ./plainproof_gen --cfg real --mine 3 > /dev/null 2> "$2"; local rc=$?
  grep -E "PERSISTENT grid|FUSED" "$2" | tail -4
  [ $rc -eq 0 ] || [ $rc -eq 2 ]
}

echo "=== [2/3] real-config standard grid (3 draws, hard target) ==="
run_real 0 /tmp/std.log || { echo "FAIL: standard run"; tail -10 /tmp/std.log; exit 1; }
echo "=== [3/3] real-config TC_PERSIST=1 (3 draws, hard target) ==="
run_real 1 /tmp/per.log || { echo "FAIL: persistent run"; tail -10 /tmp/per.log; exit 1; }

S=$(grep -oP 'FUSED \d+ tiles, \K[0-9.]+(?= ms)' /tmp/std.log | tail -1)
P=$(grep -oP 'FUSED \d+ tiles, \K[0-9.]+(?= ms)' /tmp/per.log | tail -1)
echo "=== RESULT kernel ms/draw: standard=${S:-?}  persistent=${P:-?} ==="
echo "GPU-PERSIST-AB DONE"
