#!/usr/bin/env bash
# Persistent scheduler benchmark — RUNTIME toggle (TC_PERSIST=1), ONE build.
# The grid-stride kernel in tc_cutlass_v2.cu runs identically either way;
# TC_PERSIST only changes the launch grid (524k one-trip blocks vs N_SM
# persistent blocks looping over tile ids).
set -e
cd "$(dirname "$0")/.."

BASELINE="${BASELINE:-GROUPM=128}"
echo "=== Persistent Scheduler Benchmark (runtime toggle) ==="
echo "Build config: ${BASELINE}"
export ${BASELINE}
./build.sh 2>&1 | grep -E "CUTLASS at|BUILD OK" || { echo "BUILD FAILED"; exit 1; }

run_one() {  # $1 = label, $2 = TC_PERSIST value, $3 = logfile
  echo "--- ${1} ---"
  TC_PERSIST=${2} timeout 300 ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | \
    tee "${3}" | grep -E "TH/s|POSTCHECK" | tail -3
  grep -q "POSTCHECK ok=1" "${3}" || { echo "  ✗ POSTCHECK failed (${1})"; exit 1; }
}

run_one "Standard (full grid)"      0 /tmp/standard.log
run_one "Persistent (TC_PERSIST=1)" 1 /tmp/persistent.log

S=$(grep -oP 'FUSED .*? \K[0-9.]+(?= ms)' /tmp/standard.log   | tail -1)
P=$(grep -oP 'FUSED .*? \K[0-9.]+(?= ms)' /tmp/persistent.log | tail -1)
echo ""
echo "=== Results (kernel ms, last draw) ==="
echo "Standard:   ${S} ms"
echo "Persistent: ${P} ms"
if command -v bc >/dev/null 2>&1 && [ -n "$S" ] && [ -n "$P" ]; then
  D=$(echo "scale=2; ($S / $P - 1) * 100" | bc)
  echo "Delta:      ${D}%  (positive = persistent faster)"
  echo ""
  echo "Deploy persistent by exporting TC_PERSIST=1 before launching kan — no rebuild."
fi
