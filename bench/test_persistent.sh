#!/usr/bin/env bash
# Phase 8 persistent scheduler benchmark
# Compare standard vs persistent on the SAME configuration
set -e

cd "$(dirname "$0")/.."

echo "=== Phase 8 Persistent Scheduler Benchmark ==="
echo "Compares standard grid-stride vs persistent scheduler"
echo ""

# Test baseline first (must be v2.0.0 optimized, e.g. SMALL_TILE=1)
BASELINE="${BASELINE:-SMALL_TILE=1 GROUPM=16}"
echo "Baseline config: ${BASELINE}"
echo ""

# --- Test 1: Standard scheduler (v2.0.0) ---
echo "--- [1/2] Standard scheduler ---"
export ${BASELINE}
unset PERSISTENT
./build.sh 2>&1 | grep -E "CUTLASS at|BUILD OK" || { echo "BUILD FAILED"; exit 1; }

echo -n "Running 3 draws: "
timeout 120 ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | \
  tee /tmp/standard.log | grep -oP '\d+ draws.*?(\d+\.\d+) ms/draw' | tail -1 || echo "TIMEOUT"

if grep -q "POSTCHECK ok=1" /tmp/standard.log; then
  STANDARD_MS=$(grep -oP '\d+\.\d+ ms/draw' /tmp/standard.log | tail -1 | awk '{print $1}')
  echo "  Standard: ${STANDARD_MS} ms/draw"
else
  echo "  ✗ POSTCHECK failed"
  exit 1
fi

# --- Test 2: Persistent scheduler ---
echo ""
echo "--- [2/2] Persistent scheduler ---"
export ${BASELINE}
export PERSISTENT=1
./build.sh 2>&1 | grep -E "CUTLASS at|BUILD OK" || { echo "BUILD FAILED"; exit 1; }

echo -n "Running 3 draws: "
timeout 120 ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | \
  tee /tmp/persistent.log | grep -oP '\d+ draws.*?(\d+\.\d+) ms/draw' | tail -1 || echo "TIMEOUT"

if grep -q "POSTCHECK ok=1" /tmp/persistent.log; then
  PERSISTENT_MS=$(grep -oP '\d+\.\d+ ms/draw' /tmp/persistent.log | tail -1 | awk '{print $1}')
  echo "  Persistent: ${PERSISTENT_MS} ms/draw"
else
  echo "  ✗ POSTCHECK failed"
  exit 1
fi

# --- Compare results ---
echo ""
echo "=== Results ==="
echo "Standard:   ${STANDARD_MS} ms/draw"
echo "Persistent: ${PERSISTENT_MS} ms/draw"

if command -v bc >/dev/null 2>&1; then
  SPEEDUP=$(echo "scale=2; ($STANDARD_MS / $PERSISTENT_MS - 1) * 100" | bc)
  if (( $(echo "$SPEEDUP > 0" | bc -l) )); then
    echo "Speedup:    +${SPEEDUP}% ✅"
    if (( $(echo "$SPEEDUP >= 5" | bc -l) )); then
      echo ""
      echo "✅ DEPLOY: Persistent scheduler shows >5% improvement"
      echo "   Use: PERSISTENT=1 ${BASELINE} ./build.sh"
    else
      echo ""
      echo "🤔 Marginal: <5% improvement, consider deployment cost"
    fi
  else
    SLOWDOWN=$(echo "scale=2; -$SPEEDUP" | bc)
    echo "Slowdown:   -${SLOWDOWN}% ❌"
    echo ""
    echo "❌ REJECT: Persistent scheduler is slower, keep standard"
  fi
else
  echo "(install bc for speedup calculation)"
fi

# Restore baseline
unset PERSISTENT
export ${BASELINE}
./build.sh >/dev/null 2>&1
echo ""
echo "Restored baseline build"
