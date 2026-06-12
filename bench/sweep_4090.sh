#!/usr/bin/env bash
# 4090 optimization sweep: GROUPM + SMALL_TILE
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 4090 Kernel Optimization Sweep ==="
echo "Baseline: 260 TH/s (GROUPM=8, TB 128×256)"
echo "Target: 90% roofline ≈ 315+ TH/s"
echo ""

# Test 1: GROUPM sweep (low effort, 5-10% gain expected)
echo "--- Test 1: GROUPM sweep (72MB L2 vs 3090's 6MB) ---"
for G in 16 32 64; do
  echo "Building GROUPM=$G..."
  GROUPM=$G ./build.sh >/dev/null 2>&1

  echo -n "  Testing: "
  timeout 60 ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | \
    grep -oP '\d+ draws.*?(\d+\.\d+) ms/draw' | tail -1 || echo "TIMEOUT"
done

# Test 2: SMALL_TILE (medium effort, 30-50% gain expected)
echo ""
echo "--- Test 2: SMALL_TILE (2 TB/SM occupancy) ---"
echo "Building TB 64×128 (vs 128×256)..."
SMALL_TILE=1 GROUPM=8 ./build.sh >/dev/null 2>&1

echo -n "  Testing: "
timeout 60 ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | \
  grep -oP '\d+ draws.*?(\d+\.\d+) ms/draw' | tail -1 || echo "TIMEOUT"

# Test 3: Combined (SMALL_TILE + best GROUPM)
echo ""
echo "--- Test 3: SMALL_TILE + GROUPM=16 ---"
SMALL_TILE=1 GROUPM=16 ./build.sh >/dev/null 2>&1

echo -n "  Testing: "
timeout 60 ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | \
  grep -oP '\d+ draws.*?(\d+\.\d+) ms/draw' | tail -1 || echo "TIMEOUT"

echo ""
echo "Restore baseline for live mining..."
GROUPM=8 ./build.sh >/dev/null 2>&1
echo "Done."
