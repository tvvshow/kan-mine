#!/usr/bin/env bash
# GROUPM sweep on 4090 — test L2 optimization for 72MB cache
set -euo pipefail
cd "$(dirname "$0")"

echo "=== GROUPM sweep benchmark for 4090 L2 (72MB) ==="
echo ""

for G in 8 16 32 64; do
  echo "Building with GROUPM=$G..."
  GROUPM=$G ./build.sh >/dev/null 2>&1 || { echo "FAIL"; continue; }

  echo -n "Testing GROUPM=$G: "
  # 运行10次draw测速
  timeout 30 ./build/plainproof_gen --cfg real --mine --batch 10 2>&1 | \
    grep -oP 'draw \d+.*?(\d+\.\d+) ms' | tail -1 || echo "TIMEOUT"
done

echo ""
echo "Expected: larger GROUPM may improve on 4090's 72MB L2 vs 3090's 6MB"
