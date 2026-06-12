#!/usr/bin/env bash
# 5090 完整性能测试：baseline + v2.0.0 所有路径 + Phase 8
set -e

cd "$(dirname "$0")/.."

echo "=== RTX 5090 Complete Benchmark Suite ==="
echo "Date: $(date)"
echo "Roofline estimate: ~450-500 TH/s (sm_120, 5th gen tensor cores)"
echo ""

LOG="/tmp/5090_benchmark.log"
> $LOG

test_config() {
  local name="$1"
  local env_vars="$2"

  echo "--- Testing: $name ---"
  echo "  Config: $env_vars"

  # Build
  eval "export $env_vars"
  ./build.sh 2>&1 | grep -E "CUTLASS at|BUILD OK" || { echo "  BUILD FAILED"; return 1; }

  # Run 3 draws
  echo -n "  Running 3 draws: "
  timeout 180 ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | tee /tmp/test.log | \
    grep -oP '\d+ draws.*?(\d+\.\d+) ms/draw' | tail -1 || echo "TIMEOUT"

  # Extract results
  if grep -q "POSTCHECK ok=1" /tmp/test.log; then
    MS=$(grep -oP '\d+\.\d+ ms/draw' /tmp/test.log | tail -1 | awk '{print $1}')
    THS=$(echo "scale=1; 70000000 / $MS / 1000" | bc)
    echo "  ✓ $MS ms/draw = $THS TH/s"
    echo "$name,$MS,$THS" >> $LOG
  else
    echo "  ✗ POSTCHECK failed"
    echo "$name,FAILED,0" >> $LOG
  fi

  # Cleanup
  unset $(echo $env_vars | grep -oP '\w+(?==)')
  echo ""
}

# Baseline
test_config "Baseline (GROUPM=8, 128×256)" ""

# v2.0.0 Path 1: GROUPM tuning
test_config "GROUPM=16" "GROUPM=16"
test_config "GROUPM=32" "GROUPM=32"

# v2.0.0 Path 2: SMALL_TILE
test_config "SMALL_TILE (64×128)" "SMALL_TILE=1"

# v2.0.0 Path 3: Combined
test_config "SMALL_TILE + GROUPM=16" "SMALL_TILE=1 GROUPM=16"
test_config "SMALL_TILE + GROUPM=32" "SMALL_TILE=1 GROUPM=32"

# Find best v2.0.0 config
BEST_V2=$(sort -t, -k2 -n $LOG | grep -v FAILED | head -1)
BEST_NAME=$(echo $BEST_V2 | cut -d, -f1)
BEST_MS=$(echo $BEST_V2 | cut -d, -f2)
BEST_THS=$(echo $BEST_V2 | cut -d, -f3)

echo "=== v2.0.0 Best: $BEST_NAME ==="
echo "  $BEST_MS ms/draw = $BEST_THS TH/s"
echo ""

# Phase 8: Persistent scheduler on best v2.0.0 config
BEST_ENV=$(grep "^$BEST_NAME," $LOG | cut -d, -f1 | sed 's/.*(\(.*\))/\1/' | tr ' ' '\n' | grep = | tr '\n' ' ')
if [ -n "$BEST_ENV" ]; then
  test_config "Phase 8: Persistent ($BEST_NAME)" "$BEST_ENV PERSISTENT=1"
fi

# Summary
echo "=== Summary ==="
cat $LOG | column -t -s,

# Recommendation
echo ""
echo "=== Recommendation ==="
FINAL_BEST=$(sort -t, -k2 -n $LOG | grep -v FAILED | head -1)
echo "Deploy: $(echo $FINAL_BEST | cut -d, -f1)"
echo "  $(echo $FINAL_BEST | cut -d, -f2) ms/draw = $(echo $FINAL_BEST | cut -d, -f3) TH/s"

# Restore baseline
./build.sh >/dev/null 2>&1
echo ""
echo "Baseline restored. Full log: $LOG"
