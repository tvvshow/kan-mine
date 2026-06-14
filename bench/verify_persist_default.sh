#!/usr/bin/env bash
# Verify TC_PERSIST is now ON BY DEFAULT (set 2026-06-14), with TC_PERSIST=0 as
# the escape hatch. Gates: (a) default build persists with NO env var + the
# "[default]" marker, (b) POSTCHECK ok=1 both ways, (c) TC_PERSIST=0 restores the
# full one-trip grid (no PERSISTENT line).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
EASY=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
HARD=0000000000000000000000000000000000000000000000000000000000000001

echo "=== rebuild default (A1, persistent-default) ==="
ARCH=sm_86 ./build.sh >/tmp/vp_build.log 2>&1 || { echo BUILD-FAIL; tail -8 /tmp/vp_build.log; exit 1; }
echo "  build ok"

echo "=== default launch (NO env) — expect 'PERSISTENT grid ... [default]' + ok=1 ==="
./build/plainproof_gen --cfg real --mine 1 --target "$EASY" --breakdown 777 2>&1 \
  | grep -E "PERSISTENT grid|POSTCHECK" | grep -oE "PERSISTENT grid [0-9]+ blocks \([0-9]+ SM x [0-9]+/SM\)\[?d?e?f?a?u?l?t?\]?|ok=[01]" | head -3
./build/plainproof_gen --cfg real --mine 1 --target "$EASY" --breakdown 777 2>&1 | grep -oE "\[default\]" | head -1

echo "=== TC_PERSIST=0 (escape hatch) — expect NO persistent line, full grid, ok=1 ==="
TC_PERSIST=0 ./build/plainproof_gen --cfg real --mine 1 --target "$EASY" --breakdown 777 2>&1 \
  | grep -cE "PERSISTENT grid" | sed 's/^/  persistent-lines=/'
TC_PERSIST=0 ./build/plainproof_gen --cfg real --mine 1 --target "$EASY" --breakdown 777 2>&1 | grep -oE "ok=[01]" | tail -1 | sed 's/^/  POSTCHECK /'

echo "=== quick timing (3 hard draws each) ==="
d=$(./build/plainproof_gen --cfg real --mine 3 --target "$HARD" --breakdown 777 2>&1 | grep -oE "FUSED [0-9]+ tiles, [0-9.]+ ms" | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+" | tail -1)
z=$(TC_PERSIST=0 ./build/plainproof_gen --cfg real --mine 3 --target "$HARD" --breakdown 777 2>&1 | grep -oE "FUSED [0-9]+ tiles, [0-9.]+ ms" | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+" | tail -1)
awk "BEGIN{printf \"  default(persist)=%s ms (%.1f TH/s) | TC_PERSIST=0=%s ms (%.1f TH/s)\n\",\"$d\",70368.744/$d,\"$z\",70368.744/$z}"
