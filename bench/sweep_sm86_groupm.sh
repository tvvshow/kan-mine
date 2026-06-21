#!/usr/bin/env bash
# Sweep sm_86 CUTLASS launch/raster parameters without changing production
# behavior.
#
# This is an analysis helper only.  It uses the canonical build.sh entry point,
# runs a POSTCHECK smoke for each build, then measures controlled hard-target
# mining with TC_TIMING=1 so logs contain prep+gather/search/total split.
#
# Example:
#   GROUPMS="4 8 16 32" KSTAGES_LIST="3 4" PERSIST_MODES="1 0" NDRAW=12 \
#     CUTLASS_HOME=/root/cutlass bash bench/sweep_sm86_groupm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GROUPMS="${GROUPMS:-4 8 12 16 24 32}"
PERSIST_MODES="${PERSIST_MODES:-1 0}"
NDRAW="${NDRAW:-12}"
SEED="${SEED:-999}"
KSTAGES_LIST="${KSTAGES_LIST:-${KSTAGES:-3}}"
ARCH="${ARCH:-sm_86}"
CUTLASS_HOME="${CUTLASS_HOME:-$HOME/cutlass}"
OUTDIR="${OUTDIR:-bench/results/sm86_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
EASY="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
HARD="0000000000000000000000000000000000000000000000000000000000000001"

mkdir -p "$OUTDIR"

echo "=== sm86 GROUPM sweep ===" | tee "$OUTDIR/summary.txt"
echo "outdir=$OUTDIR arch=$ARCH groupms=[$GROUPMS] kstages=[$KSTAGES_LIST] persist=[$PERSIST_MODES] ndraw=$NDRAW seed=$SEED" | tee -a "$OUTDIR/summary.txt"
echo "cutlass=$CUTLASS_HOME" | tee -a "$OUTDIR/summary.txt"
echo | tee -a "$OUTDIR/summary.txt"

if [ ! -d "$CUTLASS_HOME/include/cutlass" ]; then
  echo "ERROR: CUTLASS headers not found at $CUTLASS_HOME/include/cutlass" | tee -a "$OUTDIR/summary.txt"
  exit 2
fi

for kst in $KSTAGES_LIST; do
for g in $GROUPMS; do
  echo "=== build GROUPM=$g KSTAGES=$kst ===" | tee -a "$OUTDIR/summary.txt"
  ARCH="$ARCH" GROUPM="$g" KSTAGES="$kst" CUTLASS_HOME="$CUTLASS_HOME" WITH_AB=0 bash build.sh \
    >"$OUTDIR/build_g${g}_k${kst}.log" 2>&1

  echo "  smoke POSTCHECK ..." | tee -a "$OUTDIR/summary.txt"
  if ! TC_PERSIST=1 ./build/plainproof_gen --cfg real --mine 1 --target "$EASY" --tc "$SEED" \
      >"$OUTDIR/smoke_g${g}_k${kst}.b64" 2>"$OUTDIR/smoke_g${g}_k${kst}.log"; then
    echo "  GROUPM=$g KSTAGES=$kst smoke failed" | tee -a "$OUTDIR/summary.txt"
    tail -80 "$OUTDIR/smoke_g${g}_k${kst}.log" | tee -a "$OUTDIR/summary.txt"
    continue
  fi
  if ! grep -q 'POSTCHECK .*ok=1' "$OUTDIR/smoke_g${g}_k${kst}.log"; then
    echo "  GROUPM=$g KSTAGES=$kst smoke missing POSTCHECK ok=1" | tee -a "$OUTDIR/summary.txt"
    tail -80 "$OUTDIR/smoke_g${g}_k${kst}.log" | tee -a "$OUTDIR/summary.txt"
    continue
  fi

  for p in $PERSIST_MODES; do
    log="$OUTDIR/bench_g${g}_k${kst}_p${p}.log"
    echo "  bench GROUPM=$g KSTAGES=$kst TC_PERSIST=$p ..." | tee -a "$OUTDIR/summary.txt"
    TC_PERSIST="$p" TC_TIMING=1 ./build/plainproof_gen --cfg real --mine "$NDRAW" \
      --target "$HARD" --tc --breakdown "$SEED" >/dev/null 2>"$log" || true

    last_tc="$(grep -E 'tc\(cutlass2\):' "$log" | tail -1 || true)"
    mine_done="$(grep -E 'MINE done:' "$log" | tail -1 || true)"
    abort_line="$(grep -E 'MINE abort|No winning tile|MINE done' "$log" | tail -1 || true)"
    echo "    tc: ${last_tc:-<none>}" | tee -a "$OUTDIR/summary.txt"
    echo "    done: ${mine_done:-$abort_line}" | tee -a "$OUTDIR/summary.txt"
  done
  echo | tee -a "$OUTDIR/summary.txt"
done
done

echo "=== sweep complete ===" | tee -a "$OUTDIR/summary.txt"
echo "summary: $OUTDIR/summary.txt" | tee -a "$OUTDIR/summary.txt"
if command -v python3 >/dev/null 2>&1; then
  python3 bench/parse_tc_sweep.py "$OUTDIR" | tee -a "$OUTDIR/summary.txt" || true
elif command -v python >/dev/null 2>&1; then
  python bench/parse_tc_sweep.py "$OUTDIR" | tee -a "$OUTDIR/summary.txt" || true
fi
