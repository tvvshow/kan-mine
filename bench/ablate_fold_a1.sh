#!/usr/bin/env bash
# Phase A1 (fold direct-store) A/B + correctness gate.
#
# tc_cutlass_v2.cu now stores the fold transcript word DIRECTLY on its first
# (only, for REAL k/rank==16) touch instead of read-rotl13-xor-write. The old
# RMW path is still emitted behind -DFOLD_RMW_ALWAYS, so this is a one-flag A/B:
#
#   A1   = default build               (direct store)
#   RMW  = NVCC_EXTRA=-DFOLD_RMW_ALWAYS (pre-A1 baseline)
#
# Correctness gate = POSTCHECK ok=1 on BOTH (the CPU independently recomputes the
# GPU's winning jackpot tile; a corrupt A1 transcript would fail it). It is NOT
# proof-byte-identity: an all-wins target makes the winning tile racy, so proofs
# differ run-to-run even RMW-vs-RMW. A1 is an algebraically-equal strength
# reduction, validated empirically by ok=1 + measured by the kernel Δ%.
#
# Usage (on a GPU box with CUTLASS at $CUTLASS_HOME or ~/cutlass):
#   ARCH=sm_120 bash bench/ablate_fold_a1.sh
#   ARCH=sm_89  bash bench/ablate_fold_a1.sh   # 4090 / L40
set -uo pipefail
cd "$(dirname "$0")/.."

ARCH="${ARCH:-sm_120}"
SEED="${SEED:-777}"
EASY="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"  # bound=2^256-1 -> draw 1 wins -> POSTCHECK fires + proof emitted
HARD="0000000000000000000000000000000000000000000000000000000000000001"  # bound=1 -> no early win -> full sweep -> clean kernel timing
NDRAW="${NDRAW:-3}"
PG=./build/plainproof_gen

echo "=== Phase A1 fold direct-store A/B (ARCH=$ARCH SEED=$SEED) ==="

run_variant() {  # $1=label  $2=NVCC_EXTRA
  local label="$1" extra="$2"
  echo ""
  echo "--- build [$label] (NVCC_EXTRA='$extra') ---"
  if ! NVCC_EXTRA="$extra" ARCH="$ARCH" ./build.sh >/tmp/a1_build_$label.log 2>&1; then
    echo "  BUILD FAILED [$label] (see /tmp/a1_build_$label.log)"; tail -5 /tmp/a1_build_$label.log; return 1
  fi
  grep -qE "tc_cutlass_v2" /tmp/a1_build_$label.log || { echo "  ✗ fell back to tc_block (CUTLASS missing) — abort"; return 1; }

  # --- correctness: easy target -> draw-1 winner -> POSTCHECK (CPU re-verifies
  # the GPU's winning tile). NOTE: an all-wins target makes the winning tile
  # RACY, so proof bytes differ run-to-run — POSTCHECK ok=1 is the real gate. ---
  $PG --cfg real --mine 1 --target "$EASY" "$SEED" >/tmp/a1_proof_$label.b64 2>/tmp/a1_corr_$label.log
  local postck; postck=$(grep -E "POSTCHECK" /tmp/a1_corr_$label.log | grep -oE "ok=[01]" | tail -1)
  local sha;    sha=$(sha256sum /tmp/a1_proof_$label.b64 | awk '{print $1}')
  echo "  correctness: POSTCHECK ${postck:-<none>}   proof.sha256=${sha:0:16}…"

  # --- kernel timing: hard target -> no early-out -> full sweep every draw ---
  $PG --cfg real --mine "$NDRAW" --target "$HARD" --breakdown "$SEED" >/dev/null 2>/tmp/a1_time_$label.log
  local mss; mss=$(grep -oE "FUSED [0-9]+ tiles, [0-9.]+ ms" /tmp/a1_time_$label.log | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+")
  local last; last=$(echo "$mss" | tail -1)
  echo "  kernel FUSED ms/draw (all $NDRAW): $(echo $mss | tr '\n' ' ')-> steady=$last ms"

  # export results to caller via globals
  eval "MS_$label=$last"; eval "SHA_$label='$sha'"; eval "OK_$label='$postck'"
}

run_variant RMW "-DFOLD_RMW_ALWAYS" || exit 1
run_variant A1  ""                   || exit 1

echo ""
echo "=== verdict ==="
echo "  POSTCHECK  RMW=${OK_RMW:-?}  A1=${OK_A1:-?}"
echo "  proof sha  RMW=${SHA_RMW:0:16}…  A1=${SHA_A1:0:16}… (racy winner -> differ; not a gate)"

fail=0
[ "$OK_RMW" = "ok=1" ] || { echo "  ✗ RMW POSTCHECK not ok=1"; fail=1; }
[ "$OK_A1"  = "ok=1" ] || { echo "  ✗ A1  POSTCHECK not ok=1"; fail=1; }
[ "$fail" = 0 ] && echo "  ✓ A1 + RMW both POSTCHECK ok=1 (A1 jackpot transcript verified on GPU)"

if [ "$fail" = 0 ]; then
  # TH/s = work_per_draw / ms ; REAL work_per_draw = 70.368744e12, /1e12 /(ms/1e3) = 70368.744/ms
  ths_rmw=$(awk "BEGIN{printf \"%.1f\", 70368.744/$MS_RMW}")
  ths_a1=$(awk "BEGIN{printf \"%.1f\", 70368.744/$MS_A1}")
  delta=$(awk "BEGIN{printf \"%+.2f\", ($MS_RMW-$MS_A1)/$MS_RMW*100}")
  echo ""
  echo "  RMW : $MS_RMW ms = $ths_rmw TH/s"
  echo "  A1  : $MS_A1 ms = $ths_a1 TH/s"
  echo "  Δ   : ${delta}% kernel (positive = A1 faster)"
  echo ""
  echo "  GATE: deploy A1 if POSTCHECK ok=1 on both AND Δ stably positive AND no new ptxas spill."
fi
exit $fail
