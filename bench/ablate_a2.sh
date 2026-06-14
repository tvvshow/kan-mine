#!/usr/bin/env bash
# Phase A2 (lane-owned register transcript) A/B + correctness + register-residency gate.
#
# A2 removes the JPS shared-memory transcript entirely: each lane keeps the 16
# transcript words of the ONE jackpot tile it owns in REGISTERS (a 64x64 warp
# covers exactly 32 tiles = 32 lanes), the fold writes those registers, and the
# epilogue hashes them per-lane. This kills the fold's smem write traffic AND
# frees ~17KB dynamic smem (-> KSTAGES=4 becomes launchable).
#
#   A1  = default build               (smem transcript, direct-store on 1st touch) = SHIPPED
#   A2  = NVCC_EXTRA=-DA2_REG_TRANSCRIPT (register transcript)
#
# THREE gates (all on a single GPU box — no 5090 needed; sm_86 fold is ~47%):
#   gate-1 REGISTER RESIDENCY (make-or-break): q=c&15 is a RUNTIME index, so the
#          fold writes mytrans[] via a compile-time predicated select. If ptxas
#          still spilled it to LOCAL memory, A2 loses. We build A2 with
#          --ptxas-options=-v and FAIL if tc_cutlass_jackpot shows spill/stack.
#   gate-2 CORRECTNESS: POSTCHECK ok=1 on BOTH (CPU recomputes the GPU's winning
#          tile; a corrupt register transcript would fail it).
#   gate-3 PERF: hard-target FUSED ms/draw, Δ% (positive = A2 faster).
#
# Usage (GPU box with CUTLASS at $CUTLASS_HOME or ~/cutlass):
#   ARCH=sm_86  bash bench/ablate_a2.sh    # 3080Ti (fold ~47% -> strongest signal)
#   ARCH=sm_120 bash bench/ablate_a2.sh    # 5090
set -uo pipefail
cd "$(dirname "$0")/.."

ARCH="${ARCH:-sm_86}"
SEED="${SEED:-777}"
EASY="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"  # draw 1 wins -> POSTCHECK fires
HARD="0000000000000000000000000000000000000000000000000000000000000001"  # no early win -> clean full-sweep timing
NDRAW="${NDRAW:-3}"
PG=./build/plainproof_gen

echo "=== Phase A2 register-transcript A/B (ARCH=$ARCH SEED=$SEED) ==="

run_variant() {  # $1=label  $2=NVCC_EXTRA
  local label="$1" extra="$2"
  echo ""
  echo "--- build [$label] (NVCC_EXTRA='$extra') ---"
  # --ptxas-options=-v -> register/spill report in the log (gate-1). Single token
  # so it survives build.sh's unquoted ${EXTRA_FLAGS} word-splitting.
  if ! NVCC_EXTRA="$extra --ptxas-options=-v" ARCH="$ARCH" ./build.sh >/tmp/a2_build_$label.log 2>&1; then
    echo "  BUILD FAILED [$label] (see /tmp/a2_build_$label.log)"; tail -8 /tmp/a2_build_$label.log; return 1
  fi
  grep -qE "tc_cutlass_v2|tc_cutlass" /tmp/a2_build_$label.log || { echo "  ✗ fell back to tc_block (CUTLASS missing) — abort"; return 1; }

  # --- gate-1: register residency of the search kernel (A2 only matters here) ---
  # ptxas prints, per entry: "Used N registers" and (if it spilled) a non-zero
  # "N bytes spill stores/loads" + "N bytes stack frame". For the jackpot kernel
  # those MUST be zero, or the register transcript silently went to local memory.
  local blk reg spill
  blk=$(grep -A3 -E "Compiling entry function.*tc_cutlass_jackpot" /tmp/a2_build_$label.log)
  [ -z "$blk" ] && blk=$(grep -A3 -E "tc_cutlass_jackpot" /tmp/a2_build_$label.log)
  reg=$(echo "$blk"   | grep -oE "Used [0-9]+ registers" | grep -oE "[0-9]+" | head -1)
  spill=$(echo "$blk" | grep -oE "[0-9]+ bytes spill (stores|loads)" | grep -oE "^[0-9]+" | awk '{s+=$1} END{print s+0}')
  echo "  ptxas[jackpot]: ${reg:-?} registers, spill=${spill:-0} bytes"
  eval "REG_$label=${reg:-0}"; eval "SPILL_$label=${spill:-0}"

  # --- gate-2: easy target -> draw-1 winner -> POSTCHECK (CPU re-verifies) ---
  $PG --cfg real --mine 1 --target "$EASY" "$SEED" >/tmp/a2_proof_$label.b64 2>/tmp/a2_corr_$label.log
  local postck; postck=$(grep -E "POSTCHECK" /tmp/a2_corr_$label.log | grep -oE "ok=[01]" | tail -1)
  local sha;    sha=$(sha256sum /tmp/a2_proof_$label.b64 | awk '{print $1}')
  echo "  correctness: POSTCHECK ${postck:-<none>}   proof.sha256=${sha:0:16}…"

  # --- gate-3: hard target -> no early-out -> full sweep every draw ---
  $PG --cfg real --mine "$NDRAW" --target "$HARD" --breakdown "$SEED" >/dev/null 2>/tmp/a2_time_$label.log
  local mss; mss=$(grep -oE "FUSED [0-9]+ tiles, [0-9.]+ ms" /tmp/a2_time_$label.log | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+")
  local last; last=$(echo "$mss" | tail -1)
  echo "  kernel FUSED ms/draw (all $NDRAW): $(echo $mss | tr '\n' ' ')-> steady=$last ms"

  eval "MS_$label=$last"; eval "SHA_$label='$sha'"; eval "OK_$label='$postck'"
}

run_variant A1 ""                     || exit 1
run_variant A2 "-DA2_REG_TRANSCRIPT"  || exit 1

echo ""
echo "=== verdict ==="
echo "  ptxas      A1=${REG_A1:-?} regs / spill ${SPILL_A1:-?}B   A2=${REG_A2:-?} regs / spill ${SPILL_A2:-?}B"
echo "  POSTCHECK  A1=${OK_A1:-?}  A2=${OK_A2:-?}"
echo "  proof sha  A1=${SHA_A1:0:16}…  A2=${SHA_A2:0:16}… (racy winner -> differ; not a gate)"

fail=0
# gate-1: A2 must not have spilled the register transcript to local memory
[ "${SPILL_A2:-1}" = "0" ] || { echo "  ✗ A2 SPILLED ${SPILL_A2}B -> mytrans went to local memory (A2 defeated). Reduce live regs or revert."; fail=1; }
# gate-2: both correct
[ "${OK_A1:-}" = "ok=1" ] || { echo "  ✗ A1 POSTCHECK not ok=1"; fail=1; }
[ "${OK_A2:-}" = "ok=1" ] || { echo "  ✗ A2 POSTCHECK not ok=1"; fail=1; }
[ "$fail" = 0 ] && echo "  ✓ A2 register-resident (no spill) + A1/A2 both POSTCHECK ok=1"

if [ "$fail" = 0 ] && [ -n "${MS_A1:-}" ] && [ -n "${MS_A2:-}" ]; then
  ths_a1=$(awk "BEGIN{printf \"%.1f\", 70368.744/$MS_A1}")
  ths_a2=$(awk "BEGIN{printf \"%.1f\", 70368.744/$MS_A2}")
  delta=$(awk "BEGIN{printf \"%+.2f\", ($MS_A1-$MS_A2)/$MS_A1*100}")
  echo ""
  echo "  A1 (smem) : $MS_A1 ms = $ths_a1 TH/s"
  echo "  A2 (reg)  : $MS_A2 ms = $ths_a2 TH/s"
  echo "  Δ         : ${delta}% kernel (positive = A2 faster)"
  echo ""
  echo "  NEXT if Δ positive: retest A2 + KSTAGES=4 (now fits — JPS smem freed):"
  echo "    KSTAGES=4 NVCC_EXTRA=-DA2_REG_TRANSCRIPT ARCH=$ARCH ./build.sh && \\"
  echo "      $PG --cfg real --mine $NDRAW --target $HARD --breakdown $SEED"
  echo "  GATE to deploy: spill=0 AND POSTCHECK ok=1 AND Δ stably positive AND pool accepted>0/rejected=0."
fi
exit $fail
