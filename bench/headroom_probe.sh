#!/usr/bin/env bash
# Is the fused kernel TRULY power-bound (no headroom) or stall-bound (headroom)?
#
# Decisive metric: cap350_frac = % of active samples pinned at >=349W.
#   ~100% + clk<max  => truly power-bound: the only lever is perf/watt (WGMMA).
#   <~80% + clk<max  => stalls (occupancy/latency/mem) leave power UNUSED =>
#                        a real on-board throughput lever exists (raise occupancy).
# Also prints the kernel's ACTUAL blocks/SM (occupancy denominator) via the
# TC_PERSIST verbose line, and whether persistent grid mode helps.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
HARD=0000000000000000000000000000000000000000000000000000000000000001
EASY=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff

echo "=== rebuild clean default (A1) ==="
ARCH=sm_86 ./build.sh >/tmp/hp_build.log 2>&1 || { echo BUILD-FAIL; tail -6 /tmp/hp_build.log; exit 1; }
echo "  regs: $(grep -A2 'Compiling entry function.*tc_cutlass_jackpot' /tmp/hp_build.log | grep -oE 'Used [0-9]+ registers' | head -1)"
echo "  POSTCHECK: $(./build/plainproof_gen --cfg real --mine 1 --target $EASY 777 2>&1 | grep -oE 'ok=[01]' | tail -1)"

probe() {  # $1=label  $2=env-prefix
  local label="$1" envv="$2"
  nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader,nounits -lms 200 >/tmp/hp_$label.csv 2>/dev/null &
  local M=$!
  env $envv ./build/plainproof_gen --cfg real --mine 30 --target "$HARD" --breakdown 777 \
    2>/tmp/hp_$label.run >/dev/null
  kill "$M" 2>/dev/null || true
  local ms ths stats occ
  ms=$(grep -oE "FUSED [0-9]+ tiles, [0-9.]+ ms" /tmp/hp_$label.run | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+" | tail -10 | awk '{s+=$1;n++}END{if(n)printf "%.1f",s/n}')
  ths=$(awk "BEGIN{if($ms>0)printf \"%.1f\",70368.744/$ms}")
  stats=$(awk -F, 'NR>1&&$1>1000{n++;c+=$1;p+=$2; if($2>=349)cap++} END{if(n)printf "clk=%.0fMHz pwr_avg=%.0fW cap350=%.0f%%",c/n,p/n,100*cap/n}' /tmp/hp_$label.csv)
  occ=$(grep -oE "PERSISTENT grid [0-9]+ blocks \([0-9]+ SM x [0-9]+/SM\)" /tmp/hp_$label.run | tail -1)
  printf "  %-9s %sms (%s TH/s) | %s | %s\n" "$label" "${ms:-?}" "${ths:-?}" "${stats:-?}" "${occ:-grid-mode}"
}

echo "=== headroom probe (sustained 30-draw, 200ms sampling) ==="
probe baseline ""
probe persist  "TC_PERSIST=1"
echo ""
echo "actual occupancy denominator (max blocks/SM for THIS kernel):"
echo "  -> see 'SM x N/SM' above (N = blocks/SM; 8*N/48 = occupancy on sm_86)"
