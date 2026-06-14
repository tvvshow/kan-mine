#!/usr/bin/env bash
# Does the fold's POWER draw suppress the whole-kernel clock? (the real lever)
#
# Finding so far: pure GEMM sustains ~1786 MHz @ 350W cap, but the FUSED kernel
# only sustains ~1411 MHz @ 350W — the fold's INT XOR + 32x REDUX.SYNC + smem
# draw extra power, so the power cap pushes the clock (and thus the GEMM) down.
# The fold's TIME cost is ~5% (clock-normalized); its CLOCK cost is the big one.
#
# Test: build FULL / NOREDUX / NOFOLD, run each sustained (20 draws) with clock+
# power monitoring. If the sustained clock RISES as fold work is removed, then
# reducing the fold's power (e.g. cheaper redux) raises the clock = a real win
# even though the time saved is tiny. If the clock stays flat, the fold isn't the
# power hog and only Phase C (WGMMA, better MAC/watt) can move throughput.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
HARD=0000000000000000000000000000000000000000000000000000000000000001

run() {  # $1=name $2=flag
  local name="$1" flag="$2"
  NVCC_EXTRA="$flag" ARCH=sm_86 ./build.sh >/tmp/fp_$name.blog 2>&1 || { echo "  $name BUILD FAIL"; return; }
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader,nounits -lms 250 >/tmp/fp_$name.csv 2>/dev/null &
  local M=$!
  ./build/plainproof_gen --cfg real --mine 20 --target "$HARD" --breakdown 777 2>/tmp/fp_$name.run >/dev/null
  kill "$M" 2>/dev/null || true
  # steady ms = mean of last 8 draws; clk/pwr = mean of active (>1000 MHz) samples
  local ms clk pwr tC
  ms=$(grep -oE "FUSED [0-9]+ tiles, [0-9.]+ ms" /tmp/fp_$name.run | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+" | tail -8 | awk '{s+=$1;n++}END{if(n)printf "%.1f",s/n}')
  clk=$(awk -F, 'NR>1&&$1>1000{n++;c+=$1}END{if(n)printf "%.0f",c/n}' /tmp/fp_$name.csv)
  pwr=$(awk -F, 'NR>1&&$1>1000{n++;c+=$2}END{if(n)printf "%.0f",c/n}' /tmp/fp_$name.csv)
  tC=$(awk -F, 'NR>1&&$3>0{if($3>m)m=$3}END{printf "%s",m}' /tmp/fp_$name.csv)
  local ths
  ths=$(awk "BEGIN{if($ms>0)printf \"%.1f\",70368.744/$ms; else print \"?\"}")
  printf "  %-8s steady=%sms (%s TH/s) | clk=%sMHz | pwr=%sW | tmax=%sC\n" "$name" "${ms:-?}" "$ths" "${clk:-?}" "${pwr:-?}" "${tC:-?}"
}

echo "=== fold POWER->CLOCK test (sustained 20-draw, 350W cap) ==="
run FULL    ""
run NOREDUX "-DPROFILE_NOREDUX"
run NOFOLD  "-DPROFILE_NOFOLD"
echo ""
echo "read: clk rising FULL->NOREDUX->NOFOLD => fold power suppresses the clock"
echo "      (then cheaper redux raises clock = win). flat clk => only WGMMA helps."
