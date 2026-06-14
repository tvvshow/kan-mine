#!/usr/bin/env bash
# Test the OCCUPANCY lever on sm_86. The headroom probe showed the kernel is
# stall-bound (only ~46-54% of samples at the 350W cap, avg 316W, 1 block/SM =
# 16.7% occupancy) -> tensor cores idle during the fold phase, leaving power
# unused. Higher occupancy (>=2 blocks/SM) lets one block's GEMM overlap another
# block's fold -> fills the idle power -> potentially more throughput.
#
# SMALL_TILE (64x128 TB, 32x64 warp) halves regs+smem/block -> should reach 2+
# blocks/SM. (Earlier "证伪 on 5090" doesn't transfer: the 5090 isn't stall-bound
# like this power-capped 3080Ti, and the fold static_assert now allows JR*JC==16.)
# Default tile baseline = 97.8 / persist 99.0 TH/s @ 1 block/SM.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
HARD=0000000000000000000000000000000000000000000000000000000000000001
EASY=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff

echo "=== build SMALL_TILE (64x128 TB, 32x64 warp) ==="
if ! SMALL_TILE=1 ARCH=sm_86 ./build.sh >/tmp/st_build.log 2>&1; then
  echo "BUILD-FAIL:"; tail -18 /tmp/st_build.log; exit 1
fi
echo "  regs: $(grep -A2 'Compiling entry function.*tc_cutlass_jackpot' /tmp/st_build.log | grep -oE 'Used [0-9]+ registers' | head -1)"
echo "  POSTCHECK: $(./build/plainproof_gen --cfg real --mine 1 --target $EASY 777 2>&1 | grep -oE 'ok=[01]' | tail -1)"

probe() {  # $1=label $2=env
  local label="$1" envv="$2"
  nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader,nounits -lms 200 >/tmp/st_$label.csv 2>/dev/null &
  local M=$!
  env $envv ./build/plainproof_gen --cfg real --mine 30 --target "$HARD" --breakdown 777 2>/tmp/st_$label.run >/dev/null
  kill "$M" 2>/dev/null || true
  local ms ths
  ms=$(grep -oE "FUSED [0-9]+ tiles, [0-9.]+ ms" /tmp/st_$label.run | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+" | tail -10 | awk '{s+=$1;n++}END{if(n)printf "%.1f",s/n}')
  ths=$(awk "BEGIN{if($ms>0)printf \"%.1f\",70368.744/$ms}")
  printf "  %-9s %sms (%s TH/s) | %s | %s\n" "$label" "${ms:-?}" "${ths:-?}" \
    "$(awk -F, 'NR>1&&$1>1000{n++;c+=$1;p+=$2;if($2>=349)cap++}END{if(n)printf "clk=%.0fMHz pwr=%.0fW cap350=%.0f%%",c/n,p/n,100*cap/n}' /tmp/st_$label.csv)" \
    "$(grep -oE 'PERSISTENT grid [0-9]+ blocks \([0-9]+ SM x [0-9]+/SM\)' /tmp/st_$label.run | tail -1 || echo grid-mode)"
}
echo "=== SMALL_TILE sustained probe (vs default 97.8/99.0 @ 1 block/SM) ==="
probe baseline ""
probe persist  "TC_PERSIST=1"
