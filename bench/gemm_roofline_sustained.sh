#!/usr/bin/env bash
# Sustained CUTLASS int8 GEMM roofline under the power cap, with clock monitoring.
# WHY: the existing bench runs only iters=20 (~170ms) = a boost-clock BURST, which
# is why it reported 131 TMAC/s. The fused jackpot kernel runs for seconds and is
# SW-power-capped (350W -> ~1560 MHz). To know whether our ~100 TH/s is far from
# the GEMM ceiling or near it, we must compare against the GEMM at the SAME
# sustained, power-capped clock. iters=300 (~15s) clamps the clock like real mining.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
ITERS="${ITERS:-300}"

echo "=== build gemm bench ==="
nvcc -O3 -arch=sm_86 -std=c++17 -I"$HOME/cutlass/include" -I"$HOME/cutlass/tools/util/include" \
     bench/cutlass_int8_bench.cu -o /root/gemmbench 2>&1 | tail -5
[ -x /root/gemmbench ] || { echo "BUILD FAIL"; exit 1; }

nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,noheader,nounits -lms 250 >/root/gbmon.csv 2>/dev/null &
MON=$!
echo "=== SUSTAINED gemm roofline (16384^2 x4096, iters=$ITERS) ==="
/root/gemmbench 16384 16384 4096 "$ITERS"
kill "$MON" 2>/dev/null || true

echo "=== clock during bench ==="
awk -F, 'NR>1{n++;c+=$1; if(mn==""||$1<mn)mn=$1; if($1>mx)mx=$1} END{printf "  sm_clk min=%s max=%s avg=%.0f MHz over %d samples\n",mn,mx,c/n,n}' /root/gbmon.csv
echo "=== throttle reasons during bench ==="
awk -F, 'NR>1{print $4}' /root/gbmon.csv | sort | uniq -c
