#!/usr/bin/env bash
# Detached ncu profile of the fused search kernel (one draw).
# Usage: bash ncu_profile.sh     (run as ubuntu; sudo -n must work for ncu)
# Results land in ~/peral/build/ncu_v5.txt — poll that file, do NOT wait here.
# WHY detached: sudo ncu via paramiko exec_command hangs the channel (prompt
# read?); setsid + nohup + redirect breaks the tie. After it finishes, kill
# any TreeLauncherSubreaper leftovers and CHECK CLOCKS (a stuck profiler
# pinned SM at 1365MHz once and halved every number measured after it).
set -u
cd ~/peral/build
LOG=ncu_v5_run.log
OUT=ncu_v5.txt
rm -f "$OUT" "$LOG"

# Profile ONLY the search kernel, skip the prep kernels: -k regex.
# 1 launch is enough (-c 1). Sections: occupancy + scheduler + memory.
setsid nohup sudo -n ncu \
  -k 'tc_cutlass_jackpot' -c 1 \
  --section Occupancy --section SchedulerStats --section WarpStateStats \
  --section MemoryWorkloadAnalysis --section SpeedOfLight \
  -o /dev/null --log-file "$OUT" --print-summary per-kernel \
  ./plainproof_gen_v5 --cfg real --mine 1 --tc \
  --target 0000000000000000000000000000000000000000000000000000000000000001 \
  > "$LOG" 2>&1 < /dev/null &
echo "ncu launched detached (pid $!). Poll: tail ~/peral/build/$OUT"
