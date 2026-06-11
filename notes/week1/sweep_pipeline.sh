#!/usr/bin/env bash
# sweep_pipeline.sh — for every (BN, STAGES, RSUB, REG_PIPE, MINBLOCKS, SWIZZLE, ONE_SYNC)
# config on tc_deep_pipeline.cu, measure register usage, spills (lmem), smem, and
# TH/s, then correctness-gate the key configs with a real POSTCHECK (loosened
# target). Produces a decision table: which config approaches CUTLASS 130 TH/s
# without spilling?
#
# Run on the 3080Ti box (ssh -p 23 root@117.50.177.251), inside peral/:
#   cd peral && bash notes/week1/sweep_pipeline.sh | tee sweep.log
#
# Prereq: build.sh already ran once (to get plainproof_gen.o + blake3 *.o).
set -euo pipefail
cd "$(dirname "$0")/../.."    # peral/
ROOT="$(pwd)"
BUILD="${ROOT}/build"

if [ ! -f "${BUILD}/plainproof_gen.o" ]; then
  echo "ERROR: build/plainproof_gen.o not found. Run 'bash build.sh' first."
  exit 1
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ARCH="${ARCH:-sm_86}"

echo "=== sweep_pipeline: tc_deep_pipeline.cu config sweep ==="
echo "ARCH=${ARCH}  $(date)"
echo ""
echo "BN STGS RSUB RPIPE MINB SWZ OS | regs  lmem   smem | ms/draw   TH/s  STATUS | notes"
echo "---- ---- ---- ----- ---- --- -- | ---- ------ ----- | ------- ------ ------ | -----"

# Reusable host objects (already compiled by build.sh)
cd "${BUILD}"
BL_OBJ="blake3.o blake3_dispatch.o blake3_portable.o"
for asm in blake3_sse2_x86-64_unix.o blake3_sse41_x86-64_unix.o blake3_avx2_x86-64_unix.o blake3_avx512_x86-64_unix.o; do
  [ -f "$asm" ] && BL_OBJ="${BL_OBJ} $asm"
done

sweep_one() {
  local bn=$1 stg=$2 rsub=$3 rpipe=$4 minb=$5 swz=$6 os=$7 label=$8
  local ptx_log="ptx_${bn}_${stg}_${rsub}_${rpipe}_${minb}_${swz}_${os}.log"
  local run_log="run_${bn}_${stg}_${rsub}_${rpipe}_${minb}_${swz}_${os}.log"

  # compile: capture -Xptxas -v (regs/lmem/smem) to ptx_log
  nvcc -O3 -arch=${ARCH} -std=c++17 \
    -DBN=${bn} -DSTAGES=${stg} -DRSUB=${rsub} -DREG_PIPE=${rpipe} -DMINBLOCKS=${minb} -DSWIZZLE=${swz} -DONE_SYNC=${os} \
    -Xptxas -v -c "${ROOT}/src/tc_deep_pipeline.cu" -o tc_deep.o \
    > "${ptx_log}" 2>&1 || {
      echo "${bn}  ${stg}    ${rsub}   ${rpipe}    ${minb}   ${swz}  ${os} | COMPILE FAIL (see ${ptx_log})"
      return
    }

  # link
  g++ -O3 -fopenmp plainproof_gen.o ${BL_OBJ} tc_deep.o \
    -L"${CUDA_HOME}/lib64" -lcudart -o plainproof_gen_sweep \
    > /dev/null 2>&1 || {
      echo "${bn}  ${stg}    ${rsub}   ${rpipe}    ${minb}   ${swz}  ${os} | LINK FAIL"
      return
    }

  # extract regs/lmem/smem from ptx_log (nvcc -Xptxas -v output format):
  #   ptxas info    : Used N registers, M bytes cmem[0], K bytes lmem
  #   ptxas info    : Function properties for tc_deep_jackpot
  #   .maxntid 256, 1, 1
  local regs=$(grep -oP 'Used \K[0-9]+(?= registers)' "${ptx_log}" | head -1)
  local lmem=$(grep -oP '[0-9]+(?= bytes lmem)' "${ptx_log}" | head -1)
  local smem=$(grep -oP '[0-9]+(?= bytes smem)' "${ptx_log}" | head -1)
  regs=${regs:-"??"}
  lmem=${lmem:-0}
  smem=${smem:-"??"}

  # run: --mine 3 --tc --cfg real (3 draws, real 131072 config)
  OMP_NUM_THREADS=$(nproc) ./plainproof_gen_sweep --mine 3 --tc --cfg real \
    > "${run_log}" 2>&1 || {
      echo "${bn}  ${stg}    ${rsub}   ${rpipe}    ${minb}   ${swz}  ${os} | ${regs}  ${lmem}  ${smem} | RUN FAIL (see ${run_log})"
      return
    }

  # extract warm-draw timing: tail -1 = the LAST of 3 draws = steady-state
  # (draw 0 pays one-time persistent-buffer malloc + smem-attr setup).
  local ms=$(grep -oP 'tc\(deep\):.*\K[0-9.]+(?= ms)' "${run_log}" | tail -1)
  local ths=$(grep -oP 'tc\(deep\):.*\K[0-9.]+(?= TH/s)' "${run_log}" | tail -1)
  ms=${ms:-"??"}
  ths=${ths:-"??"}

  # health, NOT correctness: a wrapper error (tc_deep: LAUNCH err / err / FAIL)
  # is fatal; otherwise OK once the run reaches "MINE done". POSTCHECK is NOT
  # graded here -- it only prints on a jackpot win, infeasible in 3 draws at the
  # real 2^204 target. Per-config math is inherited from the byte-identical
  # tc_imma2 fold/mma/ldmatrix geometry; the recipe is correctness-gated for real
  # in the POSTCHECK section after this grid (loosened target forces a true win).
  local status="OK"
  if grep -qiE 'tc_deep:.*(err|fail)' "${run_log}"; then status="KERR"
  elif ! grep -q 'MINE done' "${run_log}"; then status="FAIL"
  fi

  printf "%3d  %3d   %3d    %d     %d    %d  %d | %4s %6s %5s | %7s %6s %-6s | %s\n" \
    ${bn} ${stg} ${rsub} ${rpipe} ${minb} ${swz} ${os} "${regs}" "${lmem}" "${smem}" "${ms}" "${ths}" "${status}" "${label}"
}

# ---- correctness gate (run AFTER the perf table, on the configs that matter) --
# The per-config table above measures speed + register pressure only. To prove a
# config still computes the RIGHT jackpot we need a POSTCHECK ok=1, but at the real
# 2^204 target a win needs ~64 draws. So we LOOSEN the target: bound = target x
# (h*w*dot_len) = target x 524288. With target = 2^216 the bound is ~2^235, so
# ~64 of the 2^27 tiles win per draw -> a win is guaranteed on draw 0, yet winners
# are still 1-in-2M, so the GPU must select a genuinely-winning tile. POSTCHECK then
# INDEPENDENTLY recomputes that exact tile's jackpot on the CPU and requires it <=
# bound -- a strong GPU-vs-CPU check that catches any fold/addressing regression
# from REG_PIPE or the BN=256 tiling. (golden cfg can't be used: it is w=8, the
# kernel needs w=16.) 2^216 big-endian = byte[4]=0x01:
LOOSE_TARGET="0000000001000000000000000000000000000000000000000000000000000000"

postcheck_one() {
  local bn=$1 stg=$2 rsub=$3 rpipe=$4 minb=$5 swz=$6 os=$7 label=$8
  local cl="pcc_${bn}_${stg}_${rsub}_${rpipe}_${minb}_${swz}_${os}.log"
  local rl="pcr_${bn}_${stg}_${rsub}_${rpipe}_${minb}_${swz}_${os}.log"
  nvcc -O3 -arch=${ARCH} -std=c++17 \
    -DBN=${bn} -DSTAGES=${stg} -DRSUB=${rsub} -DREG_PIPE=${rpipe} -DMINBLOCKS=${minb} -DSWIZZLE=${swz} -DONE_SYNC=${os} \
    -c "${ROOT}/src/tc_deep_pipeline.cu" -o tc_deep.o > "${cl}" 2>&1 \
    && g++ -O3 -fopenmp plainproof_gen.o ${BL_OBJ} tc_deep.o \
       -L"${CUDA_HOME}/lib64" -lcudart -o plainproof_gen_pcc >> "${cl}" 2>&1 || {
      printf "  POSTCHECK %-34s : BUILD FAIL (see %s)\n" "${label}" "${cl}"; return; }
  OMP_NUM_THREADS=$(nproc) ./plainproof_gen_pcc --mine 5 --tc --cfg real \
    --target "${LOOSE_TARGET}" > "${rl}" 2>&1 || true
  local ok=$(grep -oP 'POSTCHECK.*\bok=\K[01]' "${rl}" | tail -1)
  local win=$(grep -oP 'MINE WIN draw=\K[0-9]+' "${rl}" | tail -1)
  if [ "${ok}" = "1" ]; then
    printf "  POSTCHECK %-34s : ok=1 (win draw %s)  CORRECT\n" "${label}" "${win:-?}"
  elif [ -n "${win}" ]; then
    printf "  POSTCHECK %-34s : ok=%s  *** WRONG MATH *** (see %s)\n" "${label}" "${ok:-0}" "${rl}"
  else
    printf "  POSTCHECK %-34s : NO WIN in 5 draws (target too tight? see %s)\n" "${label}" "${rl}"
  fi
}

# ---- SWEEP GRID (prioritize the CUTLASS recipe first) ----
# Start with the target: BN=256 s3 RSUB=64 REG_PIPE=1 SWIZZLE=1 (CUTLASS recipe).
# Then ablation: REG_PIPE off? SWIZZLE off? STAGES 2/4?
# Then BN=128: does the validated 56 TH/s config improve with the same levers?
# (SWIZZLE is expected to be the biggest single lever: the linear layout 4-way
# bank-conflicts EVERY ldmatrix; tc_imma2's 56 TH/s was measured conflicted.)
# ONE_SYNC: the RSUB 32→64 data shows 2× (29.79→58.97), proving every stage's
# double-barrier overhead dominates. ONE_SYNC=1 removes the tail barrier, letting
# async-load overlap with MMA (CUTLASS pattern). At 1 block/SM (BN=256) this
# should eliminate the SM idle time and potentially double throughput.
# RSUB=128: BN=128 s2 already uses 72KB smem (fits sm_86 ~99KB). Doubling to
# RSUB=128 halves stage count → another ~2× from fewer barriers, targeting ~110 TH/s.

# --- primary target: BN=256 ---
sweep_one 256 3 64 1 1 1 0 "CUTLASS recipe (baseline OS=0)"
sweep_one 256 3 64 1 1 1 1 "CUTLASS recipe ONE_SYNC=1"
sweep_one 256 3 64 1 1 0 0 "recipe, no swizzle (conflict cost)"
sweep_one 256 3 64 0 1 1 0 "no reg-pipe"
sweep_one 256 2 64 1 1 1 0 "shallow pipe"
sweep_one 256 4 64 1 1 1 0 "deeper pipe"
sweep_one 256 3 32 1 1 1 0 "narrow RSUB"

# --- baseline: BN=128 (our validated 56 TH/s geometry) ---
sweep_one 128 2 64 0 2 1 0 "tc_imma2 + swizzle ONLY"
sweep_one 128 2 64 0 2 1 1 "tc_imma2 + swizzle + ONE_SYNC"
sweep_one 128 3 64 1 2 1 0 "BN128 + deep + swizzle"
sweep_one 128 2 64 0 2 0 0 "BN128 s2 no-pipe (== tc_imma2 exact)"
sweep_one 128 2 128 0 2 1 0 "BN128 RSUB=128 (half stage count)"

echo ""
echo "=== correctness gate: loosened target forces a real POSTCHECK ok=1 ==="
# Anchor first: BN128 s2 no-pipe no-swizzle == tc_imma2 geometry (already
# verifier-VALID) -> proves the harness/build/loose-target plumbing is sound.
# Then each NEW lever that could break addressing: SWIZZLE (new smem permute)
# on the anchor geometry, and the full BN=256 + REG_PIPE + SWIZZLE recipe.
postcheck_one 128 2 64 0 2 0 0 "BN128 s2 no-pipe (anchor)"
postcheck_one 128 2 64 0 2 1 0 "anchor + SWIZZLE (permute gate)"
postcheck_one 256 3 64 1 1 1 0 "CUTLASS recipe (BN256 s3 pipe swz)"

echo ""
echo "=== interpretation guide ==="
echo "lmem > 0     = register spill (bad; kills perf)"
echo "regs > 255   = likely causes low occupancy on sm_86"
echo "smem > 99000 = exceeds sm_86 per-block cap (launch fails or forces 1 block/SM)"
echo "STATUS KERR  = kernel/launch error (fatal); FAIL = run never reached MINE done"
echo "POSTCHECK    = real GPU-vs-CPU jackpot check; 'WRONG MATH' = fatal correctness bug"
echo "target       = approach 130 TH/s (CUTLASS roofline) with lmem=0 and STATUS OK"
echo ""
echo "logs: ${BUILD}/ptx_*.log run_*.log (perf) / pcc_*.log pcr_*.log (correctness)"
