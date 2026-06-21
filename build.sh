#!/usr/bin/env bash
# Native build for the Pearl(PRL) self-built PoUW solver.
#
# ONE fast path, no slow variants. This builds exactly what runs in production
# and earns on the live pool:
#   * CUTLASS fused int8 search kernel tc_cutlass_v2.cu (GROUPM=128, 350+ TH/s kernel on 5090)
#     + GPU-resident draw pipeline gpu_prep.cu  => 102+ TH/s wall-clock on a 3090
#     (fallback when CUTLASS headers absent: tc_block.cu WMMA, ~30 TH/s)
#   * BLAKE3 with SIMD asm (SSE2/SSE4.1/AVX2/AVX512), ~6x the portable scalar path
#   * fast inline splitmix64 RNG fill (in plainproof_gen.cpp)
# The old slow/experimental kernels (tc_gemm, tc_panel) and the dp4a fallback
# (jackpot_kernel) were retired - see _archive/dead-kernels/.
#
# CUDA code COMPILES without a GPU present (only *running* kernels needs one), so
# this build NEVER depends on nvidia-smi. Arch selection order:
#   ARCH env override  >  nvidia-smi detection (only if a GPU is present)  >
#   portable multi-arch fatbin (Turing..Hopper SASS + Hopper PTX for Blackwell JIT).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "=== toolchain ==="
nvcc --version | tail -2
g++ --version | head -1

echo "=== GPU (informational; build does NOT require one) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "no nvidia-smi (GPU-less build host) - OK, compiling anyway"
fi

# --- choose nvcc arch flags (this block must never fail the build) ---
GENCODE=""
if [ -n "${ARCH:-}" ]; then
  GENCODE="-arch=${ARCH}"
  echo "arch: ENV override -> ${ARCH}"
elif [ -z "${PORTABLE:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  # PORTABLE builds are REDISTRIBUTABLE -> never tune to the build box's GPU;
  # fall through to the multi-arch fatbin so the tarball runs on every arch.
  DETECTED="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
  if [[ "${DETECTED:-}" =~ ^[0-9]+$ ]]; then
    GENCODE="-arch=sm_${DETECTED}"
    echo "arch: detected GPU -> sm_${DETECTED}"
  fi
fi
if [ -z "${GENCODE}" ]; then
  # Portable fatbin: real SASS for Turing..Hopper + compute_90 PTX so the driver
  # JITs to newer arches (e.g. Blackwell sm_120) at runtime. CUDA 12.4 nvcc tops
  # out at sm_90, so sm_120 SASS is intentionally NOT emitted here.
  GENCODE="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
  echo "arch: no GPU/override -> portable multi-arch fatbin (sm_75..sm_90 + PTX)"
fi

BUILD="${ROOT}/build"; mkdir -p "${BUILD}"; cd "${BUILD}"

# ---------------------------------------------------------------------------
# BLAKE3 - SIMD asm when available (the fast path the live miner uses), else a
# portable scalar fallback. blake3_dispatch picks the best impl at runtime, so
# blake3.c/blake3_dispatch.c must be compiled WITHOUT the BLAKE3_NO_* macros for
# the SIMD objects to be referenced. The 4 .S sources are not vendored in the
# tree; fetch them from the official repo if missing (the build box has net).
# ---------------------------------------------------------------------------
echo "=== blake3 ==="
B3="${ROOT}/blake3"
B3_ASM="blake3_sse2_x86-64_unix.S blake3_sse41_x86-64_unix.S blake3_avx2_x86-64_unix.S blake3_avx512_x86-64_unix.S"
have_simd=1
for f in ${B3_ASM}; do [ -f "${B3}/${f}" ] || have_simd=0; done
if [ "${have_simd}" = "0" ] && command -v curl >/dev/null 2>&1; then
  echo "  SIMD asm not vendored -> fetching from github (BLAKE3-team/BLAKE3)..."
  base="https://raw.githubusercontent.com/BLAKE3-team/BLAKE3/master/c"
  have_simd=1
  for f in ${B3_ASM}; do
    curl -fsSL -o "${B3}/${f}" "${base}/${f}" || { have_simd=0; break; }
  done
fi

# portable impl is always linked (provides the scalar fallback symbols)
gcc -O3 -I"${B3}" -c "${B3}/blake3_portable.c"
BL_OBJ="blake3.o blake3_dispatch.o blake3_portable.o"
if [ "${have_simd}" = "1" ]; then
  echo "  blake3: SIMD (SSE2/SSE4.1/AVX2/AVX512) - fast path"
  gcc -O3 -I"${B3}" -c "${B3}/blake3.c" "${B3}/blake3_dispatch.c"
  # shellcheck disable=SC2086
  gcc -O3 -I"${B3}" -c \
    "${B3}/blake3_sse2_x86-64_unix.S" "${B3}/blake3_sse41_x86-64_unix.S" \
    "${B3}/blake3_avx2_x86-64_unix.S" "${B3}/blake3_avx512_x86-64_unix.S"
  BL_OBJ="${BL_OBJ} blake3_sse2_x86-64_unix.o blake3_sse41_x86-64_unix.o blake3_avx2_x86-64_unix.o blake3_avx512_x86-64_unix.o"
else
  echo "  WARNING: blake3 SIMD asm unavailable (offline?) -> PORTABLE scalar fallback (~6x slower hashing)."
  gcc -O3 -I"${B3}" -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512 \
    -c "${B3}/blake3.c" "${B3}/blake3_dispatch.c"
fi

echo "=== host: prover core ==="
# Two object files from the SAME proven pipeline source:
#   plainproof_gen.o : keeps its own main() -> the standalone CLI
#   prover_lib.o     : -DPROVER_LIB drops main() -> linked into pearl-miner
# (-I CUDA include: plainproof_gen.cpp includes gpu_draw.h -> cuda_runtime.h;
#  not every box has CUDA headers on g++'s default include path.)
g++ -O3 -std=c++17 -fopenmp -I"${ROOT}/blake3" -I"${CUDA_HOME}/include" -c "${ROOT}/src/plainproof_gen.cpp" -o plainproof_gen.o
g++ -O3 -std=c++17 -fopenmp -I"${ROOT}/blake3" -I"${CUDA_HOME}/include" -DPROVER_LIB -c "${ROOT}/src/plainproof_gen.cpp" -o prover_lib.o

echo "=== host: unified miner driver (pool + solo) ==="
g++ -O3 -std=c++17 -I"${ROOT}/src" -I"${ROOT}/blake3" -I"${CUDA_HOME}/include" -c "${ROOT}/src/miner_main.cpp" -o miner_main.o

# --- the ONE GPU kernel pair: CUTLASS fused search + GPU draw pipeline --------
# tc_cutlass_v2.cu: CUTLASS 3.5.1 threadblock-level FoldMmaMultistage — IMMA
#   m16n8k32 int8, TB 128x256x64, stages=3, fold callback every rank-chunk,
#   redux.sync warp XOR, GROUPM=128 grouped raster (kernel 200 ms = 350+ TH/s on
#   a 3090; POSTCHECK ok=1; live pool wall-clock 102+ TH/s, beats lpminer 91).
# gpu_prep.cu: per-draw RNG fill + blake3 tree commitments + noise ON the GPU
#   (10 ms vs 1490 ms CPU + 1 GB H2D). plainproof_gen auto-takes this path
#   when the symbols are linked (weak refs), else falls back to CPU producer.
# Needs the CUTLASS headers (header-only): CUTLASS_HOME or ~/cutlass.
# Fallback: tc_block.cu (WMMA, ~30 TH/s) keeps GPU-less/CUTLASS-less builds alive.
echo "=== fused tensor-core kernel ==="
CUTLASS_HOME="${CUTLASS_HOME:-$HOME/cutlass}"
# GROUPM is architecture-sensitive.  The historical sm_86 production baseline
# (3090/3080Ti) was tuned at GROUPM=8 (102+ TH/s wall on 3090).  Later
# 4090/5090 branches use larger grouping, but making 128 the universal default
# regresses Ampere portable builds.  Keep an explicit GROUPM env override, and
# otherwise choose the safe default for the requested ARCH.
if [ -n "${GROUPM:-}" ]; then
  GROUPM="${GROUPM}"
else
  case "${ARCH:-}" in
    sm_80|sm_86) GROUPM=8 ;;
    *) GROUPM=128 ;;
  esac
fi
KSTAGES="${KSTAGES:-3}"
SMALL_TILE="${SMALL_TILE:-}"
EXTRA_FLAGS=""
[ -n "$SMALL_TILE" ] && EXTRA_FLAGS="${EXTRA_FLAGS} -DSMALL_TILE"
# Arbitrary extra nvcc flags passthrough (A/B benchmarking + profiling switches),
# e.g. NVCC_EXTRA="-DFOLD_RMW_ALWAYS" or "-DPROFILE_NOFOLD". Empty in production.
EXTRA_FLAGS="${EXTRA_FLAGS} ${NVCC_EXTRA:-}"
if [ -d "${CUTLASS_HOME}/include/cutlass" ]; then
  echo "  CUTLASS at ${CUTLASS_HOME} -> tc_cutlass_v2 (GROUPM=${GROUPM} KSTAGES=${KSTAGES}${SMALL_TILE:+ SMALL_TILE}) + gpu_prep [LIVE 102+ TH/s]"
  echo "  (persistent grid is the DEFAULT now; disable at runtime with TC_PERSIST=0 ./build/kan ...)"
  # shellcheck disable=SC2086
  nvcc -O3 ${GENCODE} -std=c++17 -DGROUPM=${GROUPM} -DKSTAGES=${KSTAGES} ${EXTRA_FLAGS} -I"${CUTLASS_HOME}/include" -c "${ROOT}/src/tc_cutlass_v2.cu" -o tc_kernel.o
  # shellcheck disable=SC2086
  nvcc -O3 ${GENCODE} -std=c++17 -c "${ROOT}/src/gpu_prep.cu"
  TC_OBJ="tc_kernel.o gpu_prep.o"
else
  echo "  WARNING: CUTLASS not found (set CUTLASS_HOME or clone to ~/cutlass:"
  echo "           git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass ~/cutlass)"
  echo "           -> falling back to tc_block (WMMA, ~30 TH/s, CPU draw pipeline)"
  # shellcheck disable=SC2086
  nvcc -O3 ${GENCODE} -std=c++17 -c "${ROOT}/src/tc_block.cu" -o tc_kernel.o
  TC_OBJ="tc_kernel.o"
fi

echo "=== link: plainproof_gen (CLI) ==="
# PORTABLE=1 -> a "download & run" binary whose ONLY runtime dep is the NVIDIA
# driver (libcuda, always on a GPU host): static cudart (no CUDA toolkit needed
# on the target), static libstdc++/libgcc, and an $ORIGIN/lib rpath so the few
# remaining .so (libssl/libcrypto/libgomp...) can sit beside the binary. Default
# (empty PORTABLE) links exactly as before. $ORIGIN must reach the linker
# LITERALLY -> stored with \$ at assignment, NOT re-expanded on unquoted use.
if [ -n "${PORTABLE:-}" ]; then
  CUDART_LIBS="-lcudart_static -lculibos -lpthread -ldl -lrt"
  STATIC_CXX="-static-libstdc++ -static-libgcc"
  RPATH_ARG="-Wl,-rpath,\$ORIGIN/lib"
  echo "  PORTABLE: static cudart + static libstdc++/libgcc + \$ORIGIN/lib rpath"
else
  CUDART_LIBS="-lcudart -lpthread"
  STATIC_CXX=""
  RPATH_ARG=""
fi
# shellcheck disable=SC2086
g++ -O3 -fopenmp ${STATIC_CXX} ${RPATH_ARG} plainproof_gen.o ${BL_OBJ} ${TC_OBJ} \
  -L"${CUDA_HOME}/lib64" ${CUDART_LIBS} -o plainproof_gen
ls -la "${BUILD}/plainproof_gen"

echo "=== link: kan (unified pool + solo) ==="
# OpenSSL for solo HTTPS RPC; pthread for the stratum/poller threads.
# shellcheck disable=SC2086
g++ -O3 -fopenmp ${STATIC_CXX} ${RPATH_ARG} miner_main.o prover_lib.o ${BL_OBJ} ${TC_OBJ} \
  -L"${CUDA_HOME}/lib64" -lssl -lcrypto ${CUDART_LIBS} -o kan
ls -la "${BUILD}/kan"

# Compatibility alias for older launch/deploy helpers.  `kan` is the canonical
# binary name, but some production scripts and boxes still look for
# `build/pearl-miner`.  Keep a byte-identical copy so those entry points do not
# accidentally fail or run a stale pre-rename binary.
cp -f kan pearl-miner
ls -la "${BUILD}/pearl-miner"

# --- zkprove (Rust): SOLO-only PlainProof -> ZK proof -> block helper ---------
# Needs cargo (Rust). The C++ binaries above do NOT need it; only `--solo` calls
# zkprove at run time. Build it when cargo is present; otherwise warn and skip so
# the pool-only build still succeeds on a Rust-less host (e.g. the cnb CUDA image).
echo "=== zkprove (Rust solo helper) ==="
if command -v cargo >/dev/null 2>&1; then
  ( cd "${ROOT}/zk-pow" && cargo build --release --bin zkprove )
  cp -f "${ROOT}/zk-pow/target/release/zkprove" "${BUILD}/zkprove" 2>/dev/null || \
    cp -f "${ROOT}/zk-pow/target/release/zkprove.exe" "${BUILD}/zkprove" 2>/dev/null || true
  if [ -x "${BUILD}/zkprove" ]; then
    "${BUILD}/zkprove" selftest && echo "zkprove selftest OK"
    ls -la "${BUILD}/zkprove"
  fi
else
  echo "WARNING: cargo not found -> skipping zkprove build."
  echo "         pool mode works without it; --solo needs zkprove (install Rust, re-run)."
fi

echo ""
echo "BUILD OK:"
echo "  ${BUILD}/plainproof_gen   (CLI proof generator)"
echo "  ${BUILD}/kan              (unified: --pool / --solo)"
echo "  ${BUILD}/pearl-miner     (compat alias for legacy launchers)"
if [ -x "${BUILD}/zkprove" ]; then
  echo "  ${BUILD}/zkprove          (solo ZK-proof + block assembly)"
fi
# The pool binary is the deliverable; a Rust-less host legitimately has no zkprove.
# Never let the zkprove-presence test become the script's exit status - as the last
# command, a false `[ -x ]` would make build.sh exit 1 and fail the whole build
# stage on every pool-only (Rust-less) pipeline. Exit 0 explicitly on success.
exit 0
