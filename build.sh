#!/usr/bin/env bash
# Native build for the Pearl(PRL) self-built PoUW solver.
#
# CUDA code COMPILES without a GPU present (only *running* kernels needs one), so
# this build NEVER depends on nvidia-smi. Arch selection order:
#   ARCH env override  >  nvidia-smi detection (only if a GPU is present)  >
#   portable multi-arch fatbin (Turing..Hopper SASS + Hopper PTX for Blackwell JIT).
# The tensor-core path (CUTLASS, sm_80+) is built only when CUTLASS headers are
# vendored AND an explicit sm_80+ ARCH override is given; otherwise a stub links.
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
  echo "no nvidia-smi (GPU-less build host) — OK, compiling anyway"
fi

# --- choose nvcc arch flags (this block must never fail the build) ---
GENCODE=""
if [ -n "${ARCH:-}" ]; then
  GENCODE="-arch=${ARCH}"
  echo "arch: ENV override -> ${ARCH}"
elif command -v nvidia-smi >/dev/null 2>&1; then
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

echo "=== blake3 (portable; SIMD sources not vendored) ==="
BF="-DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512"
gcc -O3 -I"${ROOT}/blake3" ${BF} -c \
  "${ROOT}/blake3/blake3.c" "${ROOT}/blake3/blake3_dispatch.c" "${ROOT}/blake3/blake3_portable.c"

echo "=== host: prover core ==="
# Two object files from the SAME proven pipeline source:
#   plainproof_gen.o : keeps its own main() -> the standalone CLI
#   prover_lib.o     : -DPROVER_LIB drops main() -> linked into pearl-miner
g++ -O3 -std=c++17 -fopenmp -I"${ROOT}/blake3" -c "${ROOT}/src/plainproof_gen.cpp" -o plainproof_gen.o
g++ -O3 -std=c++17 -fopenmp -I"${ROOT}/blake3" -DPROVER_LIB -c "${ROOT}/src/plainproof_gen.cpp" -o prover_lib.o

echo "=== host: unified miner driver (pool + solo) ==="
g++ -O3 -std=c++17 -I"${ROOT}/src" -I"${ROOT}/blake3" -c "${ROOT}/src/miner_main.cpp" -o miner_main.o

echo "=== dp4a kernel (jackpot_kernel) ==="
# shellcheck disable=SC2086
nvcc -O3 ${GENCODE} -c "${ROOT}/src/jackpot_kernel.cu"

# --- fused tensor-core path (M2a: WMMA int8, NO CUTLASS) ---------------------
# tc_gemm.cu is a self-contained WMMA (16x16x16 s8) fused jackpot kernel. int8 WMMA
# needs sm_72+, and every GENCODE target above (sm_75..sm_90 SASS + PTX) qualifies,
# so we build it unconditionally with the SAME arch flags as the dp4a kernel — no
# CUTLASS headers, no separate ARCH override, no stub.
echo "=== fused tensor-core kernel (tc_gemm, WMMA int8) ==="
# shellcheck disable=SC2086
nvcc -O3 ${GENCODE} -std=c++17 -c "${ROOT}/src/tc_gemm.cu"
TC=tc_gemm.o

echo "=== link: plainproof_gen (CLI) ==="
g++ -O3 -fopenmp plainproof_gen.o blake3.o blake3_dispatch.o blake3_portable.o jackpot_kernel.o "${TC}" \
  -L"${CUDA_HOME}/lib64" -lcudart -o plainproof_gen
ls -la "${BUILD}/plainproof_gen"

echo "=== link: pearl-miner (unified pool + solo) ==="
# OpenSSL for solo HTTPS RPC; pthread for the stratum/poller threads.
g++ -O3 -fopenmp miner_main.o prover_lib.o blake3.o blake3_dispatch.o blake3_portable.o jackpot_kernel.o "${TC}" \
  -L"${CUDA_HOME}/lib64" -lcudart -lssl -lcrypto -lpthread -o pearl-miner
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
echo "  ${BUILD}/pearl-miner      (unified: --pool / --solo)"
if [ -x "${BUILD}/zkprove" ]; then
  echo "  ${BUILD}/zkprove          (solo ZK-proof + block assembly)"
fi
# The pool binary is the deliverable; a Rust-less host legitimately has no zkprove.
# Never let the zkprove-presence test become the script's exit status — as the last
# command, a false `[ -x ]` would make build.sh exit 1 and fail the whole build
# stage on every pool-only (Rust-less) pipeline. Exit 0 explicitly on success.
exit 0
