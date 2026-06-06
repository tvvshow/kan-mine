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

echo "=== host (plainproof_gen) ==="
g++ -O3 -std=c++17 -fopenmp -I"${ROOT}/blake3" -c "${ROOT}/src/plainproof_gen.cpp"

echo "=== dp4a kernel (jackpot_kernel) ==="
# shellcheck disable=SC2086
nvcc -O3 ${GENCODE} -c "${ROOT}/src/jackpot_kernel.cu"

# --- tensor-core path (optional; needs CUTLASS headers + explicit sm_80+ ARCH) ---
CUTLASS_DIR="${CUTLASS_DIR:-${ROOT}/cutlass}"
TC_OK=0
case "${ARCH:-}" in sm_8*|sm_9*|sm_10*|sm_12*) TC_OK=1 ;; esac
if [ "${TC_OK}" = "1" ] && [ -d "${CUTLASS_DIR}/include" ]; then
  echo "=== tensor-core path (CUTLASS @ ${CUTLASS_DIR}, ${ARCH}) ==="
  nvcc -O3 -arch="${ARCH}" -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
    -I"${CUTLASS_DIR}/include" -I"${CUTLASS_DIR}/tools/util/include" -c "${ROOT}/src/tc_gemm.cu"
  TC=tc_gemm.o
else
  echo "=== tensor-core path STUBBED (need CUTLASS headers + sm_80+ ARCH override) ==="
  g++ -O3 -std=c++17 -c "${ROOT}/src/tc_stub.cpp" -o tc_gemm.o
  TC=tc_gemm.o
fi

echo "=== link ==="
g++ -O3 -fopenmp plainproof_gen.o blake3.o blake3_dispatch.o blake3_portable.o jackpot_kernel.o "${TC}" \
  -L"${CUDA_HOME}/lib64" -lcudart -o plainproof_gen
ls -la "${BUILD}/plainproof_gen"
echo "BUILD OK -> ${BUILD}/plainproof_gen"
