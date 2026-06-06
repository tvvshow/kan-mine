#!/usr/bin/env bash
# Native build for the Pearl(PRL) self-built PoUW solver.
# Always builds the dp4a (CUDA-core) path; builds the tensor-core path (CUTLASS,
# requires sm_80+ and CUTLASS headers) when available, else links a stub.
# GPU compute capability is auto-detected so it runs on whatever GPU the CI host has.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo "=== toolchain ==="
nvcc --version | tail -2
g++ --version | head -1
echo "=== GPU ==="
nvidia-smi || echo "WARN: nvidia-smi unavailable (no GPU at build time?)"

# --- detect compute capability (e.g. 8.9 -> 89) ---
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
if ! [[ "${CC:-}" =~ ^[0-9]+$ ]]; then CC=75; echo "compute_cap undetected -> default 75 (sm_75)"; fi
ARCH="sm_${CC}"
echo "Target arch: ${ARCH} (compute_cap ${CC})"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
BUILD="${ROOT}/build"; mkdir -p "${BUILD}"; cd "${BUILD}"

echo "=== blake3 (portable; SIMD sources not vendored) ==="
BF="-DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512"
gcc -O3 -I"${ROOT}/blake3" ${BF} -c \
  "${ROOT}/blake3/blake3.c" "${ROOT}/blake3/blake3_dispatch.c" "${ROOT}/blake3/blake3_portable.c"

echo "=== host (plainproof_gen) ==="
g++ -O3 -std=c++17 -fopenmp -I"${ROOT}/blake3" -c "${ROOT}/src/plainproof_gen.cpp"

echo "=== dp4a kernel (jackpot_kernel) ==="
nvcc -O3 -arch="${ARCH}" -c "${ROOT}/src/jackpot_kernel.cu"

# --- tensor-core path (optional) ---
CUTLASS_DIR="${CUTLASS_DIR:-${ROOT}/cutlass}"
if [ "${CC}" -ge 80 ] && [ -d "${CUTLASS_DIR}/include" ]; then
  echo "=== tensor-core path (CUTLASS @ ${CUTLASS_DIR}) ==="
  nvcc -O3 -arch="${ARCH}" -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
    -I"${CUTLASS_DIR}/include" -I"${CUTLASS_DIR}/tools/util/include" -c "${ROOT}/src/tc_gemm.cu"
  TC=tc_gemm.o
else
  echo "=== tensor-core path STUBBED (need sm_80+ & CUTLASS; have ${ARCH}, CUTLASS_DIR=${CUTLASS_DIR}) ==="
  g++ -O3 -std=c++17 -c "${ROOT}/src/tc_stub.cpp" -o tc_gemm.o
  TC=tc_gemm.o
fi

echo "=== link ==="
g++ -O3 -fopenmp plainproof_gen.o blake3.o blake3_dispatch.o blake3_portable.o jackpot_kernel.o "${TC}" \
  -L"${CUDA_HOME}/lib64" -lcudart -o plainproof_gen
ls -la "${BUILD}/plainproof_gen"
echo "BUILD OK -> ${BUILD}/plainproof_gen"
