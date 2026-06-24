#!/usr/bin/env bash
# CNB / GPU-host validation entry point.
#
# This script is intentionally stricter than the normal CPU CI path:
#   1. require a real NVIDIA driver/GPU;
#   2. build the miner with CUTLASS on the GPU runner;
#   3. run the golden GPU smoke test;
#   4. run an easy-target real-cfg POSTCHECK gate;
#   5. optionally build and smoke-test the portable package on the same GPU;
#   6. optionally run a short live pool test when the operator explicitly
#      provides KAN_WALLET and GPU_VERIFY_POOL_SECONDS.
#
# It does NOT replace sm_86 production validation unless the assigned runner is
# actually sm_86.  CNB's public GPU runners are typically L40/H20-class shared
# datacenter GPUs, which are excellent for generic package smoke/correctness but
# not proof that the sm86-g8 tuned package is optimal on RTX 3080Ti/3090.
set -euo pipefail
cd "$(dirname "$0")"

GPU_VERIFY_MINE_DRAWS="${GPU_VERIFY_MINE_DRAWS:-20}"
GPU_VERIFY_PACKAGE_SMOKE="${GPU_VERIFY_PACKAGE_SMOKE:-1}"
GPU_VERIFY_POOL_SECONDS="${GPU_VERIFY_POOL_SECONDS:-0}"
GPU_VERIFY_REQUIRE_ACCEPTED="${GPU_VERIFY_REQUIRE_ACCEPTED:-0}"
KAN_POOL_URL="${KAN_POOL_URL:-stratum+tcp://prl.kryptex.network:7048}"
KAN_WALLET="${KAN_WALLET:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

run() {
  echo
  echo "=== $* ==="
  "$@"
}

echo "=== CNB GPU validation settings ==="
echo "GPU_VERIFY_MINE_DRAWS=${GPU_VERIFY_MINE_DRAWS}"
echo "GPU_VERIFY_PACKAGE_SMOKE=${GPU_VERIFY_PACKAGE_SMOKE}"
echo "GPU_VERIFY_POOL_SECONDS=${GPU_VERIFY_POOL_SECONDS}"
echo "GPU_VERIFY_REQUIRE_ACCEPTED=${GPU_VERIFY_REQUIRE_ACCEPTED}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found; this must run on a GPU runner"
run nvidia-smi

CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .' || true)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
[ -n "${CAP}" ] || fail "could not read GPU compute capability"
SM="sm_${CAP}"
echo "Detected GPU: ${GPU_NAME:-unknown} / ${SM}"

case "${SM}" in
  sm_80|sm_86|sm_89|sm_90)
    echo "GPU is within CUDA 12 production fatbin coverage."
    ;;
  sm_75)
    # The generic CUTLASS production package this script verifies does NOT cover
    # Turing: the Sm80 kernel needs cp.async and ~89 KB dynamic shared memory, and
    # build.sh's auto path would select CUTLASS here and fail to compile/launch on
    # sm_75. Turing IS supported, but via the SEPARATE WMMA "sm75" flavor package
    # (ARCH=sm_75 KERNEL=wmma PACKAGE_FLAVOR=sm75), validated independently on
    # 2x RTX 2080 Ti. This generic GPU-verify path therefore does not apply to Turing.
    fail "sm_75 / Turing: build+verify the dedicated WMMA sm75 flavor package instead; the generic CUTLASS verify does not run on Turing"
    ;;
  sm_120)
    echo "GPU is sm_120; CUDA 12 package will validate compute_90 PTX JIT fallback only."
    ;;
  sm_70)
    fail "sm_70 / Volta is intentionally unsupported by the current production release"
    ;;
  *)
    echo "WARNING: unprofiled GPU architecture ${SM}; continuing to expose launch/runtime failures."
    ;;
esac

echo
echo "=== toolchain ==="
nvcc --version | tail -2
g++ --version | head -1

run bash check_release_profiles.sh

if [ ! -d "${CUTLASS_HOME:-$HOME/cutlass}/include/cutlass" ]; then
  fail "CUTLASS headers missing; CNB pipeline should clone v3.5.1 before running this script"
fi

# Build native for the assigned GPU first.  This compile is not the release
# package; it exists to prove the checked-in source launches correctly on the
# GPU runner that CNB allocated.
run bash build.sh

run bash run_test.sh

EASY="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
HARD="0000000000000000000000000000000000000000000000000000000000000001"

echo
echo "=== real-cfg easy-target POSTCHECK ==="
./build/plainproof_gen --cfg real --mine 1 --target "${EASY}" 777 \
  >/tmp/kan_cnb_real_easy.b64 2>/tmp/kan_cnb_real_easy.log
tail -40 /tmp/kan_cnb_real_easy.log
grep -qE 'POSTCHECK .*ok=1|POSTCHECK.*ok=1' /tmp/kan_cnb_real_easy.log \
  || fail "real-cfg easy-target POSTCHECK did not report ok=1"
test -s /tmp/kan_cnb_real_easy.b64 || fail "real-cfg easy-target proof output is empty"

echo
echo "=== controlled hard-target timing sample (${GPU_VERIFY_MINE_DRAWS} draws, no win required) ==="
set +e
TC_TIMING=1 ./build/plainproof_gen --cfg real --mine "${GPU_VERIFY_MINE_DRAWS}" \
  --target "${HARD}" --breakdown 777 >/tmp/kan_cnb_hard.out 2>/tmp/kan_cnb_hard.log
HARD_RC=$?
set -e
echo "hard-target rc=${HARD_RC}"
grep -E 'prep|search|total|MINE done|FUSED|TH/s|POSTCHECK|CUDA|ERROR' \
  /tmp/kan_cnb_hard.log /tmp/kan_cnb_hard.out 2>/dev/null | tail -80 || true

if [ "${GPU_VERIFY_PACKAGE_SMOKE}" = "1" ]; then
  echo
  echo "=== portable package build + package smoke on assigned GPU ==="
  run env WITH_AB=0 bash package_portable.sh

  rm -rf /tmp/kan-cnb-package
  mkdir -p /tmp/kan-cnb-package
  tar xzf dist/kan-portable-linux-x64.tar.gz -C /tmp/kan-cnb-package
  cd /tmp/kan-cnb-package/kan-portable-linux-x64

  echo "--- package BUILD_INFO.txt ---"
  cat BUILD_INFO.txt
  echo "--- package status ---"
  ./status.sh || true

  ./plainproof_gen --cfg real --mine 1 --target "${EASY}" 777 \
    >/tmp/kan_cnb_pkg_easy.b64 2>/tmp/kan_cnb_pkg_easy.log
  tail -40 /tmp/kan_cnb_pkg_easy.log
  grep -qE 'POSTCHECK .*ok=1|POSTCHECK.*ok=1' /tmp/kan_cnb_pkg_easy.log \
    || fail "portable package POSTCHECK did not report ok=1"
  test -s /tmp/kan_cnb_pkg_easy.b64 || fail "portable package proof output is empty"

  cd - >/dev/null
fi

if [ "${GPU_VERIFY_POOL_SECONDS}" != "0" ]; then
  echo
  echo "=== optional live pool smoke (${GPU_VERIFY_POOL_SECONDS}s) ==="
  [ -n "${KAN_WALLET}" ] || fail "GPU_VERIFY_POOL_SECONDS set but KAN_WALLET is empty"

  PKG_DIR="/tmp/kan-cnb-package/kan-portable-linux-x64"
  if [ ! -x "${PKG_DIR}/run.sh" ]; then
    rm -rf /tmp/kan-cnb-package
    mkdir -p /tmp/kan-cnb-package
    tar xzf dist/kan-portable-linux-x64.tar.gz -C /tmp/kan-cnb-package
  fi

  set +e
  (
    cd "${PKG_DIR}" &&
    timeout "${GPU_VERIFY_POOL_SECONDS}s" env KAN_RESTART=0 TC_TIMING=1 \
      ./run.sh --algo pearl --pool "${KAN_POOL_URL}" --wallet "${KAN_WALLET}" \
      --worker cnb --batch 1000 --cfg real --tc
  ) > /tmp/kan_cnb_pool.log 2>&1
  POOL_RC=$?
  set -e
  echo "pool rc=${POOL_RC} (124 means timeout after requested smoke duration)"
  tail -120 /tmp/kan_cnb_pool.log
  grep -q "async share submit worker active" /tmp/kan_cnb_pool.log \
    || fail "pool smoke did not reach async submit worker startup"
  if [ "${GPU_VERIFY_REQUIRE_ACCEPTED}" = "1" ]; then
    grep -q "share accepted" /tmp/kan_cnb_pool.log \
      || fail "pool smoke required accepted share but none was observed"
  fi
fi

echo
echo "CNB GPU VERIFY PASS: source build, GPU smoke, real-cfg POSTCHECK, and package smoke completed."
