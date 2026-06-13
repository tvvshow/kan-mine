#!/usr/bin/env bash
# Build a "download & run" portable Pearl(PRL) miner package.
#
# WHY: every fresh GPU box otherwise needs CUDA toolkit + CUTLASS clone + a full
# CUTLASS compile (minutes). This produces ONE self-contained tarball whose only
# runtime dependency is the NVIDIA driver (libcuda, present on any GPU host):
#   * cudart STATICALLY linked    -> no CUDA toolkit needed on the target
#   * libstdc++/libgcc static     -> no GCC runtime version mismatch
#   * libssl/libcrypto/libgomp... -> bundled into ./lib, found via $ORIGIN/lib rpath
#   * CUTLASS/BLAKE3 are BUILD-time only -> absent from the package entirely
#
# Build host: any Linux + CUDA toolkit, NO GPU required (CUDA code compiles
# GPU-less). Build on the OLDEST glibc you must support — the cnb runner
# (ubuntu22.04, glibc 2.35) covers Ubuntu 22.04+ GPU containers. CUDA 12.4 nvcc
# emits sm_75..sm_90 SASS + compute_90 PTX, so a 5090 (sm_120) JITs the PTX at
# runtime (~336 TH/s); for native sm_120 SASS (~350) build on CUDA 13.
#
# Usage:
#   bash package_portable.sh                 # auto-arch (portable fatbin)
#   ARCH=sm_120 bash package_portable.sh     # native single-arch
#   WITH_AB=0 bash package_portable.sh       # skip the A1-vs-RMW A/B binary
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

PKG="kan-portable-linux-x64"
DIST="${ROOT}/dist"
STAGE="${DIST}/${PKG}"
LIBDIR="${STAGE}/lib"
WITH_AB="${WITH_AB:-1}"
rm -rf "${STAGE}"; mkdir -p "${LIBDIR}/../bench"

echo "=== [1/5] build portable binaries ==="
# ARCH (if the caller set it) flows to build.sh through the environment, so we do
# NOT inline-pass it: `PORTABLE=1 ${ARCH:+ARCH=$ARCH} ...` misparses — a $-leading
# word isn't recognized as an assignment prefix, so when ARCH is empty the NEXT
# assignment becomes the command ("NVCC_EXTRA=...: command not found").
if [ "${WITH_AB}" = "1" ]; then
  echo "  [1a] pre-A1 RMW baseline (for on-box A/B) ..."
  PORTABLE=1 NVCC_EXTRA="-DFOLD_RMW_ALWAYS" bash build.sh
  cp -f build/plainproof_gen "${STAGE}/plainproof_gen_rmw"
fi
echo "  [1b] A1 (default) -> the shipped kan + plainproof_gen ..."
PORTABLE=1 bash build.sh
cp -f build/kan build/plainproof_gen "${STAGE}/"

echo "=== [2/5] bundle non-glibc / non-driver shared libs (fixpoint over ldd) ==="
# Keep as SYSTEM (never bundle): glibc core (ABI-stable, everywhere) + the NVIDIA
# driver (host-specific, must come from the target's installed driver).
is_system() { case "$1" in
  libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|ld-linux*|libresolv.so.*|\
  libnsl.so.*|libgcc_s.so.*|libcuda.so.*|libnvidia*|linux-vdso*) return 0;; *) return 1;; esac; }
scan() { ldd "$1" 2>/dev/null | sed -n 's/.*=> \(\/[^ ]*\).*/\1/p; s/^[[:space:]]*\(\/[^ ]*\) (.*/\1/p'; }
work="${STAGE}/kan ${STAGE}/plainproof_gen"
[ "${WITH_AB}" = "1" ] && work="$work ${STAGE}/plainproof_gen_rmw"
added=1
while [ "$added" = 1 ]; do
  added=0; next=""
  for f in $work; do
    for so in $(scan "$f"); do
      [ -f "$so" ] || continue
      b="$(basename "$so")"
      is_system "$b" && continue
      [ -f "${LIBDIR}/${b}" ] && continue
      cp -L "$so" "${LIBDIR}/${b}"; echo "  bundled ${b}"
      added=1; next="$next ${LIBDIR}/${b}"
    done
  done
  work="$next"
done
[ -z "$(ls -A "${LIBDIR}" 2>/dev/null)" ] && echo "  (nothing to bundle — fully static)"

echo "=== [3/5] launcher + bench + README ==="
cat > "${STAGE}/run.sh" <<'LAUNCH'
#!/usr/bin/env bash
# Portable Pearl(PRL) miner. ONLY runtime dep = the NVIDIA driver (any GPU host
# has it). rpath $ORIGIN/lib already locates the bundled libs; this wrapper just
# checks the driver and forwards all args to ./kan.
here="$(cd "$(dirname "$0")" && pwd)"
if ! ls /dev/nvidia0 >/dev/null 2>&1 && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: no NVIDIA GPU/driver detected — this is a CUDA GPU miner." >&2
fi
exec "$here/kan" "$@"
LAUNCH
chmod +x "${STAGE}/run.sh"

cat > "${STAGE}/bench/ablate_a1_prebuilt.sh" <<'ABLATE'
#!/usr/bin/env bash
# Phase A1 (fold direct-store) A/B with NO toolchain — uses the two prebuilt
# binaries shipped in this package. A1 MUST be byte-identical to RMW for REAL
# (same proof sha256) and POSTCHECK ok=1 on both; reports kernel Δ%.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${SEED:-777}"; NDRAW="${NDRAW:-3}"
EASY="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
HARD="0000000000000000000000000000000000000000000000000000000000000001"
[ -x "$here/plainproof_gen_rmw" ] || { echo "no plainproof_gen_rmw (package built WITH_AB=0)"; exit 2; }
ab() { local L="$1" BIN="$2"
  "$BIN" --cfg real --mine 1 --target "$EASY" "$SEED" >/tmp/a1_$L.b64 2>/tmp/a1_$L.corr
  local ok; ok="$(grep -oE 'POSTCHECK ok=[01]' /tmp/a1_$L.corr | tail -1)"
  local sha; sha="$(sha256sum /tmp/a1_$L.b64 | awk '{print $1}')"
  "$BIN" --cfg real --mine "$NDRAW" --target "$HARD" --breakdown "$SEED" >/dev/null 2>/tmp/a1_$L.time
  local ms; ms="$(grep -oE 'FUSED [0-9]+ tiles, [0-9.]+ ms' /tmp/a1_$L.time | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+' | tail -1)"
  echo "  $L: ${ok:-<none>}  kernel=${ms} ms  proof=${sha:0:16}…"
  eval "MS_$L=$ms; SHA_$L='$sha'; OK_$L='$ok'"
}
echo "=== A1 vs RMW (prebuilt, SEED=$SEED) ==="
ab RMW "$here/plainproof_gen_rmw"
ab A1  "$here/plainproof_gen"
fail=0
[ "${OK_RMW:-}" = "POSTCHECK ok=1" ] || { echo "  ✗ RMW POSTCHECK"; fail=1; }
[ "${OK_A1:-}"  = "POSTCHECK ok=1" ] || { echo "  ✗ A1 POSTCHECK"; fail=1; }
[ "${SHA_RMW:-}" = "${SHA_A1:-}" ] && echo "  ✓ proof byte-identical" || { echo "  ✗ PROOF DIVERGED — revert A1"; fail=1; }
if [ "$fail" = 0 ] && [ -n "${MS_RMW:-}" ] && [ -n "${MS_A1:-}" ]; then
  awk "BEGIN{printf \"  RMW %.1f TH/s | A1 %.1f TH/s | Δ %+.2f%% kernel\n\",70368.744/$MS_RMW,70368.744/$MS_A1,($MS_RMW-$MS_A1)/$MS_RMW*100}"
fi
exit $fail
ABLATE
chmod +x "${STAGE}/bench/ablate_a1_prebuilt.sh"

WALLET="prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv.pm"
cat > "${STAGE}/README.txt" <<README
Pearl(PRL) portable miner — download & run
===========================================
Built on cnb (Ubuntu 22.04 / glibc 2.35 / CUDA 12.4). Runs on any Linux x86-64
GPU host with glibc >= 2.35 and an NVIDIA driver. NO CUDA toolkit, NO CUTLASS,
NO compiler needed on this machine.

Quick start (pool mining):
  tar xzf ${PKG}.tar.gz && cd ${PKG}
  ./run.sh --algo pearl \\
    --pool stratum+tcp://prl.kryptex.network:7048 \\
    --wallet ${WALLET} \\
    --worker pm --batch 500 --cfg real --tc

Kernel benchmark (no pool):
  ./plainproof_gen --cfg real --mine 15 --tc --breakdown

Files:
  kan                 unified miner (--pool / --solo)
  plainproof_gen      CLI proof generator + kernel benchmark (Phase A1 build)
  plainproof_gen_rmw  pre-A1 baseline (present only if built WITH_AB=1)
  bench/ablate_a1_prebuilt.sh   on-box A1-vs-RMW A/B, no toolchain
  lib/                bundled libssl/libcrypto/libgomp... (found via rpath)
  run.sh              launcher (driver check + forwards args to kan)

GPU arch note:
  CUDA 12.4 nvcc emits sm_75..sm_90 SASS + compute_90 PTX. On Blackwell (RTX
  5090, sm_120) the driver JITs the PTX at first launch (~336 TH/s). For native
  sm_120 SASS (~350) rebuild the package on a CUDA 13 host with ARCH=sm_120.

Verify it's truly portable (should list only glibc + libcuda as external):
  ldd ./kan
README

echo "=== [4/5] portability proof (ldd of staged kan) ==="
ldd "${STAGE}/kan" || true
echo "  ^ everything above must be glibc (libc/libm/ld-linux/...) or ./lib/* or"
echo "    libcuda.so.* (the driver, supplied by the GPU host). Anything else = a"
echo "    missed bundle — add it to package_portable.sh."

echo "=== [5/5] tar ==="
( cd "${DIST}" && tar czf "${PKG}.tar.gz" "${PKG}" )
ls -la "${DIST}/${PKG}.tar.gz"
du -sh "${STAGE}" | awk '{print "  unpacked size: "$1}'
echo "PORTABLE PACKAGE OK -> dist/${PKG}.tar.gz"
