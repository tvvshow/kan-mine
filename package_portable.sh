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
# emits sm_75..sm_90 SASS + compute_90 PTX. Blackwell / sm_120 needs a CUDA 13
# native package for best performance; CUDA 12 packages can only rely on PTX JIT.
#
# Usage:
#   bash package_portable.sh                 # auto-arch (portable fatbin)
#   ARCH=sm_120 bash package_portable.sh     # native single-arch
#   WITH_AB=0 bash package_portable.sh       # skip the A1-vs-RMW A/B binary
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

# Version/release metadata:
# - On tag builds this resolves to the pushed tag (for example v1.2.1).
# - On main/dev builds it resolves to vX.Y.Z-N-g<sha>[-dirty].
# The package keeps a stable directory name for easy `cd`, but emits both a
# versioned tarball and a stable CI attachment alias.
VERSION="${VERSION:-$(git describe --tags --dirty --always 2>/dev/null || echo dev)}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUILD_DATE="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
SAFE_VERSION="$(printf '%s' "${VERSION}" | tr '/ :' '---')"
PACKAGE_FLAVOR="${PACKAGE_FLAVOR:-}"
SAFE_FLAVOR="$(printf '%s' "${PACKAGE_FLAVOR}" | tr '/ :' '---')"
PKG="kan-portable-linux-x64"
if [ -n "${SAFE_FLAVOR}" ]; then
  VERSIONED_TAR="${PKG}-${SAFE_VERSION}-${SAFE_FLAVOR}.tar.gz"
  STABLE_TAR="${PKG}-${SAFE_FLAVOR}.tar.gz"
else
  VERSIONED_TAR="${PKG}-${SAFE_VERSION}.tar.gz"
  STABLE_TAR="${PKG}.tar.gz"
fi
DIST="${ROOT}/dist"
STAGE="${DIST}/${PKG}"
LIBDIR="${STAGE}/lib"
WITH_AB="${WITH_AB:-1}"

NVCC_VERSION="$(nvcc --version 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//' || true)"
CUDA_VERSION="$(printf '%s\n' "${NVCC_VERSION}" | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')"
[ -n "${NVCC_VERSION}" ] || NVCC_VERSION="unknown"
[ -n "${CUDA_VERSION}" ] || CUDA_VERSION="unknown"

# Mirror build.sh defaults so BUILD_INFO.txt records the actual compile-time
# knobs instead of vague "build-default" placeholders. These are compile-time
# parameters: target machines must select a prebuilt flavor package rather than
# trying to change them at runtime.
if [ -z "${GROUPM+x}" ] || [ -z "${GROUPM}" ]; then
  case "${ARCH:-}" in
    sm_80|sm_86) GROUPM=8 ;;
    *) GROUPM=128 ;;
  esac
fi
KSTAGES="${KSTAGES:-3}"
EFFECTIVE_SMALL_TILE="${SMALL_TILE:-0}"
export GROUPM KSTAGES

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
# Compatibility alias for old scripts/docs that still launch pearl-miner. The
# production binary remains `kan`; this is intentionally just the same build.
cp -f build/pearl-miner "${STAGE}/pearl-miner" 2>/dev/null || cp -f build/kan "${STAGE}/pearl-miner"
chmod +x "${STAGE}/kan" "${STAGE}/pearl-miner" "${STAGE}/plainproof_gen"
[ -f "${STAGE}/plainproof_gen_rmw" ] && chmod +x "${STAGE}/plainproof_gen_rmw"
[ -f "${ROOT}/GPU_PROFILES.md" ] && cp -f "${ROOT}/GPU_PROFILES.md" "${STAGE}/GPU_PROFILES.md"
[ -f "${ROOT}/CHANGELOG.md" ] && cp -f "${ROOT}/CHANGELOG.md" "${STAGE}/CHANGELOG.md"
[ -f "${ROOT}/install_kan.sh" ] && cp -f "${ROOT}/install_kan.sh" "${STAGE}/install_kan.sh" && chmod +x "${STAGE}/install_kan.sh"

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

# Give bundled libs a $ORIGIN rpath so a sibling dep (e.g. libssl -> libcrypto)
# resolves INSIDE lib/ no matter how the binary is launched; otherwise the
# transitive dep silently falls back to the host's system copy. patchelf is
# tiny; install best-effort (we're root on the build host).
if [ -n "$(ls -A "${LIBDIR}" 2>/dev/null)" ]; then
  command -v patchelf >/dev/null 2>&1 || { apt-get install -y --no-install-recommends patchelf >/dev/null 2>&1 || true; }
  if command -v patchelf >/dev/null 2>&1; then
    for so in "${LIBDIR}"/*; do patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true; done
    echo "  patchelf: \$ORIGIN rpath on bundled libs (siblings resolve within lib/)"
  else
    echo "  WARNING: patchelf unavailable — bundled libs may use system transitive deps"
  fi
fi

echo "=== [3/5] launcher + bench + README ==="
cat > "${STAGE}/run.sh" <<'LAUNCH'
#!/usr/bin/env bash
# Portable Pearl(PRL) miner. ONLY runtime dep = the NVIDIA driver (any GPU host
# has it). rpath $ORIGIN/lib already locates the bundled libs.  For pool mining
# this wrapper also provides the production restart loop: the miner exits on
# pool disconnect/job error, so unattended portable deployments must reconnect.
here="$(cd "$(dirname "$0")" && pwd)"
if ! ls /dev/nvidia0 >/dev/null 2>&1 && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: no NVIDIA GPU/driver detected — this is a CUDA GPU miner." >&2
fi

# Ampere sm_86 packages measured faster on the 3080Ti test box with the
# non-persistent launch path.  Keep this as a package/runtime default instead of
# baking it into the kernel so operators can still override:
#   TC_PERSIST=1 ./run.sh ...
if [ -z "${TC_PERSIST+x}" ]; then
  arch="$(awk -F': ' '/^arch:/ {print $2; exit}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
  flavor="$(awk -F': ' '/^package_flavor:/ {print $2; exit}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
  case "$arch:$flavor" in
    sm_86:*|*:sm86-*) export TC_PERSIST=0 ;;
  esac
fi

if [ "${KAN_SHOW_BUILD_INFO:-1}" != "0" ]; then
  ver="$(cat "$here/VERSION" 2>/dev/null || echo unknown)"
  commit="$(awk -F': ' '/^commit:/ {print $2; exit}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
  flavor="$(awk -F': ' '/^package_flavor:/ {print $2; exit}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
  arch="$(awk -F': ' '/^arch:/ {print $2; exit}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
  groupm="$(awk -F': ' '/^groupm:/ {print $2; exit}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
  kstages="$(awk -F': ' '/^kstages:/ {print $2; exit}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
  gpu="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
  {
    echo "=== Kan portable package ==="
    echo "version=${ver} commit=${commit:-unknown}"
    echo "flavor=${flavor:-generic} arch=${arch:-unknown} groupm=${groupm:-unknown} kstages=${kstages:-unknown}"
    echo "TC_PERSIST=${TC_PERSIST:-<unset>} TC_TIMING=${TC_TIMING:-0} KAN_RESTART=${KAN_RESTART:-auto}"
    [ -n "$gpu" ] && echo "gpu=${gpu}"
    echo "============================"
  } >&2
fi

is_pool=0
for arg in "$@"; do
  case "$arg" in
    --pool|--pool=*) is_pool=1 ;;
  esac
done

# Production default:
#   * pool mode: restart forever on disconnect/job-error (KAN_RESTART=auto)
#   * non-pool/help/solo modes: one-shot exec
# Operators can force one-shot pool runs with:
#   KAN_RESTART=0 ./run.sh --algo pearl --pool ...
# and adjust the delay with KAN_RESTART_DELAY=15.
restart="${KAN_RESTART:-auto}"
case "$restart:$is_pool" in
  1:*|true:*|yes:*|always:*|auto:1)
    delay="${KAN_RESTART_DELAY:-15}"
    while true; do
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Kan starting: $here/kan $*" >&2
      "$here/kan" "$@"
      rc=$?
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Kan exited (rc=$rc), restarting in ${delay}s; set KAN_RESTART=0 for one-shot." >&2
      sleep "$delay"
    done
    ;;
  *)
    exec "$here/kan" "$@"
    ;;
esac
LAUNCH
chmod +x "${STAGE}/run.sh"

cat > "${STAGE}/status.sh" <<'STATUS'
#!/usr/bin/env bash
# Human-friendly one-shot status for a portable Kan miner directory.
# Usage:
#   ./status.sh                  # auto-detect common package/local logs
#   LOG=/path/to/stress.log ./status.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
log="${LOG:-}"
if [ -z "$log" ]; then
  for cand in "$here/live_current.log" "$here/miner.log" \
              "$here/stress_24h.log" "$here/../stress_24h.log" \
              "/tmp/fast.log"; do
    [ -f "$cand" ] && { log="$cand"; break; }
  done
fi
ver="$(cat "$here/VERSION" 2>/dev/null || echo unknown)"
commit="$(awk -F': ' '/^commit:/ {print $2}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
flavor="$(awk -F': ' '/^package_flavor:/ {print $2}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
arch="$(awk -F': ' '/^arch:/ {print $2}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
groupm="$(awk -F': ' '/^groupm:/ {print $2}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"
kstages="$(awk -F': ' '/^kstages:/ {print $2}' "$here/BUILD_INFO.txt" 2>/dev/null || true)"

echo "=== Kan portable status ==="
echo "dir:     $here"
echo "version: $ver"
echo "commit:  ${commit:-unknown}"
echo "flavor:  ${flavor:-unknown}  arch=${arch:-unknown}  groupm=${groupm:-unknown}  kstages=${kstages:-unknown}"
echo "runtime: TC_PERSIST=${TC_PERSIST:-<unset>}  TC_TIMING=${TC_TIMING:-0}  KAN_RESTART=${KAN_RESTART:-auto}"
echo "log:     ${log:-<not found>}"
echo

echo "=== process ==="
found_proc=0
for pid in $(pgrep -f 'kan .*--algo pearl|run_24h.sh|run.sh' 2>/dev/null || true); do
  [ -r "/proc/$pid/cwd" ] || continue
  cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
  cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  case "$cwd" in
    "$here"|"$here"/*)
      echo "$pid  cwd=$cwd  $cmd"
      found_proc=1
      ;;
  esac
done
[ "$found_proc" = 1 ] || echo "not running from this directory"
echo

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,temperature.gpu,fan.speed,power.draw,power.limit,utilization.gpu,clocks.sm,clocks.mem,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null |
  awk -F', ' '{
    printf "gpu: %s | cc=%s driver=%s temp=%sC fan=%s%% power=%s/%sW util=%s%% sm=%sMHz memclk=%sMHz vram=%s/%sMiB\n",
           $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12
  }'
else
  echo "nvidia-smi not found"
fi
echo

if [ -n "$log" ] && [ -f "$log" ]; then
  echo "=== latest hashrate table ==="
  awk '
    /DEVICE MODEL/{buf=$0; cap=1; next}
    cap{buf=buf "\n" $0; if ($0 ~ /accept - ver\./){last=buf; cap=0}}
    END{if(last!="") print last; else print "hashrate table not seen yet"}
  ' "$log"
  echo
  echo "=== shares ==="
  acc="$(grep -ic "share accepted" "$log" || true)"
  rej="$(grep -Eic "share rejected|rejected" "$log" || true)"
  tout="$(grep -ic "share submit timeout" "$log" || true)"
  echo "accepted=$acc rejected=$rej submit_timeouts=$tout"
  grep -E "share accepted|share rejected|share submit timeout" "$log" | tail -10 || true
  echo
  echo "=== runtime events ==="
  async="$(grep -ic "async share submit worker active" "$log" || true)"
  stale_abort="$(grep -ic "MINE proof abort" "$log" || true)"
  echo "async_submit_worker_seen=$async stale_proof_aborts=$stale_abort"
  grep -E "async share submit worker active|MINE proof abort" "$log" | tail -10 || true
  echo
  echo "=== latest perf attempts ==="
  grep -E " info +perf +" "$log" | tail -12 || echo "no perf lines yet"
  echo
  echo "=== latest log tail ==="
  tail -30 "$log"
else
  echo "log file not found. Set LOG=/path/to/stress_24h.log ./status.sh"
fi
STATUS
chmod +x "${STAGE}/status.sh"

cat > "${STAGE}/bench/ablate_a1_prebuilt.sh" <<'ABLATE'
#!/usr/bin/env bash
# Phase A1 (fold direct-store) A/B with NO toolchain — uses the two prebuilt
# binaries shipped in this package. Correctness gate = POSTCHECK ok=1 on BOTH
# (CPU recomputes the GPU's winning tile). Reports kernel Δ%.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${SEED:-777}"; NDRAW="${NDRAW:-3}"
EASY="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
HARD="0000000000000000000000000000000000000000000000000000000000000001"
[ -x "$here/plainproof_gen_rmw" ] || { echo "no plainproof_gen_rmw (package built WITH_AB=0)"; exit 2; }
ab() { local L="$1" BIN="$2"
  "$BIN" --cfg real --mine 1 --target "$EASY" "$SEED" >/tmp/a1_$L.b64 2>/tmp/a1_$L.corr
  local ok; ok="$(grep -E 'POSTCHECK' /tmp/a1_$L.corr | grep -oE 'ok=[01]' | tail -1)"
  local sha; sha="$(sha256sum /tmp/a1_$L.b64 | awk '{print $1}')"
  "$BIN" --cfg real --mine "$NDRAW" --target "$HARD" --breakdown "$SEED" >/dev/null 2>/tmp/a1_$L.time
  local ms; ms="$(grep -oE 'FUSED [0-9]+ tiles, [0-9.]+ ms' /tmp/a1_$L.time | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+' | tail -1)"
  echo "  $L: POSTCHECK ${ok:-<none>}  kernel=${ms} ms  proof=${sha:0:16}…"
  eval "MS_$L=$ms; SHA_$L='$sha'; OK_$L='$ok'"
}
echo "=== A1 vs RMW (prebuilt, SEED=$SEED) ==="
ab RMW "$here/plainproof_gen_rmw"
ab A1  "$here/plainproof_gen"
# Gate on POSTCHECK ok=1, NOT proof identity: the easy target makes EVERY tile a
# winner, so which one atomicCAS records is racy and differs run-to-run (even
# RMW vs RMW). ok=1 means the CPU re-verified the GPU's winning jackpot tile.
fail=0
[ "${OK_RMW:-}" = "ok=1" ] || { echo "  ✗ RMW POSTCHECK not ok=1"; fail=1; }
[ "${OK_A1:-}"  = "ok=1" ] || { echo "  ✗ A1 POSTCHECK not ok=1"; fail=1; }
[ "$fail" = 0 ] && echo "  ✓ A1 + RMW both POSTCHECK ok=1 (A1 jackpot transcript verified on real GPU)"
[ "${SHA_RMW:-}" != "${SHA_A1:-}" ] && echo "  (proofs differ — expected: easy-target winner is racy, not an A1 fault)"
if [ -n "${MS_RMW:-}" ] && [ -n "${MS_A1:-}" ]; then
  awk "BEGIN{printf \"  RMW %.1f TH/s | A1 %.1f TH/s | Δ %+.2f%% kernel\n\",70368.744/$MS_RMW,70368.744/$MS_A1,($MS_RMW-$MS_A1)/$MS_RMW*100}"
fi
exit $fail
ABLATE
chmod +x "${STAGE}/bench/ablate_a1_prebuilt.sh"

cat > "${STAGE}/VERSION" <<EOF
${VERSION}
EOF

cat > "${STAGE}/BUILD_INFO.txt" <<EOF
version: ${VERSION}
commit: ${COMMIT}
built_utc: ${BUILD_DATE}
package_dir: ${PKG}
versioned_tar: ${VERSIONED_TAR}
stable_tar_alias: ${STABLE_TAR}
with_ab: ${WITH_AB}
portable: 1
arch: ${ARCH:-portable-fatbin}
groupm: ${GROUPM}
kstages: ${KSTAGES}
small_tile: ${EFFECTIVE_SMALL_TILE}
package_flavor: ${PACKAGE_FLAVOR:-generic}
package_policy: $([ -n "${PACKAGE_FLAVOR:-}" ] && echo tuned || echo generic-compatible)
toolchain: CUDA ${CUDA_VERSION} portable build; static cudart/libstdc++; bundled non-glibc shared libs
cuda_version: ${CUDA_VERSION}
nvcc_version: ${NVCC_VERSION}
runtime_dependency: NVIDIA driver + Linux x86-64 glibc >= 2.35
EOF

release_notes="${STAGE}/RELEASE_NOTES.txt"
{
  echo "Kan portable release notes"
  echo "=========================="
  echo
  echo "Version: ${VERSION}"
  echo "Commit:  ${COMMIT}"
  echo "Built:   ${BUILD_DATE}"
  echo "Flavor:  ${PACKAGE_FLAVOR:-generic}"
  echo
  echo "Package profile:"
  echo "----------------"
  case "${PACKAGE_FLAVOR:-generic}" in
    generic)
      echo "Generic compatibility package for NVIDIA RTX 20 series / Turing and newer."
      echo "Covers sm_75, sm_80, sm_86, sm_89, sm_90 SASS plus compute_90 PTX."
      echo "Volta / sm_70 (V100/V100S) is not covered by this production package."
      echo "Use this when no validated tuned package exists for the target GPU."
      echo "This package prioritizes compatibility and does not promise per-architecture optimal performance."
      ;;
    sm86-g8)
      echo "Production recommended tuned package for sm_86 / RTX 3080 Ti / RTX 3090 class GPUs."
      echo "Compile-time profile: ARCH=sm_86 GROUPM=8 KSTAGES=3."
      echo "Runtime defaults: TC_PERSIST=0 and pool-mode auto-restart, unless explicitly overridden by the operator."
      ;;
    sm120-*)
      echo "Candidate tuned package for sm_120 / RTX 50 series / Blackwell GPUs."
      echo "Compile-time profile: ARCH=${ARCH:-unknown} GROUPM=${GROUPM} KSTAGES=${KSTAGES}."
      echo "Status: Candidate/Experimental until GPU_PROFILES.md records POSTCHECK, benchmark, and pool accepted data."
      echo "Do not make this an automatic install target before it beats the generic fallback in documented live testing."
      ;;
    sm86-*-k4)
      echo "Experimental / known-not-for-RTX-3080Ti profile."
      echo "KSTAGES=4 has failed smoke on RTX 3080 Ti due to dynamic shared memory limit."
      echo "Do not use as a production package unless GPU_PROFILES.md is updated with new verified data."
      ;;
    *)
      echo "Non-default flavor. Treat as Candidate/Experimental unless GPU_PROFILES.md marks it Production recommended."
      echo "Do not make this an automatic install target without benchmark, POSTCHECK and pool accepted records."
      ;;
  esac
  echo
  echo "Runtime behavior:"
  echo "-----------------"
  echo "Pool mode uses the portable run.sh auto-restart loop by default (KAN_RESTART=auto)."
  echo "Pool mode requires an explicit --wallet ADDR[.WORKER]; public packages do not"
  echo "carry a real default wallet address."
  echo "The miner queues fresh found shares to an async submit worker so network"
  echo "submit_wait does not block the next mining attempt."
  echo "If a newer pool job arrives while a proof is being prepared, stale proof"
  echo "assembly is aborted before wasting CPU on a proof that cannot be submitted."
  echo
  if git rev-parse -q --verify "refs/tags/${VERSION}" >/dev/null 2>&1; then
    echo "Tag notes:"
    echo "----------"
    git for-each-ref "refs/tags/${VERSION}" --format='%(contents)' | sed '/^[[:space:]]*$/N;/^\n$/D'
  else
    last_tag="$(git describe --tags --abbrev=0 2>/dev/null || true)"
    if [ -n "${last_tag}" ]; then
      echo "Changes since ${last_tag}:"
      echo "-----------------------"
      git log --oneline --decorate --no-merges "${last_tag}..HEAD" 2>/dev/null || true
      if [ "${VERSION}" != "${last_tag}" ]; then
        echo
        echo "Latest release baseline (${last_tag}):"
        echo "-----------------------------------"
        git for-each-ref "refs/tags/${last_tag}" --format='%(contents)' | sed '/^[[:space:]]*$/N;/^\n$/D'
      fi
    else
      echo "No git tag found. This is a development portable build."
      git log --oneline --decorate --no-merges -20 2>/dev/null || true
    fi
  fi
  echo
  echo "Open-box use:"
  echo "-------------"
  echo "Unpack ${STABLE_TAR} or ${VERSIONED_TAR}, cd ${PKG}, then run ./run.sh."
} > "${release_notes}"

WALLET="<PRL_ADDRESS>.pm"
cat > "${STAGE}/README.txt" <<README
Pearl(PRL) portable miner — download & run
===========================================
Version: ${VERSION}
Commit:  ${COMMIT}
Built:   ${BUILD_DATE}
Flavor:  ${PACKAGE_FLAVOR:-generic}
Arch:    ${ARCH:-portable-fatbin}
GROUPM:  ${GROUPM:-build-default}
KSTAGES: ${KSTAGES:-build-default}

Built by the release workflow / build worker with CUDA ${CUDA_VERSION}
(${NVCC_VERSION}). Runs on any Linux x86-64 GPU host with glibc >= 2.35 and an
NVIDIA driver. NO CUDA toolkit, NO CUTLASS, NO compiler needed on this machine.

Package selection:
  kan-portable-linux-x64.tar.gz
      Generic compatibility package for NVIDIA RTX 20 series / Turing and newer
      (sm_75+). Volta / sm_70 (V100/V100S) is not covered.
      Use it when no validated tuned package exists for the target GPU.

  kan-portable-linux-x64-sm86-g8.tar.gz
      Production tuned package for sm_86 / RTX 3080 Ti / RTX 3090 class GPUs.
      Compile-time profile: ARCH=sm_86 GROUPM=8 KSTAGES=3.
      Runtime defaults: TC_PERSIST=0 and pool-mode auto-restart unless explicitly
      overridden.

  See GPU_PROFILES.md for the authoritative generic/tuned profile table.

Quick start (pool mining):
  tar xzf ${STABLE_TAR} && cd ${PKG}
  ./run.sh --algo pearl \\
    --pool stratum+tcp://prl.kryptex.network:7048 \\
    --wallet ${WALLET} \\
    --worker pm --batch 1000 --cfg real --tc

Kernel benchmark (no pool):
  ./plainproof_gen --cfg real --mine 15 --tc --breakdown

Files:
  kan                 unified miner (--pool / --solo)
  pearl-miner         compatibility alias of kan for old launch scripts
  plainproof_gen      CLI proof generator + kernel benchmark (Phase A1 build)
  plainproof_gen_rmw  pre-A1 baseline (present only if built WITH_AB=1)
  bench/ablate_a1_prebuilt.sh   on-box A1-vs-RMW A/B, no toolchain
  lib/                bundled libssl/libcrypto/libgomp... (found via rpath)
  run.sh              launcher (driver check + package defaults + pool auto-restart)
  status.sh           readable process/GPU/hashrate/share/perf status
  VERSION             exact package version / git describe string
  BUILD_INFO.txt      build metadata for audit and support
  RELEASE_NOTES.txt   version-specific notes generated from the pushed git tag
  CHANGELOG.md         public changelog for production operators
  GPU_PROFILES.md     authoritative generic/tuned GPU package profile table
  install_kan.sh      installer/updater that selects tuned package or generic fallback

GPU arch note:
  CUDA 12.4 nvcc emits sm_75..sm_90 SASS + compute_90 PTX. Volta / sm_70
  (V100/V100S) is not included in the production fatbin and is not a supported
  production target. On Blackwell (RTX 50 series / sm_120) the driver can only
  JIT the PTX fallback. For best Blackwell performance, use a future CUDA 13
  native sm_120 package after it is validated and listed in GPU_PROFILES.md.

Verify it's truly portable (should list only glibc + libcuda as external):
  ldd ./kan

Ampere tuning note:
  sm86-g8 defaults TC_PERSIST=0 because v1.2.11/v1.2.12 VPS data showed it is
  faster on RTX 3080Ti. Override with TC_PERSIST=1 ./run.sh ... if needed.

Restart note:
  Pool mode auto-restarts by default because pool disconnect/job-error exits the
  miner.  Use KAN_RESTART=0 ./run.sh ... for a one-shot run, or
  KAN_RESTART_DELAY=5 ./run.sh ... to change the reconnect delay.

Pool runtime note:
  Pool mode requires an explicit --wallet <PRL_ADDRESS>[.WORKER].
  Found shares are submitted by an async worker, so the mining loop does not
  idle while waiting for the pool response.  If a newer job arrives while a
  winning proof is being prepared, stale proof assembly is aborted and the miner
  continues on the fresh job.
README

echo "=== [4/5] portability proof (ldd of staged kan) ==="
ldd "${STAGE}/kan" || true
echo "  ^ everything above must be glibc (libc/libm/ld-linux/...) or ./lib/* or"
echo "    libcuda.so.* (the driver, supplied by the GPU host). Anything else = a"
echo "    missed bundle — add it to package_portable.sh."

echo "=== [5/5] tar ==="
( cd "${DIST}" && tar czf "${VERSIONED_TAR}" "${PKG}" )
cp -f "${DIST}/${VERSIONED_TAR}" "${DIST}/${STABLE_TAR}"
ls -la "${DIST}/${VERSIONED_TAR}" "${DIST}/${STABLE_TAR}"
du -sh "${STAGE}" | awk '{print "  unpacked size: "$1}'
echo "PORTABLE PACKAGE OK -> dist/${VERSIONED_TAR}"
echo "STABLE CI ALIAS      -> dist/${STABLE_TAR}"
