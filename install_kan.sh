#!/usr/bin/env bash
# Install/update a prebuilt Kan portable package on a GPU target machine.
#
# Production rule:
#   detect GPU -> choose validated tuned package if available -> fallback generic
#
# This script intentionally does NOT:
#   * git pull
#   * run nvcc
#   * sweep GROUPM/KSTAGES
#   * modify source
#
# Required on target:
#   NVIDIA driver, curl or wget, tar, Linux x86-64 / glibc >= 2.35.
#
# Common usage:
#   VERSION=v1.2.19 ./install_kan.sh
#   VERSION=v1.2.19 DEST=/opt/kan ./install_kan.sh
#   ./install_kan.sh --version v1.2.19 --dest /opt/kan
#   ./install_kan.sh --dry-run --gpu-sm sm_86
#   ./install_kan.sh --force-generic
#   ./install_kan.sh --force-package kan-portable-linux-x64-sm86-g8.tar.gz
set -euo pipefail

REPO_URL="${REPO_URL:-https://cnb.cool/wuyueyi/peral}"
VERSION="${VERSION:-latest}"
DEST="${DEST:-$HOME/kan}"
TMPDIR="${TMPDIR:-/tmp}"
PKG_DIR="kan-portable-linux-x64"
DRY_RUN="${DRY_RUN:-0}"
RUN_STATUS="${RUN_STATUS:-1}"
work=""

usage() {
  cat <<'EOF'
Usage: install_kan.sh [options]

Options:
  --version TAG             Release tag to install (default: VERSION env or latest)
  --dest DIR                Install directory (default: DEST env or ~/kan)
  --base-url URL            Release asset base URL
  --repo-url URL            CNB repository URL
  --force-generic           Install kan-portable-linux-x64.tar.gz
  --force-sm86-g8           Install kan-portable-linux-x64-sm86-g8.tar.gz
  --force-package FILE      Install an explicit package file from the release
  --gpu-sm sm_86            Override GPU compute capability detection
  --dry-run                 Print selection only; do not download/install
  --no-status               Do not run ./status.sh after install
  -h, --help                Show this help

Environment equivalents: VERSION, DEST, RELEASE_BASE_URL, FORCE_PKG,
GPU_SM_OVERRIDE, DRY_RUN, RUN_STATUS.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --version=*) VERSION="${1#*=}"; shift ;;
    --dest) DEST="$2"; shift 2 ;;
    --dest=*) DEST="${1#*=}"; shift ;;
    --base-url) RELEASE_BASE_URL="$2"; shift 2 ;;
    --base-url=*) RELEASE_BASE_URL="${1#*=}"; shift ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --repo-url=*) REPO_URL="${1#*=}"; shift ;;
    --force-generic) FORCE_PKG="kan-portable-linux-x64.tar.gz"; shift ;;
    --force-sm86-g8) FORCE_PKG="kan-portable-linux-x64-sm86-g8.tar.gz"; shift ;;
    --force-package) FORCE_PKG="$2"; shift 2 ;;
    --force-package=*) FORCE_PKG="${1#*=}"; shift ;;
    --gpu-sm) GPU_SM_OVERRIDE="$2"; shift 2 ;;
    --gpu-sm=*) GPU_SM_OVERRIDE="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-status) RUN_STATUS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

while [ "${DEST}" != "/" ] && [ "${DEST%/}" != "${DEST}" ]; do
  DEST="${DEST%/}"
done

# CNB release asset URLs may be overridden by operators or deployment tooling.
# Expected layout:
#   ${RELEASE_BASE_URL}/kan-portable-linux-x64.tar.gz
#   ${RELEASE_BASE_URL}/kan-portable-linux-x64-sm86-g8.tar.gz
if [ -z "${RELEASE_BASE_URL:-}" ]; then
  RELEASE_BASE_URL="${REPO_URL}/-/releases/download/${VERSION}"
fi

have() { command -v "$1" >/dev/null 2>&1; }

cleanup() {
  if [ -n "${work:-}" ] && [ -d "${work}" ]; then
    rm -rf "${work}"
  fi
}
trap cleanup EXIT

download() {
  local url="$1" out="$2"
  if have curl; then
    curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
  elif have wget; then
    wget -O "$out" "$url"
  else
    echo "ERROR: need curl or wget" >&2
    return 127
  fi
}

check_glibc() {
  local ver major minor
  ver="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)"
  [ -n "$ver" ] || { echo "glibc:   unknown"; return 0; }
  major="${ver%%.*}"
  minor="${ver#*.}"; minor="${minor%%.*}"
  echo "glibc:   ${ver}"
  if [ "${major:-0}" -lt 2 ] || { [ "${major:-0}" -eq 2 ] && [ "${minor:-0}" -lt 35 ]; }; then
    echo "WARNING: glibc ${ver} is older than the package baseline 2.35." >&2
  fi
}

detect_sm() {
  if [ -n "${GPU_SM_OVERRIDE:-}" ]; then
    case "${GPU_SM_OVERRIDE}" in
      sm_*) echo "${GPU_SM_OVERRIDE}" ;;
      *) echo "sm_${GPU_SM_OVERRIDE}" ;;
    esac
    return 0
  fi

  if ! have nvidia-smi; then
    echo ""
    return 0
  fi

  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | tr -d '.[:space:]' || true)"

  if [[ "${cap}" =~ ^[0-9]+$ ]]; then
    echo "sm_${cap}"
  else
    echo ""
  fi
}

select_pkg() {
  local sm="$1"

  if [ -n "${FORCE_PKG:-}" ]; then
    echo "${FORCE_PKG}"
    return 0
  fi

  # Keep this table synchronized with GPU_PROFILES.md.
  case "${sm}" in
    sm_86)
      echo "kan-portable-linux-x64-sm86-g8.tar.gz"
      ;;
    sm_75|sm_80|sm_89|sm_90|sm_120)
      echo "kan-portable-linux-x64.tar.gz"
      ;;
    *)
      echo "kan-portable-linux-x64.tar.gz"
      ;;
  esac
}

print_gpu() {
  if have nvidia-smi; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version \
      --format=csv,noheader 2>/dev/null | head -1 || true
  else
    echo "<nvidia-smi not found>"
  fi
}

verify_checksum_if_available() {
  local tarball="$1" pkg="$2" sums="${work}/SHA256SUMS"
  if ! download "${RELEASE_BASE_URL%/}/SHA256SUMS" "${sums}" >/dev/null 2>&1; then
    echo "sha256: SHA256SUMS not available; skipping checksum verification"
    return 0
  fi
  if ! grep -Fq "  ${pkg}" "${sums}"; then
    echo "sha256: ${pkg} not listed in SHA256SUMS; skipping checksum verification"
    return 0
  fi
  if have sha256sum; then
    ( cd "$(dirname "$tarball")" && grep -F "  ${pkg}" "${sums}" | sha256sum -c - )
  else
    echo "sha256: sha256sum not found; skipping checksum verification"
  fi
}

main() {
  local sm pkg generic_pkg url tarball install_parent
  if [ -z "${DEST}" ] || [ "${DEST}" = "/" ]; then
    echo "ERROR: unsafe DEST='${DEST}'" >&2
    exit 2
  fi

  sm="$(detect_sm)"
  pkg="$(select_pkg "${sm}")"
  generic_pkg="kan-portable-linux-x64.tar.gz"
  work="$(mktemp -d "${TMPDIR%/}/kan-install.XXXXXX")"
  tarball="${work}/${pkg}"

  echo "=== Kan portable installer ==="
  echo "gpu:      $(print_gpu)"
  echo "sm:       ${sm:-unknown}"
  check_glibc
  echo "version:  ${VERSION}"
  echo "base url: ${RELEASE_BASE_URL}"
  echo "dest:     ${DEST}"
  echo "package:  ${pkg}"
  case "${sm}" in
    sm_70)
      echo
      echo "WARNING: sm_70 / Volta (V100/V100S) is not supported by current production packages." >&2
      echo "         The portable fatbin starts at sm_75; generic fallback is not expected to run." >&2
      ;;
    sm_75)
      echo
      echo "WARNING: sm_75 / Turing is treated as experimental until a real GPU L3 record exists." >&2
      echo "         If the generic package fails on this GPU, report it as unsupported for now." >&2
      ;;
  esac
  echo

  if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1: selection complete; not downloading or installing."
    exit 0
  fi

  url="${RELEASE_BASE_URL%/}/${pkg}"
  if ! download "${url}" "${tarball}"; then
    if [ "${pkg}" != "${generic_pkg}" ]; then
      echo
      echo "WARNING: tuned package download failed; falling back to generic." >&2
      pkg="${generic_pkg}"
      tarball="${work}/${pkg}"
      url="${RELEASE_BASE_URL%/}/${pkg}"
      download "${url}" "${tarball}"
    else
      echo "ERROR: generic package download failed: ${url}" >&2
      exit 1
    fi
  fi

  verify_checksum_if_available "${tarball}" "${pkg}"

  tar -xzf "${tarball}" -C "${work}"
  if [ ! -d "${work}/${PKG_DIR}" ]; then
    echo "ERROR: package did not contain ${PKG_DIR}/" >&2
    exit 1
  fi

  install_parent="$(dirname "${DEST}")"
  mkdir -p "${install_parent}"
  rm -rf "${DEST}.new"
  mv "${work}/${PKG_DIR}" "${DEST}.new"

  if [ -d "${DEST}" ]; then
    rm -rf "${DEST}.prev"
    mv "${DEST}" "${DEST}.prev"
  fi
  mv "${DEST}.new" "${DEST}"

  echo
  echo "=== Installed BUILD_INFO.txt ==="
  cat "${DEST}/BUILD_INFO.txt" 2>/dev/null || true

  echo
  echo "Install OK."
  echo "Run:"
  echo "  cd ${DEST}"
  echo "  ./run.sh --algo pearl --pool stratum+tcp://prl.kryptex.network:7048 --wallet <PRL_ADDRESS.WORKER> --batch 1000 --cfg real --tc"
  echo
  echo "Status:"
  echo "  cd ${DEST} && ./status.sh"
  if [ "${RUN_STATUS}" = "1" ] && [ -x "${DEST}/status.sh" ]; then
    echo
    ( cd "${DEST}" && ./status.sh ) || true
  fi
}

main "$@"
