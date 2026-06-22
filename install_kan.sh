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
#   VERSION=v1.2.15 ./install_kan.sh
#   VERSION=v1.2.15 DEST=/opt/kan ./install_kan.sh
#   RELEASE_BASE_URL=https://example/releases/v1.2.15 ./install_kan.sh
#   FORCE_PKG=kan-portable-linux-x64.tar.gz ./install_kan.sh
#   DRY_RUN=1 GPU_SM_OVERRIDE=sm_86 ./install_kan.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://cnb.cool/wuyueyi/peral}"
VERSION="${VERSION:-latest}"
DEST="${DEST:-$HOME/kan}"
while [ "${DEST}" != "/" ] && [ "${DEST%/}" != "${DEST}" ]; do
  DEST="${DEST%/}"
done
TMPDIR="${TMPDIR:-/tmp}"
PKG_DIR="kan-portable-linux-x64"
work=""

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
  echo "version:  ${VERSION}"
  echo "base url: ${RELEASE_BASE_URL}"
  echo "dest:     ${DEST}"
  echo "package:  ${pkg}"
  echo

  if [ "${DRY_RUN:-0}" = "1" ]; then
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
  echo "  ./run.sh --algo pearl --pool stratum+tcp://prl.kryptex.network:7048 --wallet <PRL_ADDRESS.WORKER> --batch 500 --cfg real --tc"
  echo
  echo "Status:"
  echo "  cd ${DEST} && ./status.sh"
}

main "$@"
