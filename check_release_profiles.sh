#!/usr/bin/env bash
# Static release-profile checks for the production portable package plan.
#
# This is intentionally GPU-free. It verifies that the release matrix and helper
# scripts remain aligned with GPU_PROFILES.md:
#   * default Release assets are generic + sm86-g8 only;
#   * unsupported / historical sm86 sweep packages are not uploaded by default;
#   * shell scripts parse;
#   * install_kan.sh selects sm86-g8 for sm_86 and generic for unvalidated archs.
set -euo pipefail
cd "$(dirname "$0")"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

check_file() {
  [ -f "$1" ] || fail "missing $1"
}

expect_contains() {
  local file="$1" pattern="$2"
  grep -Fq -- "$pattern" "$file" || fail "$file missing expected text: $pattern"
}

expect_not_contains() {
  local file="$1" pattern="$2"
  if grep -Fq -- "$pattern" "$file"; then
    fail "$file contains forbidden text: $pattern"
  fi
}

assert_release_attachments_exact() {
  local expected actual
  expected="$(cat <<'EOF'
dist/SHA256SUMS
dist/kan-portable-linux-x64-sm75.tar.gz
dist/kan-portable-linux-x64-sm86-g8.tar.gz
dist/kan-portable-linux-x64.tar.gz
install_kan.sh
EOF
)"
  # Sort both sides under LC_ALL=C so the comparison is locale-independent.
  # (zh_CN.UTF-8 and other UTF-8 locales sort "...-sm86-g8.tar.gz" AFTER
  # "....tar.gz" because they ignore punctuation; C locale orders by byte, where
  # '-' < '.', putting sm86-g8 first. The expected list below is C-sorted.)
  expected="$(printf '%s\n' "${expected}" | LC_ALL=C sort)"
  actual="$(
    awk '
      /^[[:space:]]*attachments:[[:space:]]*$/ {in_list=1; next}
      in_list && /^[[:space:]]*-[[:space:]]*/ {
        sub(/^[[:space:]]*-[[:space:]]*/, "", $0)
        print
        next
      }
      in_list && /^[[:space:]]*[A-Za-z0-9_-]+:/ {in_list=0}
    ' .cnb.yml | LC_ALL=C sort
  )"

  if [ "${actual}" != "${expected}" ]; then
    echo "ERROR: release attachment list changed unexpectedly" >&2
    echo "Expected:" >&2
    printf '%s\n' "${expected}" >&2
    echo "Actual:" >&2
    printf '%s\n' "${actual}" >&2
    exit 1
  fi
}

dry_pkg() {
  local sm="$1"
  DRY_RUN=1 GPU_SM_OVERRIDE="$sm" VERSION=vCHECK bash ./install_kan.sh \
    2>/dev/null | awk -F':  +' '/^package:/ {print $2; exit}' || true
}

echo "=== release profile static checks ==="

check_file ".cnb.yml"
check_file "package_portable.sh"
check_file "install_kan.sh"
check_file "GPU_PROFILES.md"
check_file "README.md"
check_file "CHANGELOG.md"
check_file "ci_gpu_verify.sh"
check_file ".cnb/web_trigger.yml"
check_file "run_test.sh"
check_file "install_service.sh"
check_file "kan.service"
check_file "kan.logrotate"

echo "[1/6] bash syntax"
bash -n package_portable.sh
bash -n install_kan.sh
bash -n check_release_profiles.sh
bash -n ci_gpu_verify.sh
bash -n run_test.sh
bash -n install_service.sh

echo "[2/6] release matrix assets"
expect_contains ".cnb.yml" "dist/kan-portable-linux-x64.tar.gz"
expect_contains ".cnb.yml" "dist/kan-portable-linux-x64-sm75.tar.gz"
expect_contains ".cnb.yml" "dist/kan-portable-linux-x64-sm86-g8.tar.gz"
expect_contains ".cnb.yml" "package-sm75-turing"
expect_contains ".cnb.yml" "ARCH=sm_75 KERNEL=wmma PACKAGE_FLAVOR=sm75"
expect_contains ".cnb.yml" "dist/SHA256SUMS"
expect_contains ".cnb.yml" "cd dist && sha256sum kan-portable-linux-x64.tar.gz"
expect_contains ".cnb.yml" "install_kan.sh > SHA256SUMS"
expect_contains ".cnb.yml" "install_kan.sh"
assert_release_attachments_exact

for forbidden in \
  "dist/kan-portable-linux-x64-sm86-g4-k3.tar.gz" \
  "dist/kan-portable-linux-x64-sm86-g12-k3.tar.gz" \
  "dist/kan-portable-linux-x64-sm86-g16-k3.tar.gz" \
  "dist/kan-portable-linux-x64-sm86-g24-k3.tar.gz" \
  "dist/kan-portable-linux-x64-sm86-g8-k4.tar.gz" \
  "dist/kan-portable-linux-x64-sm86-g4-k4.tar.gz" \
  "dist/kan-portable-linux-x64-sm86-g12-k4.tar.gz"; do
  expect_not_contains ".cnb.yml" "$forbidden"
done

echo "[3/6] profile docs"
expect_contains "GPU_PROFILES.md" "kan-portable-linux-x64-sm86-g8.tar.gz"
expect_contains "GPU_PROFILES.md" "GROUPM:    8"
expect_contains "GPU_PROFILES.md" "KSTAGES:   3"
expect_contains "GPU_PROFILES.md" "runtime:   TC_PERSIST=0"
expect_contains "README.md" "generic compatibility package"
expect_contains "README.md" "tuned production package"
expect_contains "README.md" "async share submit"
expect_contains "README.md" "V100/V100S"
expect_contains "README.md" "sm_70"
expect_contains "GPU_PROFILES.md" "async share submit"
expect_contains "GPU_PROFILES.md" "stale proof early-abort"
expect_contains "README.md" "sm_75"
expect_contains "README.md" "不支持"
expect_contains "GPU_PROFILES.md" "sm_75"
# sm_70 / Volta stays unsupported; sm_75 / Turing is now a supported WMMA flavor.
expect_contains "GPU_PROFILES.md" "不支持当前 release"
expect_contains "GPU_PROFILES.md" "kan-portable-linux-x64-sm75.tar.gz"
expect_contains "install_kan.sh" "kan-portable-linux-x64-sm75.tar.gz"
expect_contains "package_portable.sh" "kan-portable-linux-x64-sm75.tar.gz"
expect_contains "build.sh" "KERNEL"
expect_contains "GPU_PROFILES.md" "shared-memory"
expect_contains "GPU_PROFILES.md" 'Volta / `sm_70`'
expect_contains ".cnb.yml" "web_trigger_gpu_verify"
expect_contains ".cnb.yml" "gpu-verify:"
expect_contains ".cnb.yml" "cnb:arch:amd64:gpu"
expect_contains ".cnb/web_trigger.yml" "GPU 验证"
expect_not_contains "src/miner_main.cpp" "prl1patz"
expect_not_contains "package_portable.sh" "prl1patz"

echo "[4/6] install_kan selection matrix"
[ "$(dry_pkg sm_75)" = "kan-portable-linux-x64-sm75.tar.gz" ] || fail "sm_75 should select the sm75 WMMA package"
[ "$(dry_pkg sm_70)" = "unsupported" ] || fail "sm_70 should be unsupported"
[ "$(dry_pkg sm_80)" = "kan-portable-linux-x64.tar.gz" ] || fail "sm_80 should fallback generic"
[ "$(dry_pkg sm_86)" = "kan-portable-linux-x64-sm86-g8.tar.gz" ] || fail "sm_86 should select sm86-g8"
[ "$(dry_pkg sm_89)" = "kan-portable-linux-x64.tar.gz" ] || fail "sm_89 should fallback generic"
[ "$(dry_pkg sm_90)" = "kan-portable-linux-x64.tar.gz" ] || fail "sm_90 should fallback generic"
[ "$(dry_pkg sm_120)" = "kan-portable-linux-x64.tar.gz" ] || fail "sm_120 should fallback generic until CUDA 13 native package"
[ "$(dry_pkg unknown)" = "kan-portable-linux-x64.tar.gz" ] || fail "unknown should fallback generic"

echo "[5/6] package metadata generation hooks"
expect_contains "package_portable.sh" "GPU_PROFILES.md"
expect_contains "package_portable.sh" "install_kan.sh"
expect_contains "package_portable.sh" "package_policy:"
expect_contains "package_portable.sh" "TC_PERSIST="
expect_contains "package_portable.sh" "compute_cap"
expect_contains "package_portable.sh" "cuda_version:"
expect_contains "package_portable.sh" "nvcc_version:"
expect_contains "package_portable.sh" "KAN_RESTART="
expect_contains "package_portable.sh" "KAN_RESTART_DELAY"
expect_contains "package_portable.sh" "pool-mode auto-restart"
expect_contains "package_portable.sh" "live_current.log"
expect_contains "package_portable.sh" "async share submit"
expect_contains "package_portable.sh" "MINE proof abort"
expect_contains "package_portable.sh" "CHANGELOG.md"
expect_contains "package_portable.sh" "requires an explicit --wallet"
expect_contains "package_portable.sh" "Volta / sm_70"
expect_contains "package_portable.sh" "SHA256SUMS"
expect_contains "package_portable.sh" "install_service.sh"
expect_contains "package_portable.sh" "kan.service"
expect_contains "package_portable.sh" "kan.logrotate"
expect_contains "install_kan.sh" "--force-generic"
expect_contains "install_kan.sh" "--dry-run"
expect_contains "install_kan.sh" "SHA256SUMS"
expect_contains "install_kan.sh" "glibc"
expect_contains "install_service.sh" "systemctl daemon-reload"
expect_contains "kan.service" "EnvironmentFile=__KAN_ENV__"
expect_contains "kan.service" "ExecStartPre="
expect_contains "kan.service" "Restart=on-failure"
expect_contains "kan.logrotate" "copytruncate"
expect_contains "install_kan.sh" "sm_70 / Volta"
expect_contains "ci_gpu_verify.sh" "POSTCHECK"
expect_contains "ci_gpu_verify.sh" "GPU_VERIFY_POOL_SECONDS"
expect_contains "run_test.sh" "--cfg real"
expect_contains "run_test.sh" "POSTCHECK"
expect_contains "src/miner_main.cpp" "async share submit worker active"
expect_contains "src/miner_main.cpp" "pool mode requires --wallet"
expect_contains "src/plainproof_gen.cpp" "MINE proof abort"

echo "[6/6] single-machine multi-GPU release hooks"
# Multi-GPU is a production runtime capability: the release docs, the portable
# launcher banner, and the miner itself must keep these hooks aligned so a
# release cannot silently drop the documented single/multi-GPU behavior.
expect_contains "src/miner_main.cpp" "--devices"
expect_contains "src/miner_main.cpp" "auto fanout disabled"
expect_contains "src/miner_main.cpp" "per-GPU connection"
expect_contains "src/miner_main.cpp" "mutually exclusive"
expect_contains "package_portable.sh" "auto multi-GPU fanout ENABLED"
expect_contains "package_portable.sh" "CUDA_VISIBLE_DEVICES"
expect_contains "package_portable.sh" "Multi-GPU note"
expect_contains "README.md" "单机多卡"
expect_contains "README.md" "CUDA_VISIBLE_DEVICES"
expect_contains "README.md" "per-GPU isolated lane process"
expect_contains "GPU_PROFILES.md" "multi-GPU auto fanout"
expect_contains "CHANGELOG.md" "multi-GPU auto fanout"

echo "OK: release profile checks passed"
