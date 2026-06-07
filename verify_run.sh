#!/usr/bin/env bash
# Offline self-verification of OUR PlainProofs against the OFFICIAL Rust verifier.
#
# Builds the official `verify_plain` bin — zk_pow::api::verify::verify_plain_proof,
# the exact check a Pearl gateway/node runs (minus the plonky2 proving wrap) — from
# the vendored pearl-spec sources, builds OUR CUDA solver, then has the solver emit a
# PlainProof via TWO independent paths and checks BOTH against the official verifier:
#   CPU : plainproof_gen 12345     reference tile search (already known-good on the box)
#   GPU : plainproof_gen --mine N  the dp4a redraw loop the LIVE kryptex hunt actually
#                                  used — its winning-share assembly has NEVER yet been
#                                  checked against the official verifier.
# A GPU-path VALID is the missing link: it proves our winning-share proof is officially
# correct, so any kryptex rejection is purely their third-party submit contract — not us.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

# ---------- Rust toolchain via USTC mirror (China-direct, fast) ----------
export CARGO_HOME="${ROOT}/.cargo"
export RUSTUP_HOME="${ROOT}/.rustup"
export RUSTUP_DIST_SERVER="https://mirrors.ustc.edu.cn/rust-static"
export RUSTUP_UPDATE_ROOT="https://mirrors.ustc.edu.cn/rust-static/rustup"
mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"
export PATH="${CARGO_HOME}/bin:${PATH}"

if ! command -v rustc >/dev/null 2>&1; then
  echo "=== install Rust (USTC rust-static) ==="
  curl -sSfL "${RUSTUP_UPDATE_ROOT}/dist/x86_64-unknown-linux-gnu/rustup-init" -o /tmp/rustup-init
  chmod +x /tmp/rustup-init
  /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain stable
fi
# crates.io -> USTC sparse mirror (edition 2024 deps resolve fast & China-direct)
cat > "${CARGO_HOME}/config.toml" <<'EOF'
[source.crates-io]
replace-with = "ustc"
[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
[net]
git-fetch-with-cli = true
EOF
rustc --version || { echo "FATAL: Rust install failed"; exit 1; }
cargo --version

# ---------- build the OFFICIAL verifier ----------
echo "=== build official verify_plain (zk-pow) ==="
( cd "${ROOT}/zk-pow" && RUSTFLAGS="-C target-cpu=native" cargo build --release --bin verify_plain )
VP="${ROOT}/zk-pow/target/release/verify_plain"
[ -x "${VP}" ] || { echo "FATAL: verify_plain build failed"; exit 1; }
echo "official verifier: ${VP}"

# ---------- build OUR solver ----------
echo "=== build our CUDA solver (build.sh) ==="
bash "${ROOT}/build.sh"
GEN="${ROOT}/build/plainproof_gen"
[ -x "${GEN}" ] || { echo "FATAL: solver build failed"; exit 1; }

# ---------- generate a proof one way, verify it the official way ----------
: > /tmp/verdicts.txt
verify_one() {            # $1 = label ; rest = solver args
  local label="$1"; shift
  echo ""
  echo "############################################################"
  echo "### [$label] generate: plainproof_gen $*"
  echo "############################################################"
  "${GEN}" "$@" > "/tmp/${label}.b64" 2> "/tmp/${label}.log"; local rc=$?
  local hdr; hdr=$(grep -oE 'HEADER_HEX=[0-9a-f]+' "/tmp/${label}.log" | head -1 | cut -d= -f2)
  local n;   n=$(wc -c < "/tmp/${label}.b64")
  echo "[$label] solver rc=${rc}  b64_bytes=${n}  header_hex_len=${#hdr}"
  echo "--- [$label] solver log (tail) ---"; tail -8 "/tmp/${label}.log"
  if [ "${rc}" -ne 0 ] || [ -z "${hdr}" ] || [ "${n}" -le 32 ]; then
    echo "${label}=NOPROOF" | tee -a /tmp/verdicts.txt; return
  fi
  echo "--- [$label] OFFICIAL verify_plain ---"
  if "${VP}" "${hdr}" < "/tmp/${label}.b64"; then
    echo "${label}=VALID"   | tee -a /tmp/verdicts.txt
  else
    echo "${label}=INVALID" | tee -a /tmp/verdicts.txt
  fi
}

verify_one CPU 12345
verify_one GPU --mine 5000

echo ""
echo "============================================================"
echo "=== SELF-VERIFY SUMMARY (official zk_pow::verify_plain_proof) ==="
cat /tmp/verdicts.txt
echo "============================================================"
# Overall gate: BOTH paths must be VALID for a clean pass.
if grep -q '=INVALID' /tmp/verdicts.txt || grep -q '=NOPROOF' /tmp/verdicts.txt; then
  echo "RESULT: NOT-ALL-VALID  (a path failed official verification — real, free-fixable bug)"
  exit 1
fi
echo "RESULT: ALL-VALID  (both CPU and GPU --mine proofs pass the official verifier)"
