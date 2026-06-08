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
echo "=== build official verify_plain + verify_share (zk-pow) ==="
( cd "${ROOT}/zk-pow" && RUSTFLAGS="-C target-cpu=native" cargo build --release --bin verify_plain --bin verify_share )
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

# --- REAL kryptex network config (m=n=131072,k=4096,r=256) -------------------
# THE live config captured from lpminer --pearl-share-dump (oracle/). Its 52-byte
# mining_config is reproduced bit-for-bit by our Config::to_bytes() with these
# patterns. Fed the oracle's own easy header (nbits=207fffff) so the CPU search
# wins tile 0 immediately and exercises the real-config jackpot fold. A VALID here
# proves our REAL-config proof is officially correct end-to-end = the oracle the
# fast tensor-core kernel (M2a Phase B) will be validated against.
REAL_HDR="01000000f9661239d86cd892e31455d6ad6c1a55745ab7d16a63c82143d271f417ca49994f2738ce9c121c22c08598078e168bf4e1b8167b4e6f30fe911d555492a1afacf2e3246affff7f20"
verify_one REAL --cfg real --header "$REAL_HDR"

# --- M2a fused tensor-core kernel (WMMA int8) --------------------------------
# The official verifier ALWAYS derives its bound from the header nbits, so a real
# correctness test needs a MODERATE nbits (not the easy 207fffff whose bound
# saturates to U256::MAX and accepts any jackpot). Golden uses nbits 1D2FFFFF
# (bound ~2^247, ~1/512 tiles win). REAL_HDR_MOD = the oracle header with nbits
# swapped to 1D2FFFFF so the real-config search wins at a bound the verifier
# actually checks => only a CORRECT jackpot passes.
#   TCGOLD : golden config, tensor-core path  -> validates the WMMA jackpot math.
#   TCREAL : REAL config (h=8,w=16,rank=256)   -> validates real-config TC path,
#            and the solver prints "tc(fused): ... TMAC/s" = the speed benchmark.
REAL_HDR_MOD="01000000f9661239d86cd892e31455d6ad6c1a55745ab7d16a63c82143d271f417ca49994f2738ce9c121c22c08598078e168bf4e1b8167b4e6f30fe911d555492a1afacf2e3246affff2f1d"
verify_one TCGOLD --tc
verify_one TCREAL --cfg real --tc --header "$REAL_HDR_MOD"

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
echo "RESULT: ALL-VALID  (golden CPU + golden GPU --mine + REAL-config proofs all pass the official verifier)"
