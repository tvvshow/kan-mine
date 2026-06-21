#!/usr/bin/env bash
# start_pool.sh — one-command Pearl(PRL) pool miner launcher
#
# Usage:
#   bash start_pool.sh                 # build + start (defaults below)
#   bash start_pool.sh stop            # stop the running miner
#   bash start_pool.sh log             # tail the live log
#   bash start_pool.sh rebuild         # git pull + rebuild + restart
#
# What it does:
#   1. checks for GPU + CUDA toolkit
#   2. installs build deps (gcc g++ libssl-dev) if missing
#   3. builds Kan (auto-detects GPU arch via nvidia-smi)
#   4. runs Kan --pool in an auto-restart loop
#
# Config (env overrides):
#   WALLET   PRL payout address  (default: built-in)
#   WORKER   worker name          (default: hostname)
#   BRANCH   git branch           (default: unified)

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# ---- defaults ----
WALLET="${WALLET:-prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv}"
WORKER="${WORKER:-$(hostname 2>/dev/null || echo box)}"
BRANCH="${BRANCH:-unified}"
LOG="$DIR/miner.log"
PIDF="$DIR/miner.pid"
BUILD_LOG="${BUILD_LOG:-$DIR/build.log}"
ALLOW_FALLBACK="${ALLOW_FALLBACK:-0}"

# ---- commands: stop / log ----
if [ "${1:-}" = "stop" ]; then
  if [ -f "$PIDF" ]; then
    PID="$(cat "$PIDF")"
    # Kill the wrapper (which kills the miner); wait up to 5s
    kill "$PID" 2>/dev/null && echo "stopped PID=$PID" || echo "PID=$PID not running"
    rm -f "$PIDF"
  else
    echo "no PID file — miner not running?"
  fi
  exit 0
fi

if [ "${1:-}" = "log" ]; then
  exec tail -f "$LOG"
fi

# ---- GPU check ----
echo "=== Pearl(PRL) Pool Miner ==="
echo ""
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found — no NVIDIA GPU driver installed?"
  echo "       This miner requires an NVIDIA GPU with CUDA."
  exit 1
fi
nvidia-smi -L
echo ""

if ! command -v nvcc &>/dev/null; then
  echo "ERROR: nvcc not found — no CUDA toolkit installed?"
  echo "       Install CUDA toolkit (apt install nvidia-cuda-toolkit or full toolkit from NVIDIA)."
  exit 1
fi

# ---- deps ----
echo "=== build deps ==="
if ! dpkg -s gcc g++ libssl-dev &>/dev/null 2>&1; then
  apt-get update -qq && \
    apt-get install -y --no-install-recommends gcc g++ make libssl-dev ca-certificates
else
  echo "gcc/g++/libssl-dev already installed"
fi
echo ""

# ---- git pull (if in a git repo) ----
if [ -d ".git" ]; then
  echo "=== updating code ==="
  git fetch origin "$BRANCH" 2>/dev/null && \
    git reset --hard "origin/$BRANCH" --quiet 2>/dev/null || true
  echo "HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo ""
fi

# ---- build ----
echo "=== building Kan miner ==="
if ! bash build.sh >"$BUILD_LOG" 2>&1; then
  echo "FATAL: build failed. Last 80 lines from $BUILD_LOG:"
  tail -80 "$BUILD_LOG"
  exit 1
fi
tail -20 "$BUILD_LOG"
echo ""

if grep -q "falling back to tc_block" "$BUILD_LOG" && [ "$ALLOW_FALLBACK" != "1" ]; then
  echo "FATAL: CUTLASS kernel was not built; build fell back to tc_block (~30 TH/s)."
  echo "       Install/point CUTLASS_HOME, or set ALLOW_FALLBACK=1 for an explicit fallback run."
  echo "       Full build log: $BUILD_LOG"
  exit 1
fi

MINER_BIN="$DIR/build/kan"
if [ ! -x "$MINER_BIN" ]; then
  # Compatibility fallback for older build trees.
  MINER_BIN="$DIR/build/pearl-miner"
fi
if [ ! -x "$MINER_BIN" ]; then
  echo "FATAL: neither build/kan nor build/pearl-miner is executable -- build failed."
  echo "Full build log: $BUILD_LOG"
  exit 1
fi

# ---- rebuild: stop old miner first ----
if [ "${1:-}" = "rebuild" ]; then
  if [ -f "$PIDF" ]; then
    OLD="$(cat "$PIDF")"
    kill "$OLD" 2>/dev/null && echo "stopped old miner PID=$OLD" || true
    sleep 2
  fi
fi

# ---- stop any previous miner ----
if [ -f "$PIDF" ]; then
  OLD="$(cat "$PIDF")"
  if kill -0 "$OLD" 2>/dev/null; then
    echo "miner already running (PID=$OLD). Use: bash $0 stop"
    exit 0
  fi
  rm -f "$PIDF"
fi

# ---- launch ----
echo "=== starting pool miner ==="
echo "  wallet : ${WALLET:0:20}...${WALLET: -8}"
echo "  worker : $WORKER"
echo "  pool   : prl.kryptex.network:7048"
echo "  binary : $MINER_BIN"
echo "  config : real (m=n=131072, k=4096, rank=256), CUTLASS primary"
echo "  log    : $LOG"
echo "  pid    : $PIDF"
echo ""
echo "commands:"
echo "  bash $0 stop      — stop miner"
echo "  bash $0 log       — live log"
echo "  bash $0 rebuild   — git pull + rebuild + restart"
echo ""

# Write the restart wrapper as a separate script so the nohup'd process
# is easy to manage (one PID, kills cleanly).
cat > "$DIR/_run_loop.sh" <<LOOP
#!/usr/bin/env bash
# auto-restart wrapper for Kan --pool
# The miner exits on disconnect/job-error; this loop reconnects.
WALLET="$WALLET"
WORKER="$WORKER"
LOG="$LOG"
MINER_BIN="$MINER_BIN"

while true; do
  echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Kan starting: \$MINER_BIN" >> "\$LOG"
  "\$MINER_BIN" --pool \\
    --wallet "\$WALLET" \\
    --worker "\$WORKER" \\
    >> "\$LOG" 2>&1
  RC=\$?
  echo "[\$(date '+%Y-%m-%d %H:%M:%S')] miner exited (rc=\$RC), restarting in 15s..." >> "\$LOG"
  sleep 15
done
LOOP
chmod +x "$DIR/_run_loop.sh"

nohup bash "$DIR/_run_loop.sh" >> "$LOG" 2>&1 &
echo $! > "$PIDF"

echo "launched PID=$(cat "$PIDF")"
echo ""
echo "=== miner running ==="
