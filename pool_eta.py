#!/usr/bin/env python3
# pool_eta.py — connect to the kryptex Pearl LuckyPool, read ONE live job, and
# compute the exact expected draws-per-share + ETA for OUR solver, so we decide
# whether to commit to a hunt BEFORE spending GPU hours. Read-only: authorize +
# read one mining.notify, never submits. Plaintext TCP 7048 (raw socket => no
# HTTP-proxy interference, china-direct).
#
# WIN MODEL (matches plainproof_gen.cpp --target): a tile wins iff
#   U256(blake3 transcript) <= bound,  bound = target_int * (h*w*dot_len).
# target_int = the pool's announced 64-hex target as a big-endian integer.
# So  P(one tile wins) = bound / 2^256,  and over NTILES tiles per draw:
#   draws_per_share = 1 / (NTILES * P_tile) = 2^256 / (target_int * FACTOR * NTILES).
import socket, json, time, os, sys

HOST  = os.environ.get("POOL_HOST", "prl.kryptex.network")
PORT  = int(os.environ.get("POOL_PORT", "7048"))
WALLET= os.environ.get("WALLET", "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv")
WORKER= os.environ.get("WORKER", "etaprobe")
AGENT = os.environ.get("AGENT", "lpminer/0.1.9-552bdfe")

# REAL kryptex network config (captured): m=n=131072, k=4096, rank=256, h=8, w=16.
FACTOR = int(os.environ.get("FACTOR", str(8 * 16 * 4096)))      # h*w*dot_len = 524288 = 2^19
NTILES = int(os.environ.get("NTILES", str(134217728)))         # nrow_off*ncol_off = 2^27 (from TCREAL run)
# per-draw wall-clock on the search GPU (TCREAL on L40 = 41s kernel; +~31s CPU RNG fill).
DRAW_S = [("kernel-only 41s", 41.0), ("kernel+RNG ~72s", 72.0), ("lpminer-speed 0.2s", 0.2)]

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    log(f"[eta] connecting {HOST}:{PORT} (plaintext) ...")
    try:
        s = socket.create_connection((HOST, PORT), timeout=20)
    except Exception as e:
        log(f"[eta] FATAL connect failed: {e}"); sys.exit(2)
    s.settimeout(30)
    auth = {"id": 1, "method": "mining.authorize",
            "params": {"wallet": f"{WALLET}.{WORKER}", "worker": WORKER, "agent": AGENT}}
    s.sendall((json.dumps(auth) + "\n").encode())
    log(f"[eta] >> authorize wallet={WALLET}.{WORKER}")

    buf = bytearray(); job = None; t0 = time.time()
    while time.time() - t0 < 28:
        try:
            d = s.recv(8192)
        except socket.timeout:
            break
        if not d:
            log("[eta] pool closed connection"); break
        buf.extend(d)
        while b"\n" in buf:
            i = buf.index(b"\n"); line = bytes(buf[:i]).decode(errors="replace").strip(); del buf[:i+1]
            if not line: continue
            try: m = json.loads(line)
            except Exception: log("[eta] <<RAW", line[:200]); continue
            if m.get("method") == "mining.notify":
                job = m.get("params", {}); log("[eta] << notify", json.dumps(job)[:400])
            else:
                log("[eta] <<", line[:300])
        if job: break
    try: s.close()
    except Exception: pass

    if not job:
        log("[eta] NO JOB received — cannot compute ETA"); sys.exit(3)

    tgt_hex = job.get("target", ""); job_id = job.get("job_id", ""); height = job.get("height")
    if not tgt_hex:
        log("[eta] notify had no target field — cannot compute ETA"); sys.exit(4)
    target_int = int(tgt_hex, 16)
    lead_zero_bits = 256 - target_int.bit_length() if target_int else 256
    d_share = None
    if "_" in job_id:
        try: d_share = int(job_id.split("_")[-1])
        except Exception: pass

    bound = target_int * FACTOR
    # P_tile = bound / 2^256 ; draws_per_share = 1/(NTILES*P_tile)
    denom = bound * NTILES
    draws = (1 << 256) / denom if denom else float("inf")
    wins_per_draw = (NTILES * bound) / (1 << 256)

    print("\n================= POOL ETA =================")
    print(f"height          : {height}")
    print(f"job_id          : {job_id}   (share-diff suffix D_share = {d_share})")
    print(f"target (hex)    : {tgt_hex}")
    print(f"target_int      : ~2^{target_int.bit_length()-1}  ({lead_zero_bits} leading zero bits)")
    print(f"FACTOR h*w*dot  : {FACTOR} (=2^{FACTOR.bit_length()-1})   NTILES: {NTILES} (=2^{NTILES.bit_length()-1})")
    print(f"bound=tgt*FACTOR: ~2^{bound.bit_length()-1}")
    print(f"P(tile wins)    : ~2^-{256-bound.bit_length()}   wins/draw: {wins_per_draw:.4g}")
    print(f"draws / share   : {draws:.4g}")
    for name, sec in DRAW_S:
        eta = draws * sec
        print(f"  ETA @ {name:20s}: {eta:10.1f} s  = {eta/60:7.2f} min = {eta/3600:6.2f} h")
    if d_share:
        # cross-check: if acceptance is purely 'achieved_diff >= D_share' (no FACTOR),
        # draws would be D_share/NTILES. Report it so we see which regime we're in.
        alt = d_share / NTILES
        print(f"[xcheck] if accept= U256<=2^256/D_share (no FACTOR): draws/share = {alt:.4g}")
    print("============================================")

if __name__ == "__main__":
    main()
