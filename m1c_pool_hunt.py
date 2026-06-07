#!/usr/bin/env python3
# M1c — kryptex pool-acceptance de-risk for the self-built trusted solver.
#
# Goal: land ONE share-target-meeting proof from OUR plainproof_gen at golden
# dims and capture the pool's ground-truth response to mining.submit. The prior
# project established the pool is SILENT on losers; only a real (share-target)
# win reveals whether kryptex accepts our chosen-dimension proofs. We mine the
# LATEST header (rotates ~70s), abort+restart on job change (draws independent),
# submit a fresh win immediately, and log EVERY non-notify line verbatim.
#
# Env: WORKER (default m1c), GEN, GENDIR, BATCH, MAXSEC (0=forever),
#      STOP_ON_ACK (1=stop after first non-silent submit response).
import socket, ssl, json, sys, time, threading, subprocess, os, select

HOST   = "prl.kryptex.network"; PORT = 8048
WALLET = "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv"
WORKER = os.environ.get("WORKER", "m1c")
GEN    = os.environ.get("GEN", "/root/m1/plainproof_gen")
GENDIR = os.environ.get("GENDIR", "/root/m1")
BATCH  = int(os.environ.get("BATCH", "1000000"))
MAXSEC = int(os.environ.get("MAXSEC", "0"))
STOP_ON_ACK = int(os.environ.get("STOP_ON_ACK", "1"))
STOP_AFTER_WIN = int(os.environ.get("STOP_AFTER_WIN", "1"))  # stop after first win+verdict-wait
WIN_FILE = os.environ.get("WIN_FILE", "/root/m1/m1c_win.json")

st = {"header": None, "target": None, "job_id": None, "height": None, "gen": 0,
      "stop": False, "submit_ids": set(), "got_resp": False}
lock = threading.Lock()
io_lock = threading.Lock()
stats = {"jobs": 0, "submits": 0, "resps": 0, "wins": 0}

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def send(s, o):
    data = (json.dumps(o) + "\n").encode()
    with io_lock:
        s.sendall(data)

def reader(s):
    buf = bytearray()
    while not st["stop"]:
        r, _, _ = select.select([s], [], [], 0.5)
        if not r:
            continue
        with io_lock:
            try:
                d = s.recv(8192)
            except ssl.SSLWantReadError:
                continue
            except Exception as e:
                log("[pool] recv error:", e); st["stop"] = True; break
        if not d:
            log("[pool] EOF (closed by pool)"); st["stop"] = True; break
        buf.extend(d)
        while b"\n" in buf:
            i = buf.index(b"\n"); txt = bytes(buf[:i]).decode(errors="replace").strip(); del buf[:i+1]
            if not txt:
                continue
            try:
                m = json.loads(txt)
            except Exception:
                log("[pool] <<RAW", txt[:300]); continue
            if m.get("method") == "mining.notify":
                p = m.get("params", {})
                with lock:
                    changed = (p.get("header") != st["header"])
                    st["header"] = p.get("header"); st["target"] = p.get("target")
                    st["job_id"] = p.get("job_id"); st["height"] = p.get("height")
                    if changed:
                        st["gen"] += 1; stats["jobs"] += 1
                    g = st["gen"]
                log("[pool] notify job_id=%s height=%s gen=%d target=%s" % (
                    p.get("job_id"), p.get("height"), g, p.get("target")))
            else:
                # AUTH ack / submit response / error — the ground truth we want.
                stats["resps"] += 1
                mid = m.get("id")
                with lock:
                    is_submit = mid in st["submit_ids"]
                    if is_submit:
                        st["got_resp"] = True
                tag = "*** SUBMIT RESPONSE ***" if is_submit else "(other ack)"
                log("[pool] <<", tag, txt[:600])

def main():
    if not os.path.exists(GEN):
        log("FATAL: solver binary not found:", GEN); return
    log("[drv] M1c kryptex de-risk start; GEN=%s WORKER=%s MAXSEC=%d" % (GEN, WORKER, MAXSEC))
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    s = ctx.wrap_socket(socket.create_connection((HOST, PORT), timeout=30), server_hostname=HOST)
    log("[pool] TLS", s.version(), s.cipher()[0])
    th = threading.Thread(target=reader, args=(s,), daemon=True); th.start()
    send(s, {"id": 1, "method": "mining.authorize",
             "params": {"wallet": WALLET, "worker": WORKER, "agent": "pearl-miner/0.1"}})

    t0 = time.time()
    while time.time() - t0 < 25:
        with lock:
            if st["header"]:
                break
        time.sleep(0.2)
    with lock:
        if not st["header"]:
            log("[drv] no job within 25s; abort"); st["stop"] = True; s.close(); return

    start = time.time(); submit_id = 100
    while not st["stop"]:
        if MAXSEC and time.time() - start > MAXSEC:
            log("[drv] MAXSEC=%ds reached, stopping" % MAXSEC); break
        with lock:
            hdr = st["header"]; tgt = st["target"]; jid = st["job_id"]; cur_gen = st["gen"]
            if st["got_resp"] and STOP_ON_ACK:
                log("[drv] got a submit response — de-risk question ANSWERED, stopping"); break
        cmd = [GEN, "--mine", str(BATCH), "--header", hdr, "--target", tgt]
        bt = time.time()
        log("[drv] spawn batch gen=%d job_id=%s target=%s" % (cur_gen, jid, tgt))
        p = subprocess.Popen(cmd, cwd=GENDIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        killed = False
        while True:
            rc = p.poll()
            if rc is not None:
                break
            with lock:
                g = st["gen"]; stop = st["stop"]; got = st["got_resp"]
            if g != cur_gen or stop or (got and STOP_ON_ACK):
                p.terminate(); killed = True; break
            if MAXSEC and time.time() - start > MAXSEC:
                p.terminate(); killed = True; break
            time.sleep(0.1)
        out, err = p.communicate()
        if err:
            for ln in err.strip().splitlines()[-3:]:
                log("[gen]", ln)
        if killed:
            log("[drv] batch gen=%d aborted after %.1fs (job change/stop → restart latest)" % (
                cur_gen, time.time() - bt))
            continue
        b64 = (out or "").strip()
        if p.returncode == 0 and b64:
            with lock:
                fresh = (st["gen"] == cur_gen); sjid = st["job_id"]
                st["submit_ids"].add(submit_id)
            stats["wins"] += 1; stats["submits"] += 1
            log("[drv] *** WIN *** %d b64 chars gen=%d fresh=%s → mining.submit id=%d job_id=%s" % (
                len(b64), cur_gen, fresh, submit_id, sjid))
            try:
                with open(WIN_FILE, "w") as f:
                    json.dump({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "submit_id": submit_id,
                               "job_id": sjid, "target": tgt, "header": hdr, "fresh": fresh,
                               "proof_b64": b64}, f)
                log("[drv] saved winning proof -> %s" % WIN_FILE)
            except Exception as e:
                log("[drv] WARN could not save win file:", e)
            send(s, {"id": submit_id, "method": "mining.submit",
                     "params": {"job_id": sjid, "wallet": WALLET, "worker": WORKER, "proof": b64}})
            submit_id += 1
            # wait up to 30s for the pool's verdict before next batch
            wt = time.time()
            while time.time() - wt < 30:
                with lock:
                    if st["got_resp"] or st["stop"]:
                        break
                time.sleep(0.5)
            with lock:
                got = st["got_resp"]
            if not got:
                log("[drv] no submit response within 30s (SILENT, like prior probe)")
            if STOP_AFTER_WIN:
                log("[drv] STOP_AFTER_WIN: first real share submitted (acked=%s) — de-risk data captured" % got)
                break
        else:
            log("[drv] batch gen=%d: no win (rc=%s) in %.1fs" % (cur_gen, p.returncode, time.time() - bt))
    el = time.time() - start
    log("[drv] DONE elapsed=%.0fs jobs=%d wins/submits=%d/%d pool_responses=%d got_resp=%s" % (
        el, stats["jobs"], stats["wins"], stats["submits"], stats["resps"], st["got_resp"]))
    st["stop"] = True
    try: s.close()
    except Exception: pass

if __name__ == "__main__":
    main()
