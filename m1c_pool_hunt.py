#!/usr/bin/env python3
# M1c — kryptex pool-acceptance: land ONE real-config share from OUR fused
# tensor-core solver (plainproof_gen --mine --tc --cfg real) and CAPTURE the
# pool's verdict. The reply to mining.submit IS the M1c ground truth.
#
# WIRE PROTOCOL = the LuckyPool contract captured by MITM-ing lpminer
# (memory: reference_kryptex_stratum_wire):
#   * prl.kryptex.network:7048  PLAINTEXT TCP  (NO TLS, NO mining.subscribe)
#   * authorize params = {"wallet":"<addr>.<worker>", "worker":<worker>,
#                         "agent":"lpminer/0.1.9-552bdfe"}
#   * notify   params  = {"header":<152hex>, "height":<int>,
#                         "job_id":"<8hex>_<sharediff>", "target":<64hex BE>}
#   * submit   params  = {"job_id":<id>, "plain_proof": base64(bincode PlainProof),
#                         "hs": <int hashrate>}   (NB: key is plain_proof, + hs;
#                         NO wallet/worker in submit)
#   * reply {"result":true}=ACCEPTED ; {"result":false,"error":[-1,"Invalid share",null]}=REJECTED
#
# We mine the LATEST job, abort+restart on job change (draws are independent),
# submit the first win immediately, and log EVERY non-notify line verbatim.
#
# Env: WORKER, GEN, GENDIR, BATCH, HS, MAXSEC (0=forever), GEN_EXTRA (solver flags),
#      STOP_ON_ACK (stop after first submit response), STOP_AFTER_WIN, WIN_FILE,
#      POOL_HOST, POOL_PORT, WALLET, AGENT.
import socket, json, sys, time, threading, subprocess, os, select

HOST   = os.environ.get("POOL_HOST", "prl.kryptex.network")
PORT   = int(os.environ.get("POOL_PORT", "7048"))
WALLET = os.environ.get("WALLET", "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv")
WORKER = os.environ.get("WORKER", "m1ctc")
AGENT  = os.environ.get("AGENT", "lpminer/0.1.9-552bdfe")
GEN    = os.environ.get("GEN", "/root/peral/build/plainproof_gen")
GENDIR = os.environ.get("GENDIR", "/root/peral/build")
BATCH  = int(os.environ.get("BATCH", "1000000"))
HS     = int(os.environ.get("HS", "8000000"))          # reported hashrate (telemetry only)
MAXSEC = int(os.environ.get("MAXSEC", "0"))
STOP_ON_ACK    = int(os.environ.get("STOP_ON_ACK", "1"))
STOP_AFTER_WIN = int(os.environ.get("STOP_AFTER_WIN", "1"))
WIN_FILE = os.environ.get("WIN_FILE", "/root/peral/m1c_win.json")
GEN_EXTRA = os.environ.get("GEN_EXTRA", "--tc --cfg real").split()

st = {"header": None, "target": None, "job_id": None, "height": None, "gen": 0,
      "stop": False, "submit_ids": set(), "submit_resps": {}, "got_resp": False}
lock = threading.Lock()
io_lock = threading.Lock()
stats = {"jobs": 0, "submits": 0, "resps": 0, "wins": 0,
         "accepted": 0, "rejected": 0, "stale_drops": 0}

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def send(s, o):
    data = (json.dumps(o) + "\n").encode()
    with io_lock:
        s.sendall(data)

def same_job(p_or_state, job_id, header, target, height):
    return (p_or_state.get("job_id") == job_id and
            p_or_state.get("header") == header and
            p_or_state.get("target") == target and
            p_or_state.get("height") == height)

def reader(s):
    buf = bytearray()
    while not st["stop"]:
        r, _, _ = select.select([s], [], [], 0.5)
        if not r:
            continue
        try:
            d = s.recv(8192)
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
                    changed = not same_job(p, st["job_id"], st["header"], st["target"], st["height"])
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
                        st["submit_resps"][mid] = m
                        if m.get("result") is True:
                            stats["accepted"] += 1
                        else:
                            stats["rejected"] += 1
                tag = "*** SUBMIT RESPONSE ***" if is_submit else "(other ack)"
                log("[pool] <<", tag, txt[:600])

def main():
    if not os.path.exists(GEN):
        log("FATAL: solver binary not found:", GEN); sys.exit(2)
    log("[drv] M1c kryptex hunt start; GEN=%s WORKER=%s MAXSEC=%d EXTRA=%s" % (
        GEN, WORKER, MAXSEC, " ".join(GEN_EXTRA)))
    s = socket.create_connection((HOST, PORT), timeout=30)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    log("[pool] connected %s:%d (plaintext)" % (HOST, PORT))
    th = threading.Thread(target=reader, args=(s,), daemon=True); th.start()
    send(s, {"id": 1, "method": "mining.authorize",
             "params": {"wallet": "%s.%s" % (WALLET, WORKER), "worker": WORKER, "agent": AGENT}})
    log("[drv] >> authorize wallet=%s.%s agent=%s" % (WALLET, WORKER, AGENT))

    t0 = time.time()
    while time.time() - t0 < 25:
        with lock:
            if st["header"]:
                break
        time.sleep(0.2)
    with lock:
        if not st["header"]:
            log("[drv] no job within 25s; abort"); st["stop"] = True; s.close(); sys.exit(3)

    start = time.time(); submit_id = 100
    while not st["stop"]:
        if MAXSEC and time.time() - start > MAXSEC:
            log("[drv] MAXSEC=%ds reached, stopping" % MAXSEC); break
        with lock:
            hdr = st["header"]; tgt = st["target"]; jid = st["job_id"]
            height = st["height"]; cur_gen = st["gen"]
            if st["got_resp"] and STOP_ON_ACK:
                log("[drv] got a submit response ? M1c question ANSWERED, stopping"); break
        if not (hdr and tgt and jid):
            log("[drv] no complete job snapshot yet; waiting")
            time.sleep(0.5)
            continue

        # Freeze the exact job tuple used by the solver.  A proof mined for one
        # (job_id, header, target, height) MUST NOT be submitted against a later
        # job_id; that stale race was a direct cause of 100% rejected shares.
        cmd = [GEN, "--mine", str(BATCH)] + GEN_EXTRA + ["--header", hdr, "--target", tgt]
        bt = time.time()
        log("[drv] spawn gen=%d job_id=%s height=%s target=%s :: %s" % (
            cur_gen, jid, height, tgt, " ".join(cmd)))
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
        try:
            out, err = p.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill(); out, err = p.communicate()
        if err:
            for ln in err.strip().splitlines()[-6:]:
                log("[gen]", ln)
        if killed:
            log("[drv] gen=%d aborted after %.1fs (job change/stop -> restart latest)" % (
                cur_gen, time.time() - bt))
            continue
        b64 = (out or "").strip().splitlines()[-1].strip() if (out or "").strip() else ""
        if p.returncode == 0 and b64:
            with lock:
                fresh = (st["gen"] == cur_gen and same_job(st, jid, hdr, tgt, height))
            if not fresh:
                with lock:
                    stats["stale_drops"] += 1
                    cur_jid = st["job_id"]; cur_height = st["height"]; cur_gen_now = st["gen"]
                log("[drv] DROP STALE WIN: mined gen=%d job_id=%s height=%s, current gen=%d job_id=%s height=%s" % (
                    cur_gen, jid, height, cur_gen_now, cur_jid, cur_height))
                continue

            with lock:
                st["submit_ids"].add(submit_id)
            stats["wins"] += 1; stats["submits"] += 1
            log("[drv] *** WIN *** %d b64 chars gen=%d fresh=%s -> mining.submit id=%d job_id=%s" % (
                len(b64), cur_gen, True, submit_id, jid))
            try:
                with open(WIN_FILE, "w") as f:
                    json.dump({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "submit_id": submit_id,
                               "job_id": jid, "target": tgt, "header": hdr, "height": height,
                               "fresh": True, "proof_b64": b64}, f)
                log("[drv] saved winning proof -> %s" % WIN_FILE)
            except Exception as e:
                log("[drv] WARN could not save win file:", e)
            this_submit_id = submit_id
            send(s, {"id": submit_id, "method": "mining.submit",
                     "params": {"job_id": jid, "plain_proof": b64, "hs": HS}})
            log("[drv] >> submit id=%d job_id=%s plain_proof=%dB hs=%d" % (submit_id, jid, len(b64), HS))
            submit_id += 1
            wt = time.time()
            while time.time() - wt < 30:
                with lock:
                    if this_submit_id in st["submit_resps"] or st["stop"]:
                        break
                time.sleep(0.5)
            with lock:
                resp = st["submit_resps"].get(this_submit_id)
                got = resp is not None
            if not got:
                log("[drv] no submit response within 30s (SILENT)")
            else:
                log("[drv] submit verdict id=%d result=%s error=%s" % (
                    this_submit_id, resp.get("result"), resp.get("error")))
            if STOP_AFTER_WIN:
                log("[drv] STOP_AFTER_WIN: first real share submitted (acked=%s)" % got)
                break
        else:
            log("[drv] gen=%d: no win (rc=%s) in %.1fs" % (cur_gen, p.returncode, time.time() - bt))
    el = time.time() - start
    log("[drv] DONE elapsed=%.0fs jobs=%d wins/submits=%d/%d accepted=%d rejected=%d stale_drops=%d pool_responses=%d got_resp=%s" % (
        el, stats["jobs"], stats["wins"], stats["submits"], stats["accepted"],
        stats["rejected"], stats["stale_drops"], stats["resps"], st["got_resp"]))
    st["stop"] = True
    try: s.close()
    except Exception: pass

if __name__ == "__main__":
    main()
