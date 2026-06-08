#!/usr/bin/env python3
# M1c pre-flight: prove THIS runner can reach the Kryptex/LuckyPool Pearl stratum,
# authorize with the lpminer-compatible shape, and receive a real job.  Read-only:
# no mining and no submit.  Defaults match the captured lpminer contract:
# plaintext TCP 7048, authorize params as a dict, wallet includes .worker.
import socket, ssl, json, sys, time, os

HOST   = os.environ.get("POOL_HOST", "prl.kryptex.network")
PORT   = int(os.environ.get("POOL_PORT", "7048"))
USE_TLS = os.environ.get("POOL_TLS", "0") == "1"
WALLET = os.environ.get("WALLET", "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv")
WORKER = os.environ.get("WORKER", "probe")
AGENT  = os.environ.get("AGENT", "lpminer/0.1.9-552bdfe")
TIMEOUT = int(os.environ.get("PROBE_SEC", "30"))

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    mode = "TLS" if USE_TLS else "plaintext"
    log("[probe] connecting %s:%d (%s, timeout 15s)" % (HOST, PORT, mode))
    try:
        raw = socket.create_connection((HOST, PORT), timeout=15)
    except Exception as e:
        log("[probe] FAIL tcp connect:", repr(e)); return 2
    if USE_TLS:
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        try:
            s = ctx.wrap_socket(raw, server_hostname=HOST)
        except Exception as e:
            log("[probe] FAIL tls handshake:", repr(e)); return 3
        log("[probe] TLS up:", s.version(), s.cipher()[0])
    else:
        s = raw
    auth = {"id": 1, "method": "mining.authorize",
            "params": {"wallet": f"{WALLET}.{WORKER}", "worker": WORKER, "agent": AGENT}}
    s.sendall((json.dumps(auth) + "\n").encode())
    log("[probe] sent mining.authorize wallet=%s.%s agent=%s" % (WALLET[:12] + "...", WORKER, AGENT))
    s.settimeout(1.0)
    buf = bytearray(); got_auth = False; got_job = False; job = None
    t0 = time.time()
    while time.time() - t0 < TIMEOUT:
        try:
            d = s.recv(8192)
        except socket.timeout:
            continue
        except Exception as e:
            log("[probe] recv error:", repr(e)); break
        if not d:
            log("[probe] EOF (pool closed connection)"); break
        buf.extend(d)
        while b"\n" in buf:
            i = buf.index(b"\n")
            txt = bytes(buf[:i]).decode(errors="replace").strip(); del buf[:i+1]
            if not txt: continue
            log("[probe] <<", txt[:500])
            try: m = json.loads(txt)
            except Exception: continue
            if m.get("id") == 1:
                got_auth = (m.get("result") is True)
                if m.get("error"): log("[probe] WARN authorize returned error:", m.get("error"))
            if m.get("method") == "mining.notify":
                got_job = True; job = m.get("params", {})
                log("[probe] JOB header_len=%s target=%s height=%s job_id=%s" % (
                    len(job.get("header", "")), job.get("target"), job.get("height"), job.get("job_id")))
        if got_auth and got_job: break
    try: s.close()
    except Exception: pass
    log("[probe] RESULT auth_ack=%s job_received=%s elapsed=%.1fs" % (got_auth, got_job, time.time() - t0))
    if got_auth and got_job:
        log("[probe] PASS - runner can mine this pool with the captured lpminer contract")
        return 0
    log("[probe] FAIL - connectivity/auth/job incomplete; do NOT launch the hunt here")
    return 1

if __name__ == "__main__":
    sys.exit(main())
