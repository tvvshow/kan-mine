#!/usr/bin/env python3
# M1c pre-flight: prove THIS runner can reach the kryptex stratum pool,
# authorize with our wallet, and receive a real job. Pure network — no GPU,
# no mining, no submit. Exit 0 only if BOTH auth-ack AND a job were received.
#
# This gates the full M1c hunt: if the cnb GPU runner has no egress to the
# pool (or auth is rejected), the hunt can't run here and we must change infra
# BEFORE spending ~1.8h of GPU mining for a share.
import socket, ssl, json, sys, time, os

HOST   = "prl.kryptex.network"
PORT   = 8048
WALLET = "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv"
WORKER = os.environ.get("WORKER", "probe")
TIMEOUT = int(os.environ.get("PROBE_SEC", "30"))

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    log("[probe] resolving + connecting %s:%d (timeout 15s)" % (HOST, PORT))
    try:
        raw = socket.create_connection((HOST, PORT), timeout=15)
    except Exception as e:
        log("[probe] FAIL tcp connect:", repr(e))
        return 2
    log("[probe] TCP connected; starting TLS handshake")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        s = ctx.wrap_socket(raw, server_hostname=HOST)
    except Exception as e:
        log("[probe] FAIL tls handshake:", repr(e))
        return 3
    log("[probe] TLS up:", s.version(), s.cipher()[0])
    s.sendall((json.dumps({"id": 1, "method": "mining.authorize",
        "params": {"wallet": WALLET, "worker": WORKER, "agent": "pearl-miner/0.1"}}) + "\n").encode())
    log("[probe] sent mining.authorize wallet=%s worker=%s" % (WALLET[:12] + "...", WORKER))
    s.settimeout(1.0)
    buf = bytearray()
    t0 = time.time()
    got_auth = False
    got_job = False
    while time.time() - t0 < TIMEOUT:
        try:
            d = s.recv(8192)
        except socket.timeout:
            continue
        except Exception as e:
            log("[probe] recv error:", repr(e))
            break
        if not d:
            log("[probe] EOF (pool closed connection)")
            break
        buf.extend(d)
        while b"\n" in buf:
            i = buf.index(b"\n")
            txt = bytes(buf[:i]).decode(errors="replace").strip()
            del buf[:i + 1]
            if not txt:
                continue
            log("[probe] <<", txt[:400])
            try:
                m = json.loads(txt)
            except Exception:
                continue
            if m.get("id") == 1:
                got_auth = True
                if m.get("error"):
                    log("[probe] WARN authorize returned error:", m.get("error"))
            if m.get("method") == "mining.notify":
                got_job = True
                p = m.get("params", {})
                log("[probe] JOB header_len=%s target=%s height=%s job_id=%s" % (
                    len(p.get("header", "")), p.get("target"), p.get("height"), p.get("job_id")))
        if got_auth and got_job:
            break
    try:
        s.close()
    except Exception:
        pass
    log("[probe] RESULT auth_ack=%s job_received=%s elapsed=%.1fs" % (
        got_auth, got_job, time.time() - t0))
    if got_auth and got_job:
        log("[probe] PASS — cnb runner CAN mine this pool; M1c hunt is viable here")
        return 0
    log("[probe] FAIL — connectivity/auth incomplete; do NOT launch the hunt here")
    return 1

if __name__ == "__main__":
    sys.exit(main())
