#!/usr/bin/env python3
# Submit-contract probe for Kryptex/LuckyPool Pearl.  This does NOT mine and does
# NOT send a real-job proof.  It only sends bogus-job submits to learn whether the
# pool parser recognizes each JSON shape.  The correct captured lpminer shape is
# expected to return "Job not found" for a bogus job_id; legacy shapes usually go
# silent and must not be used for production mining.
import socket, ssl, json, time, os, sys

HOST   = os.environ.get("POOL_HOST", "prl.kryptex.network")
PORT   = int(os.environ.get("POOL_PORT", "7048"))
USE_TLS = os.environ.get("POOL_TLS", "0") == "1"
WALLET = os.environ.get("WALLET", "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv")
WORKER = os.environ.get("WORKER", "probe2")
AGENT  = os.environ.get("AGENT", "lpminer/0.1.9-552bdfe")

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    raw = socket.create_connection((HOST, PORT), timeout=30)
    if USE_TLS:
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        s = ctx.wrap_socket(raw, server_hostname=HOST)
        log("TLS up:", s.version())
    else:
        s = raw
        log("TCP up: %s:%d plaintext" % (HOST, PORT))
    s.settimeout(1.0)
    buf = bytearray(); acks = {}; last_job = {}

    def send(o):
        shown = json.loads(json.dumps(o))
        pa = shown.get("params")
        if isinstance(pa, dict):
            for k in ("plain_proof", "proof"):
                if k in pa and isinstance(pa[k], str): pa[k] = "<%d chars>" % len(pa[k])
        log(">>", json.dumps(shown, separators=(",", ":"))[:500])
        s.sendall((json.dumps(o) + "\n").encode())

    def pump(timeout):
        end = time.time() + timeout
        while time.time() < end:
            try:
                d = s.recv(8192)
            except socket.timeout:
                continue
            if not d:
                log("<<EOF"); return
            buf.extend(d)
            while b"\n" in buf:
                i = buf.index(b"\n"); txt = bytes(buf[:i]).decode(errors="replace").strip(); del buf[:i+1]
                if not txt: continue
                log("<<", txt[:500])
                try: m = json.loads(txt)
                except Exception: continue
                if m.get("method") == "mining.notify": last_job.update(m.get("params", {}))
                if m.get("id") is not None: acks[m["id"]] = m

    send({"id": 1, "method": "mining.authorize",
          "params": {"wallet": f"{WALLET}.{WORKER}", "worker": WORKER, "agent": AGENT}})
    pump(8)
    bogus = "deadbeef_1048576"
    bad_b64 = "not_base64!!!"

    log("--- #10 captured lpminer shape: bogus job + plain_proof + hs ---")
    send({"id": 10, "method": "mining.submit",
          "params": {"job_id": bogus, "plain_proof": bad_b64, "hs": 1}})
    pump(6)

    log("--- #11 legacy shape: bogus job + wallet/worker/proof ---")
    send({"id": 11, "method": "mining.submit",
          "params": {"job_id": bogus, "wallet": WALLET, "worker": WORKER, "proof": bad_b64}})
    pump(6)

    ok10 = 10 in acks
    ok11 = 11 in acks
    log("SUMMARY acks: captured_shape=%s legacy_shape=%s" % (ok10, ok11))
    if ok10:
        log("VERDICT: captured lpminer submit shape is recognized by the pool")
    else:
        log("VERDICT: captured shape was not acked; re-capture lpminer wire before mining")
    if ok11:
        log("WARN: legacy shape got an ack too; inspect response before deciding")
    else:
        log("VERDICT: legacy wallet/worker/proof shape is not a safe production shape")
    try: s.close()
    except Exception: pass
    return 0 if ok10 else 1

if __name__ == "__main__":
    sys.exit(main())
