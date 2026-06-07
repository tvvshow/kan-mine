#!/usr/bin/env python3
# M1c v2 disambiguation: does kryptex ACK our mining.submit SHAPE at all?
# The hunt found a REAL share-target win but the pool was SILENT on the submit
# (same as losers). auth IS acked, so the pool speaks req/resp -> silence on submit
# means our submit is likely UNRECOGNIZED. This isolates that:
#   #10 real job_id + well-formed proof  -> silence = low-diff drop OR format issue
#   #11 BOGUS job_id + same proof        -> an "unknown/stale job" ack PROVES our shape is recognized
#   #12 real job_id + garbage proof      -> a parse-error ack calibrates the reject path
# All-silent => the pool doesn't recognize our submit method/shape (need lpminer's real format).
# Env: WORKER, GEN (solver, for a well-formed proof), SUBSCRIBE (1 = send mining.subscribe first).
import socket, ssl, json, subprocess, sys, time, os

HOST   = "prl.kryptex.network"
PORT   = 8048
WALLET = "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv"
WORKER = os.environ.get("WORKER", "probe2")
GEN    = os.environ.get("GEN", "./build/plainproof_gen")
DO_SUB = os.environ.get("SUBSCRIBE", "0") == "1"

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    # A well-formed proof from the golden-nbits CPU reference path (no GPU needed).
    if os.path.exists(GEN):
        try:
            r = subprocess.run([GEN, "12345"], capture_output=True, text=True, timeout=180)
            b64 = (r.stdout or "").strip()
            log("gen proof: rc=%s b64len=%d" % (r.returncode, len(b64)))
        except Exception as e:
            b64 = "A" * 58264
            log("gen FAILED (%r); using dummy b64 len=%d" % (e, len(b64)))
    else:
        b64 = "A" * 58264
        log("GEN missing at %s; using dummy b64 len=%d" % (GEN, len(b64)))
    if not b64:
        b64 = "A" * 58264
        log("empty proof; using dummy b64 len=%d" % len(b64))

    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    s = ctx.wrap_socket(socket.create_connection((HOST, PORT), timeout=30), server_hostname=HOST)
    log("TLS up:", s.version())
    buf = bytearray(); last_job = {}; acks = {}

    def send(o):
        s.sendall((json.dumps(o) + "\n").encode())
        d = dict(o)
        if isinstance(d.get("params"), dict) and "proof" in d["params"]:
            d["params"] = {**d["params"], "proof": "<%dB>" % len(d["params"]["proof"])}
        log(">>", json.dumps(d))

    def pump(timeout):
        end = time.time() + timeout
        while time.time() < end:
            s.settimeout(max(0.1, end - time.time()))
            try:
                d = s.recv(8192)
            except socket.timeout:
                break
            if not d:
                log("<<EOF (pool closed)"); break
            buf.extend(d)
            while b"\n" in buf:
                i = buf.index(b"\n"); txt = bytes(buf[:i]).decode(errors="replace").strip(); del buf[:i + 1]
                if not txt:
                    continue
                try:
                    m = json.loads(txt)
                except Exception:
                    log("<<RAW", txt[:200]); continue
                if m.get("method") == "mining.notify":
                    last_job.clear(); last_job.update(m.get("params", {})); continue
                log("<<", txt[:300])
                if m.get("id") is not None:
                    acks[m["id"]] = m

    if DO_SUB:
        send({"id": 0, "method": "mining.subscribe", "params": {"agent": "pearl-miner/0.1"}})
        pump(4)
    send({"id": 1, "method": "mining.authorize",
          "params": {"wallet": WALLET, "worker": WORKER, "agent": "pearl-miner/0.1"}})
    pump(8)
    if not last_job.get("header"):
        pump(6)
    jid = last_job.get("job_id", "unknown_1048576")
    log("JOB job_id=%s height=%s target=%s" % (jid, last_job.get("height"), last_job.get("target")))

    log("--- #10 real job_id + well-formed proof ---")
    send({"id": 10, "method": "mining.submit",
          "params": {"job_id": jid, "wallet": WALLET, "worker": WORKER, "proof": b64}})
    pump(12)
    log("--- #11 BOGUS job_id + same proof ---")
    send({"id": 11, "method": "mining.submit",
          "params": {"job_id": "deadbeef_1048576", "wallet": WALLET, "worker": WORKER, "proof": b64}})
    pump(12)
    log("--- #12 real job_id + garbage proof ---")
    send({"id": 12, "method": "mining.submit",
          "params": {"job_id": jid, "wallet": WALLET, "worker": WORKER, "proof": "not_base64!!!"}})
    pump(12)

    log("SUMMARY subscribe=%s acks: #10=%s #11=%s #12=%s" % (DO_SUB, 10 in acks, 11 in acks, 12 in acks))
    for k in (10, 11, 12):
        if k in acks:
            log("  ack #%d:" % k, json.dumps(acks[k])[:240])
    if not any(k in acks for k in (10, 11, 12)):
        log("VERDICT: ALL submit variants SILENT -> our submit method/shape is NOT recognized by the pool")
    else:
        log("VERDICT: pool DID ack a submit variant -> our shape IS recognized; analyze which to learn the contract")
    try:
        s.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
