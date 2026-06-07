#!/usr/bin/env python3
# stratum_capture_proxy.py — learn kryptex's Pearl share-submit CONTRACT by
# man-in-the-middling the CLOSED lpminer against the REAL kryptex pool.
#
# WHY: our PlainProof is officially correct end to end (proved offline against
# zk_pow::api::verify::verify_plain_proof — CPU and the GPU --mine hunt path both
# VALID). Yet kryptex credits 0 shares, so the gap is purely kryptex's THIRD-PARTY
# stratum submit contract (method name / proof payload encoding / target semantics)
# that only lpminer knows. This proxy captures lpminer's exact bytes = ground truth.
#
# TOPOLOGY:   lpminer  <->  THIS PROXY  <--TLS-->  prl.kryptex.network:8048
# The proxy relays the REAL handshake + jobs (so we also see kryptex's exact notify /
# extranonce / subscribe-reply), but REWRITES each job's `target` to f*64 (easiest
# possible) so lpminer's very first GEMM attempt wins and emits mining.submit within
# seconds — even on the cheapest CUDA GPU. The submit line is logged VERBATIM (and
# appended to a .jsonl) before being forwarded upstream (upstream may reject the easy
# share — irrelevant, we already have the bytes).
#
# TWO LISTEN MODES (pick whichever lpminer accepts):
#   --listen-plain  : proxy listens PLAINTEXT tcp; point lpminer at stratum+tcp://HOST:PORT.
#                     Dodges all cert problems (lpminer<->proxy is cleartext; proxy<->kryptex is TLS).
#   --listen-tls    : proxy listens TLS with a self-signed cert (auto-generated via openssl);
#                     use if lpminer FORCES ssl. lpminer must accept a self-signed cert
#                     (look for an insecure/no-verify flag in `lpminer --help`).
#
# stdlib only. Run on the rented GPU box next to lpminer.
import argparse, json, os, socket, ssl, subprocess, sys, threading, time

LOCK = threading.Lock()
START = None  # set in main (Date.now-free not needed here; this is plain python)

def ts():
    return time.strftime("%H:%M:%S") + (".%03d" % int((time.time() % 1) * 1000))

def log(line, fh):
    with LOCK:
        sys.stdout.write(line + "\n"); sys.stdout.flush()
        if fh:
            fh.write(line + "\n"); fh.flush()

def make_self_signed(cert, key, cn):
    if os.path.exists(cert) and os.path.exists(key):
        return
    print("[*] generating self-signed cert CN=%s -> %s / %s" % (cn, cert, key))
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", key, "-out", cert, "-days", "3650", "-subj", "/CN=%s" % cn],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def downgrade_obj(obj, easy_hex):
    """Rewrite any job 'target'/'difficulty' to the easiest value so lpminer wins at once.
    Returns (changed, note). Only touches known target-carrying shapes to avoid clobbering
    the header/job_key hex. Handles dict-params (kryptex) and classic list/number forms."""
    changed = []
    if not isinstance(obj, dict):
        return False, ""
    method = obj.get("method")

    def fix_target_value(v):
        # hex string -> all-f of same length; int -> 2^(bits)-1; else leave
        if isinstance(v, str) and len(v) >= 8 and all(c in "0123456789abcdefABCDEF" for c in v):
            return easy_hex if len(v) == len(easy_hex) else "f" * len(v)
        if isinstance(v, int):
            return (1 << 256) - 1
        return v

    # (a) dict params with a 'target' / 'difficulty' key (kryptex mining.notify shape)
    p = obj.get("params")
    if isinstance(p, dict):
        for k in ("target", "difficulty", "diff"):
            if k in p:
                old = p[k]; p[k] = fix_target_value(old)
                if p[k] != old: changed.append("params.%s %s->%s" % (k, str(old)[:24], str(p[k])[:16]))
    # (b) classic mining.set_difficulty/set_target with list/number params
    if method in ("mining.set_difficulty", "mining.set_target", "set_difficulty", "set_target"):
        if isinstance(p, list) and p:
            old = p[0]; p[0] = fix_target_value(old)
            if p[0] != old: changed.append("%s[0] %s->%s" % (method, str(old)[:24], str(p[0])[:16]))
    # (c) result-embedded target (some pools put it in subscribe result)
    r = obj.get("result")
    if isinstance(r, dict):
        for k in ("target", "difficulty", "diff"):
            if k in r:
                old = r[k]; r[k] = fix_target_value(old)
                if r[k] != old: changed.append("result.%s %s->%s" % (k, str(old)[:24], str(r[k])[:16]))
    return (len(changed) > 0), "; ".join(changed)

def redact_proof_for_log(obj):
    """Shorten huge proof blobs in a COPY so the relay log stays readable.
    (The verbatim submit is saved separately, un-redacted.)"""
    try:
        d = json.loads(json.dumps(obj))
    except Exception:
        return obj
    pa = d.get("params")
    if isinstance(pa, dict):
        for k, v in list(pa.items()):
            if isinstance(v, str) and len(v) > 120:
                pa[k] = "<%dB:%s...>" % (len(v), v[:32])
    return d

def pump(src, dst, direction, easy_hex, do_downgrade, fh, submits_path):
    buf = bytearray()
    tag = "C>U" if direction == "up" else "U>C"
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                log("[%s] %s <EOF>" % (ts(), tag), fh); break
            buf.extend(chunk)
            while b"\n" in buf:
                i = buf.index(b"\n"); raw = bytes(buf[:i]); del buf[: i + 1]
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                out_bytes = raw + b"\n"
                obj = None
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = None

                if direction == "up":
                    # client(lpminer) -> upstream(kryptex). The SUBMIT is the prize.
                    m = obj.get("method") if isinstance(obj, dict) else None
                    is_submit = isinstance(m, str) and "submit" in m.lower()
                    if is_submit:
                        log("[%s] %s *** SUBMIT (VERBATIM, %dB) ***" % (ts(), tag, len(line)), fh)
                        log(line, fh)  # full, untruncated
                        with LOCK:
                            with open(submits_path, "a") as sf:
                                sf.write(line + "\n")
                        log("[%s] %s ^ saved to %s" % (ts(), tag, submits_path), fh)
                    else:
                        # handshake (subscribe/authorize/...) — also interesting, log full
                        log("[%s] %s %s" % (ts(), tag, line[:600]), fh)
                else:
                    # upstream -> client. Downgrade target so lpminer wins instantly.
                    if do_downgrade and isinstance(obj, dict):
                        changed, note = downgrade_obj(obj, easy_hex)
                        if changed:
                            out_bytes = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
                            log("[%s] %s [DOWNGRADED %s]" % (ts(), tag, note), fh)
                    shown = redact_proof_for_log(obj) if isinstance(obj, dict) else line
                    log("[%s] %s %s" % (ts(), tag,
                        json.dumps(shown, separators=(",", ":"))[:600] if isinstance(obj, dict) else line[:600]), fh)

                dst.sendall(out_bytes)
    except Exception as e:
        log("[%s] %s pump error: %r" % (ts(), tag, e), fh)
    finally:
        try: dst.shutdown(socket.SHUT_WR)
        except Exception: pass

def handle(client, addr, args, fh):
    log("[%s] === client connected: %s ===" % (ts(), addr), fh)
    uctx = ssl.create_default_context()
    uctx.check_hostname = False; uctx.verify_mode = ssl.CERT_NONE
    try:
        raw_up = socket.create_connection((args.upstream_host, args.upstream_port), timeout=30)
        up = uctx.wrap_socket(raw_up, server_hostname=args.upstream_host)
        log("[%s] upstream TLS up: %s -> %s:%d" % (ts(), up.version(), args.upstream_host, args.upstream_port), fh)
    except Exception as e:
        log("[%s] FAILED upstream connect: %r" % (ts(), e), fh)
        try: client.close()
        except Exception: pass
        return
    t1 = threading.Thread(target=pump, args=(client, up, "up", args.easy_target, not args.no_downgrade, fh, args.submits), daemon=True)
    t2 = threading.Thread(target=pump, args=(up, client, "down", args.easy_target, not args.no_downgrade, fh, args.submits), daemon=True)
    t1.start(); t2.start(); t1.join(); t2.join()
    for s in (client, up):
        try: s.close()
        except Exception: pass
    log("[%s] === client disconnected: %s ===" % (ts(), addr), fh)

def main():
    ap = argparse.ArgumentParser(description="Logging + target-downgrading stratum MITM proxy (capture kryptex submit contract via lpminer)")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=8048)
    ap.add_argument("--upstream-host", default="prl.kryptex.network")
    ap.add_argument("--upstream-port", type=int, default=8048)
    ap.add_argument("--listen-tls", action="store_true", help="listen with TLS (self-signed); default is plaintext")
    ap.add_argument("--cert", default="proxy_cert.pem")
    ap.add_argument("--key", default="proxy_key.pem")
    ap.add_argument("--easy-target", default="f" * 64, help="64-hex target forwarded to lpminer (default easiest)")
    ap.add_argument("--no-downgrade", action="store_true", help="relay target unchanged (capture from a REAL win instead)")
    ap.add_argument("--log", default="capture.log")
    ap.add_argument("--submits", default="captured_submits.jsonl")
    args = ap.parse_args()

    fh = open(args.log, "a")
    mode = "TLS(self-signed)" if args.listen_tls else "PLAINTEXT"
    log("[%s] === proxy start: listen %s %s:%d  upstream TLS %s:%d  downgrade=%s ===" % (
        ts(), mode, args.listen_host, args.listen_port, args.upstream_host, args.upstream_port,
        "OFF" if args.no_downgrade else ("ON->%s" % args.easy_target[:8] + "...")), fh)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.listen_host, args.listen_port)); srv.listen(8)

    sctx = None
    if args.listen_tls:
        make_self_signed(args.cert, args.key, args.upstream_host)
        sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        sctx.load_cert_chain(args.cert, args.key)

    print("[*] point lpminer at: %s://%s:%d" % ("stratum+ssl" if args.listen_tls else "stratum+tcp",
                                                 args.listen_host, args.listen_port))
    print("[*] submits -> %s   full relay log -> %s" % (args.submits, args.log))
    try:
        while True:
            cli, addr = srv.accept()
            if sctx is not None:
                try:
                    cli = sctx.wrap_socket(cli, server_side=True)
                except Exception as e:
                    log("[%s] TLS handshake with lpminer FAILED: %r (lpminer likely verifies the cert; try --listen-plain + stratum+tcp, or lpminer's insecure flag)" % (ts(), e), fh)
                    try: cli.close()
                    except Exception: pass
                    continue
            threading.Thread(target=handle, args=(cli, addr, args, fh), daemon=True).start()
    except KeyboardInterrupt:
        log("[%s] === proxy stop (ctrl-c) ===" % ts(), fh)

if __name__ == "__main__":
    main()
