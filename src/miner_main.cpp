// miner_main.cpp — unified, self-contained Pearl(PRL) PoUW miner.
//
// One binary, two modes:
//   pearl-miner --pool  ...   LuckyPool/kryptex stratum (the earning path)
//   pearl-miner --solo  ...   direct pearld JSON-RPC (getblocktemplate/submitblock)
//
// Mining is in-process via mine_plain_proof() (prover.h) — the proven PlainProof
// pipeline (real-config fused tensor-core search by default). Pool mode submits the
// base64 PlainProof verbatim (the pool builds the ZK proof). Solo mode turns the
// PlainProof into a full block via the `zkprove` helper (PlainProof -> plonky2 ZK
// proof -> assembled block) and submits it to the node.
//
// Networking is plain POSIX sockets + OpenSSL (pool is plaintext TCP; solo RPC is
// HTTPS with self-signed cert tolerance + HTTP basic auth). No other deps.
#include "prover.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <openssl/err.h>
#include <openssl/ssl.h>

// ===========================================================================
// small utilities
// ===========================================================================
static const char* B64="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string base64(const std::string& in) {
  std::string out;
  size_t i = 0;
  while (i + 2 < in.size()) {
    unsigned n = (unsigned char)in[i] << 16 | (unsigned char)in[i+1] << 8 | (unsigned char)in[i+2];
    out += B64[(n>>18)&63]; out += B64[(n>>12)&63]; out += B64[(n>>6)&63]; out += B64[n&63];
    i += 3;
  }
  if (i + 1 == in.size()) {
    unsigned n = (unsigned char)in[i] << 16;
    out += B64[(n>>18)&63]; out += B64[(n>>12)&63]; out += "==";
  } else if (i + 2 == in.size()) {
    unsigned n = (unsigned char)in[i] << 16 | (unsigned char)in[i+1] << 8;
    out += B64[(n>>18)&63]; out += B64[(n>>12)&63]; out += B64[(n>>6)&63]; out += "=";
  }
  return out;
}

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static void log_ts(const std::string& m) {
  fprintf(stderr, "[%.1f] %s\n", now_s(), m.c_str());
}
static std::string fmt2(double v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.2f", v);
  return std::string(buf);
}

// ---- minimal JSON field extraction (sufficient for stratum lines + small
// zkprove/RPC objects; the heavy template JSON is parsed by zkprove in Rust) ----
static bool json_str(const std::string& s, const std::string& key, std::string& out) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return false;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  p++;
  while (p < s.size() && (s[p]==' '||s[p]=='\t'||s[p]=='\n'||s[p]=='\r')) p++;
  if (p >= s.size() || s[p] != '"') return false;
  p++;
  std::string v;
  while (p < s.size() && s[p] != '"') {
    if (s[p] == '\\' && p + 1 < s.size()) { v += s[p+1]; p += 2; }
    else v += s[p++];
  }
  out = v;
  return true;
}
static bool json_int(const std::string& s, const std::string& key, long long& out) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return false;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  p++;
  while (p < s.size() && (s[p]==' '||s[p]=='\t')) p++;
  bool neg = false;
  if (p < s.size() && s[p]=='-') { neg = true; p++; }
  if (p >= s.size() || s[p] < '0' || s[p] > '9') return false;
  long long v = 0;
  while (p < s.size() && s[p] >= '0' && s[p] <= '9') { v = v*10 + (s[p]-'0'); p++; }
  out = neg ? -v : v;
  return true;
}
// Extract the raw JSON value (object/array/scalar) following "key": — used to
// lift the getblocktemplate `result` object out of the RPC envelope.
static bool json_value(const std::string& s, const std::string& key, std::string& out) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return false;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  p++;
  while (p < s.size() && (s[p]==' '||s[p]=='\t'||s[p]=='\n'||s[p]=='\r')) p++;
  if (p >= s.size()) return false;
  if (s[p] == '{' || s[p] == '[') {
    char open = s[p], close = (open=='{') ? '}' : ']';
    int depth = 0; bool instr = false; size_t start = p;
    for (; p < s.size(); p++) {
      char c = s[p];
      if (instr) { if (c=='\\') p++; else if (c=='"') instr=false; continue; }
      if (c=='"') instr = true;
      else if (c==open) depth++;
      else if (c==close) { depth--; if (depth==0) { out = s.substr(start, p-start+1); return true; } }
    }
    return false;
  }
  // scalar: read until , } ]
  size_t start = p;
  while (p < s.size() && s[p]!=','&&s[p]!='}'&&s[p]!=']') p++;
  out = s.substr(start, p-start);
  return true;
}

static bool write_file(const std::string& path, const std::string& data) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t w = fwrite(data.data(), 1, data.size(), f);
  fclose(f);
  return w == data.size();
}
// Run a command, capture stdout. Returns exit code (-1 on spawn failure).
static int run_capture(const std::string& cmd, std::string& out) {
  out.clear();
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return -1;
  char buf[8192];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
  int rc = pclose(p);
  if (rc == -1) return -1;
  // WEXITSTATUS
  return (rc & 0xff00) >> 8;
}

// ===========================================================================
// TCP / TLS
// ===========================================================================
static int tcp_connect(const std::string& host, int port) {
  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  char ports[16]; snprintf(ports, sizeof(ports), "%d", port);
  if (getaddrinfo(host.c_str(), ports, &hints, &res) != 0 || !res) {
    log_ts("DNS failed for " + host);
    return -1;
  }
  int fd = -1;
  for (auto* a = res; a; a = a->ai_next) {
    fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
    close(fd); fd = -1;
  }
  freeaddrinfo(res);
  if (fd >= 0) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }
  return fd;
}

// ===========================================================================
// POOL MODE — LuckyPool/kryptex stratum (replicates the captured lpminer contract)
// ===========================================================================
struct PoolOpts {
  std::string host = "prl.kryptex.network";
  int port = 7048;
  std::string wallet = "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv";
  std::string worker = "pm";
  std::string agent = "lpminer/0.1.9-552bdfe";
  long long hs = 8000000;
  uint64_t batch = 1000000;
  bool real_cfg = true;
  bool use_tc = true;       // fused tensor-core kernel (the only GPU path)
  bool breakdown = false;
  bool net_probe = false;   // connect+authorize+await one job, then exit (no mining)
};

struct PoolState {
  std::mutex mu;
  std::string header, target, job_id;
  long long height = 0;
  uint64_t gen = 0;          // bumped on every job change
  bool have_job = false;
  std::atomic<bool> stop{false};        // global stop (EOF / fatal)
  std::atomic<std::atomic<bool>*> active_stop{nullptr}; // current mining attempt's abort flag
  // submit responses by id
  std::unordered_set<long long> submit_ids;
  std::unordered_map<long long, std::string> submit_resp; // id -> raw line
};

static bool pool_send(int fd, const std::string& json) {
  std::string m = json + "\n";
  size_t sent = 0;
  while (sent < m.size()) {
    ssize_t n = send(fd, m.data() + sent, m.size() - sent, 0);
    if (n <= 0) return false;
    sent += (size_t)n;
  }
  return true;
}

static void pool_reader(int fd, PoolState* st) {
  std::string buf;
  char tmp[8192];
  while (!st->stop.load()) {
    fd_set fds; FD_ZERO(&fds); FD_SET(fd, &fds);
    struct timeval tv{0, 500000};
    int sr = select(fd + 1, &fds, nullptr, nullptr, &tv);
    if (sr < 0) { st->stop = true; break; }
    if (sr == 0) continue;
    ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) { log_ts("[pool] EOF/closed by pool"); st->stop = true; break; }
    buf.append(tmp, n);
    size_t nl;
    while ((nl = buf.find('\n')) != std::string::npos) {
      std::string line = buf.substr(0, nl);
      buf.erase(0, nl + 1);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      if (line.find("\"mining.notify\"") != std::string::npos) {
        std::string header, target, job_id; long long height = 0;
        json_str(line, "header", header);
        json_str(line, "target", target);
        json_str(line, "job_id", job_id);
        json_int(line, "height", height);
        std::atomic<bool>* to_abort = nullptr;
        {
          std::lock_guard<std::mutex> lk(st->mu);
          bool changed = !(st->have_job && st->job_id==job_id && st->header==header &&
                           st->target==target && st->height==height);
          st->header = header; st->target = target; st->job_id = job_id; st->height = height;
          st->have_job = true;
          if (changed) {
            st->gen++;
            to_abort = st->active_stop.load();
          }
          log_ts("[pool] notify job_id=" + job_id + " height=" + std::to_string(height) +
                 " gen=" + std::to_string(st->gen) + " target=" + target);
        }
        if (to_abort) to_abort->store(true);  // abort the stale mining attempt
      } else {
        long long id = 0;
        bool has_id = json_int(line, "id", id);
        {
          std::lock_guard<std::mutex> lk(st->mu);
          bool is_submit = has_id && st->submit_ids.count(id);
          if (is_submit) st->submit_resp[id] = line;
          log_ts(std::string("[pool] << ") + (is_submit ? "*** SUBMIT RESPONSE *** " : "(ack) ") +
                 line.substr(0, 400));
        }
      }
    }
  }
}

static int run_pool(const PoolOpts& o) {
  SSL_library_init();
  PoolState st;
  int fd = tcp_connect(o.host, o.port);
  if (fd < 0) { log_ts("pool connect failed"); return 2; }
  log_ts("[pool] connected " + o.host + ":" + std::to_string(o.port) + " (plaintext)");
  std::thread rd(pool_reader, fd, &st);

  // mining.authorize — EXACT captured shape (wallet="ADDR.WORKER", worker, agent).
  std::string auth = "{\"id\":1,\"method\":\"mining.authorize\",\"params\":{\"wallet\":\"" +
                     o.wallet + "." + o.worker + "\",\"worker\":\"" + o.worker +
                     "\",\"agent\":\"" + o.agent + "\"}}";
  pool_send(fd, auth);
  log_ts("[drv] >> authorize wallet=" + o.wallet + "." + o.worker + " agent=" + o.agent);

  // wait for first job (<=25s)
  double t0 = now_s();
  while (now_s() - t0 < 25) {
    { std::lock_guard<std::mutex> lk(st.mu); if (st.have_job) break; }
    if (st.stop.load()) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  { std::lock_guard<std::mutex> lk(st.mu);
    if (!st.have_job) { log_ts("[drv] no job within 25s; abort"); st.stop = true; rd.join(); close(fd); return 3; } }

  // --net-probe: prove the C++ stratum speaks the live contract (authorize + a
  // real job) without spending any mining time, then exit cleanly.
  if (o.net_probe) {
    std::lock_guard<std::mutex> lk(st.mu);
    log_ts("[probe] AUTHORIZED + JOB OK job_id=" + st.job_id + " height=" +
           std::to_string(st.height) + " target=" + st.target +
           " header=" + st.header.substr(0, 24) + "...");
    log_ts("[probe] C++ stratum live contract validated -> exit 0");
    st.stop = true; rd.join(); close(fd);
    return 0;
  }

  long long submit_id = 100;
  // Stable-lifetime abort flag. The reader thread holds &stop_attempt via
  // active_stop and may store(true) the instant a new job arrives — which can
  // race just past mine_plain_proof() returning. A per-iteration local could be
  // out of scope by then (UB); one flag for the whole run is always valid memory.
  // Reset before each attempt.
  std::atomic<bool> stop_attempt{false};
  uint64_t hr_draws_60s = 0;
  double hr_work_per_draw = 0.0;
  double hr_t0 = now_s();
  while (!st.stop.load()) {
    // snapshot the current job
    std::string hdr, tgt, jid; long long height; uint64_t cur_gen;
    { std::lock_guard<std::mutex> lk(st.mu);
      hdr = st.header; tgt = st.target; jid = st.job_id; height = st.height; cur_gen = st.gen; }
    if (hdr.empty() || tgt.empty() || jid.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      continue;
    }

    MineParams P;
    P.header_hex = hdr;
    P.target_hex = tgt;
    P.real_cfg = o.real_cfg;
    P.use_tc = o.use_tc;
    P.breakdown = o.breakdown;
    P.mine = true;
    P.maxdraws = o.batch;
    P.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    MineResult R;
    stop_attempt.store(false);
    st.active_stop.store(&stop_attempt);
    log_ts("[drv] mining gen=" + std::to_string(cur_gen) + " job_id=" + jid +
           " height=" + std::to_string(height));
    double bt = now_s();
    int rc = mine_plain_proof(P, R, &stop_attempt);
    st.active_stop.store(nullptr);
    if (R.draws > 0 && R.elapsed_s > 0.0 && R.work_per_draw > 0.0) {
      double ths = (double)R.draws * R.work_per_draw / R.elapsed_s / 1e12;
      log_ts("[drv] hashrate avg=" + fmt2(ths) +
             " TH/s (SRBMiner-MULTI/lpminer-compatible, " +
             std::to_string(R.draws) + " draws/" + fmt2(R.elapsed_s) + "s)");
      hr_draws_60s += R.draws;
      hr_work_per_draw = R.work_per_draw;
      double hr_dt = now_s() - hr_t0;
      if (hr_dt >= 60.0) {
        double ths60 = (double)hr_draws_60s * hr_work_per_draw / hr_dt / 1e12;
        log_ts("[drv] hashrate 60s=" + fmt2(ths60) +
               " TH/s (SRBMiner-MULTI display window, " +
               std::to_string(hr_draws_60s) + " draws/" + fmt2(hr_dt) + "s)");
        hr_draws_60s = 0;
        hr_t0 = now_s();
      }
    }

    // is this win still for the current job?
    bool fresh;
    { std::lock_guard<std::mutex> lk(st.mu);
      fresh = (st.gen == cur_gen && st.job_id==jid && st.header==hdr &&
               st.target==tgt && st.height==height); }

    if (rc == 0 && R.found && fresh) {
      long long this_id = submit_id++;
      { std::lock_guard<std::mutex> lk(st.mu); st.submit_ids.insert(this_id); }
      std::string sub = "{\"id\":" + std::to_string(this_id) +
                        ",\"method\":\"mining.submit\",\"params\":{\"job_id\":\"" + jid +
                        "\",\"plain_proof\":\"" + R.proof_b64 +
                        "\",\"hs\":" + std::to_string(o.hs) + "}}";
      pool_send(fd, sub);
      log_ts("[drv] *** WIN *** -> submit id=" + std::to_string(this_id) + " job_id=" + jid +
             " plain_proof=" + std::to_string(R.proof_b64.size()) + "B");
      double wt = now_s();
      std::string verdict;
      while (now_s() - wt < 30) {
        { std::lock_guard<std::mutex> lk(st.mu);
          auto it = st.submit_resp.find(this_id);
          if (it != st.submit_resp.end()) { verdict = it->second; break; } }
        if (st.stop.load()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
      }
      if (verdict.empty()) log_ts("[drv] no submit response within 30s (SILENT)");
      else log_ts("[drv] submit verdict id=" + std::to_string(this_id) + ": " + verdict.substr(0,300));
    } else if (rc == 0 && R.found && !fresh) {
      log_ts("[drv] DROP STALE WIN (job changed mid-mine)");
    } else {
      log_ts("[drv] gen=" + std::to_string(cur_gen) + " no win (rc=" + std::to_string(rc) +
             ") in " + std::to_string(now_s()-bt) + "s -> next job");
    }
  }
  st.stop = true;
  if (rd.joinable()) rd.join();
  close(fd);
  return 0;
}

// ===========================================================================
// SOLO MODE — pearld JSON-RPC over HTTPS (getblocktemplate / submitblock)
// ===========================================================================
struct SoloOpts {
  std::string host = "127.0.0.1";
  int port = 44107;             // pearld default RPC
  std::string user, pass;
  std::string addr;             // P2TR mining address
  std::string zkprove = "./zkprove";
  uint64_t batch = 1000000;
  bool real_cfg = true;
  bool use_tc = true;       // fused tensor-core kernel (the only GPU path)
  bool breakdown = false;
  int poll_s = 10;
};

// One-shot HTTPS POST of a JSON body; returns full response body (or "" on error).
static bool https_post(const SoloOpts& o, const std::string& body, std::string& resp_body) {
  resp_body.clear();
  int fd = tcp_connect(o.host, o.port);
  if (fd < 0) return false;
  SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
  if (!ctx) { close(fd); return false; }
  SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);  // pearld uses a self-signed cert
  SSL* ssl = SSL_new(ctx);
  SSL_set_fd(ssl, fd);
  SSL_set_tlsext_host_name(ssl, o.host.c_str());
  if (SSL_connect(ssl) != 1) {
    log_ts("[solo] TLS handshake failed to " + o.host + ":" + std::to_string(o.port));
    SSL_free(ssl); SSL_CTX_free(ctx); close(fd); return false;
  }
  std::string cred = base64(o.user + ":" + o.pass);
  std::string req = "POST / HTTP/1.1\r\n";
  req += "Host: " + o.host + ":" + std::to_string(o.port) + "\r\n";
  req += "Authorization: Basic " + cred + "\r\n";
  req += "Content-Type: application/json\r\n";
  req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  req += "Connection: close\r\n\r\n";
  req += body;
  if (SSL_write(ssl, req.data(), (int)req.size()) <= 0) {
    SSL_free(ssl); SSL_CTX_free(ctx); close(fd); return false;
  }
  std::string raw;
  char buf[8192];
  int n;
  while ((n = SSL_read(ssl, buf, sizeof(buf))) > 0) raw.append(buf, n);
  SSL_shutdown(ssl); SSL_free(ssl); SSL_CTX_free(ctx); close(fd);
  size_t hdr_end = raw.find("\r\n\r\n");
  if (hdr_end == std::string::npos) { resp_body = raw; return true; }
  resp_body = raw.substr(hdr_end + 4);
  return true;
}

static bool rpc_call(const SoloOpts& o, const std::string& method, const std::string& params_json,
                     std::string& result_out, std::string& error_out) {
  std::string body = "{\"jsonrpc\":\"1.0\",\"id\":\"pm\",\"method\":\"" + method +
                     "\",\"params\":" + params_json + "}";
  std::string resp;
  if (!https_post(o, body, resp)) { error_out = "transport"; return false; }
  // error: null on success
  std::string errval;
  if (json_value(resp, "error", errval)) {
    std::string trimmed; for (char c : errval) if (c!=' '&&c!='\t') trimmed += c;
    if (trimmed != "null" && !trimmed.empty()) { error_out = errval; }
  }
  json_value(resp, "result", result_out);
  return error_out.empty();
}

static int run_solo(const SoloOpts& o) {
  SSL_library_init();
  SSL_load_error_strings();
  if (o.addr.empty()) { log_ts("[solo] --addr <p2tr> required"); return 2; }
  if (o.user.empty()) { log_ts("[solo] --rpcuser required"); return 2; }
  log_ts("[solo] node=" + o.host + ":" + std::to_string(o.port) + " addr=" + o.addr);

  std::string tpl_path = "/tmp/pm_tpl.json";
  std::string pp_path  = "/tmp/pm_pp.b64";

  // shared "latest template" for the abort-on-new-block poller
  std::mutex mu;
  std::string cur_prevhash;
  long long cur_height = -1;
  std::atomic<bool> shutdown{false};
  std::atomic<std::atomic<bool>*> active_stop{nullptr};

  auto fetch_template = [&](std::string& result) -> bool {
    std::string err;
    if (!rpc_call(o, "getblocktemplate", "[{\"rules\":[\"segwit\"]}]", result, err)) {
      log_ts("[solo] getblocktemplate error: " + err);
      return false;
    }
    return !result.empty();
  };

  // poller: detect a new block (height/prevhash change) and abort current mining
  std::thread poller([&]() {
    while (!shutdown.load()) {
      for (int i = 0; i < o.poll_s * 2 && !shutdown.load(); i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      if (shutdown.load()) break;
      std::string tpl;
      if (!fetch_template(tpl)) continue;
      std::string ph; long long h = 0;
      json_str(tpl, "previousblockhash", ph);
      json_int(tpl, "height", h);
      std::atomic<bool>* to_abort = nullptr;
      { std::lock_guard<std::mutex> lk(mu);
        if (!ph.empty() && (ph != cur_prevhash || h != cur_height) && cur_height >= 0) {
          to_abort = active_stop.load();
        } }
      if (to_abort) { log_ts("[solo] new block detected -> abort current mining"); to_abort->store(true); }
    }
  });

  int rc_final = 0;
  // Stable-lifetime abort flag (the poller may store(true) just after
  // mine_plain_proof() returns; a per-iteration local would be UB). Reset per round.
  std::atomic<bool> stop_attempt{false};
  uint64_t hr_draws_60s = 0;
  double hr_work_per_draw = 0.0;
  double hr_t0 = now_s();
  while (!shutdown.load()) {
    std::string tpl;
    if (!fetch_template(tpl)) { std::this_thread::sleep_for(std::chrono::seconds(3)); continue; }
    std::string ph; long long h = 0;
    json_str(tpl, "previousblockhash", ph);
    json_int(tpl, "height", h);
    { std::lock_guard<std::mutex> lk(mu); cur_prevhash = ph; cur_height = h; }
    if (!write_file(tpl_path, tpl)) { log_ts("[solo] cannot write template file"); rc_final = 5; break; }

    // zkprove header -> incomplete_header + target
    std::string hjson;
    int zrc = run_capture(o.zkprove + " header --addr " + o.addr + " --tpl " + tpl_path + " 2>/dev/null", hjson);
    std::string header_hex, target_hex;
    if (zrc != 0 || !json_str(hjson, "incomplete_header", header_hex) ||
        !json_str(hjson, "target", target_hex)) {
      log_ts("[solo] zkprove header failed (rc=" + std::to_string(zrc) + "): " + hjson.substr(0,200));
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }
    log_ts("[solo] height=" + std::to_string(h) + " header=" + header_hex.substr(0,24) +
           "... target=" + target_hex.substr(0,16) + "...");

    MineParams P;
    P.header_hex = header_hex;
    P.target_hex = target_hex;
    P.real_cfg = o.real_cfg;
    P.use_tc = o.use_tc;
    P.breakdown = o.breakdown;
    P.mine = true;
    P.maxdraws = o.batch;
    P.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    MineResult R;
    stop_attempt.store(false);
    active_stop.store(&stop_attempt);
    int mrc = mine_plain_proof(P, R, &stop_attempt);
    active_stop.store(nullptr);
    if (R.draws > 0 && R.elapsed_s > 0.0 && R.work_per_draw > 0.0) {
      double ths = (double)R.draws * R.work_per_draw / R.elapsed_s / 1e12;
      log_ts("[solo] hashrate avg=" + fmt2(ths) +
             " TH/s (SRBMiner-MULTI/lpminer-compatible, " +
             std::to_string(R.draws) + " draws/" + fmt2(R.elapsed_s) + "s)");
      hr_draws_60s += R.draws;
      hr_work_per_draw = R.work_per_draw;
      double hr_dt = now_s() - hr_t0;
      if (hr_dt >= 60.0) {
        double ths60 = (double)hr_draws_60s * hr_work_per_draw / hr_dt / 1e12;
        log_ts("[solo] hashrate 60s=" + fmt2(ths60) +
               " TH/s (SRBMiner-MULTI display window, " +
               std::to_string(hr_draws_60s) + " draws/" + fmt2(hr_dt) + "s)");
        hr_draws_60s = 0;
        hr_t0 = now_s();
      }
    }

    if (mrc != 0 || !R.found) {
      log_ts("[solo] no win this round (rc=" + std::to_string(mrc) + ") -> refetch template");
      continue;
    }

    // assemble + submit
    if (!write_file(pp_path, R.proof_b64)) { log_ts("[solo] cannot write proof file"); continue; }
    log_ts("[solo] *** WIN *** -> zkprove block (plonky2 ZK proof + block assembly)…");
    std::string block_hex;
    int brc = run_capture(o.zkprove + " block --addr " + o.addr + " --tpl " + tpl_path +
                          " --ppfile " + pp_path, block_hex);
    // strip trailing whitespace/newline
    while (!block_hex.empty() && (block_hex.back()=='\n'||block_hex.back()=='\r'||block_hex.back()==' '))
      block_hex.pop_back();
    if (brc != 0 || block_hex.empty()) {
      log_ts("[solo] zkprove block failed (rc=" + std::to_string(brc) + ")");
      continue;
    }
    log_ts("[solo] submitting block (" + std::to_string(block_hex.size()/2) + " bytes)…");
    std::string result, err;
    rpc_call(o, "submitblock", "[\"" + block_hex + "\"]", result, err);
    // submitblock returns null on success, else an error string/reason
    std::string r2; for (char c : result) if (c!=' '&&c!='\t'&&c!='"') r2 += c;
    if (err.empty() && (r2 == "null" || r2.empty()))
      log_ts("[solo] *** BLOCK ACCEPTED *** height=" + std::to_string(h));
    else
      log_ts("[solo] submitblock rejected: result=" + result + " error=" + err);
  }
  shutdown = true;
  if (poller.joinable()) poller.join();
  return rc_final;
}

// ===========================================================================
// main
// ===========================================================================
static void usage() {
  fprintf(stderr,
    "pearl-miner — self-built Pearl(PRL) PoUW miner\n\n"
    "  pearl-miner --pool [opts]     mine to LuckyPool/kryptex stratum\n"
    "  pearl-miner --solo [opts]     mine directly against a pearld node\n\n"
    "common:  --cfg real|golden (default real)  --tc (tensor-core, default)  --batch N  --breakdown\n"
    "pool:    --pool-host H  --pool-port P  --wallet ADDR  --worker W  --agent A  --hs N\n"
    "solo:    --node host:port  --rpcuser U  --rpcpass P  --addr <p2tr>  --zkprove PATH\n");
}

int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 2; }
  bool pool = false, solo = false;
  PoolOpts po; SoloOpts so;
  std::string cfg = "real";
  bool use_tc = true;
  bool breakdown = false;
  uint64_t batch = 1000000;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* def="") -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
    };
    if (a == "--pool") pool = true;
    else if (a == "--solo") solo = true;
    else if (a == "--cfg") cfg = next("real");
    else if (a == "--tc") use_tc = true;   // accepted for back-compat; tensor-core is the default
    else if (a == "--breakdown") breakdown = true;
    else if (a == "--batch") batch = strtoull(next("1000000").c_str(), nullptr, 10);
    // pool
    else if (a == "--pool-host") po.host = next();
    else if (a == "--pool-port") po.port = atoi(next().c_str());
    else if (a == "--wallet") po.wallet = next();
    else if (a == "--worker") po.worker = next();
    else if (a == "--agent") po.agent = next();
    else if (a == "--hs") po.hs = atoll(next().c_str());
    else if (a == "--net-probe") po.net_probe = true;
    // solo
    else if (a == "--node") {
      std::string hp = next();  // host:port
      size_t c = hp.rfind(':');
      if (c != std::string::npos) { so.host = hp.substr(0,c); so.port = atoi(hp.substr(c+1).c_str()); }
      else so.host = hp;
    }
    else if (a == "--rpcuser") so.user = next();
    else if (a == "--rpcpass") so.pass = next();
    else if (a == "--addr") so.addr = next();
    else if (a == "--zkprove") so.zkprove = next();
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(); return 2; }
  }

  bool real_cfg = (cfg == "real");

  if (pool == solo) { fprintf(stderr, "choose exactly one of --pool / --solo\n"); usage(); return 2; }
  if (pool) {
    po.batch = batch; po.real_cfg = real_cfg; po.use_tc = use_tc; po.breakdown = breakdown;
    return run_pool(po);
  } else {
    so.batch = batch; so.real_cfg = real_cfg; so.use_tc = use_tc; so.breakdown = breakdown;
    return run_solo(so);
  }
}
