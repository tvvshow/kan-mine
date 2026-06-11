// miner_main.cpp — unified Pearl(PRL) PoUW miner (lpminer-compatible logging)
#include "prover.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fcntl.h>
#include <termios.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <cuda_runtime.h>

extern "C" { int g_miner_verbose = 0; }  // gate per-draw prints in plainproof_gen/tc_cutlass

// ===========================================================================
// lpminer-style logger
// ===========================================================================
static void log_line(const char* cat, const char* msg) {
  time_t t = time(nullptr);
  struct tm* lt = localtime(&t);
  fprintf(stderr, "%02d:%02d:%02d  info   %-14s %s\n",
          lt->tm_hour, lt->tm_min, lt->tm_sec, cat, msg);
}
static void log_linef(const char* cat, const char* fmt, ...) __attribute__((format(printf,2,3)));
static void log_linef(const char* cat, const char* fmt, ...) {
  char buf[2048];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  log_line(cat, buf);
}

// ===========================================================================
// NVML (via dlopen for optional HW monitoring)
// ===========================================================================
struct NVML {
  void* lib = nullptr;
  int (*Init)() = nullptr;
  int (*DeviceGetCount)(unsigned*) = nullptr;
  int (*DeviceGetHandleByIndex)(unsigned, void**) = nullptr;
  int (*DeviceGetName)(void*, char*, unsigned) = nullptr;
  int (*DeviceGetMemoryInfo)(void*, void*) = nullptr;
  int (*DeviceGetTemperature)(void*, int, unsigned*) = nullptr;
  int (*DeviceGetFanSpeed)(void*, unsigned*) = nullptr;
  int (*DeviceGetPowerUsage)(void*, unsigned*) = nullptr;
  int (*DeviceGetCudaComputeCapability)(void*, int*, int*) = nullptr;
  int (*DeviceGetPciInfo)(void*, void*) = nullptr;
  bool ok = false;
  bool init() {
    lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!lib) return false;
#define LD(name) name = (decltype(name))dlsym(lib, "nvml" #name)
    LD(Init); LD(DeviceGetCount); LD(DeviceGetHandleByIndex); LD(DeviceGetName);
    LD(DeviceGetMemoryInfo); LD(DeviceGetTemperature); LD(DeviceGetFanSpeed);
    LD(DeviceGetPowerUsage); LD(DeviceGetCudaComputeCapability); LD(DeviceGetPciInfo);
#undef LD
    ok = (Init && Init() == 0);
    return ok;
  }
  ~NVML() { if (lib) dlclose(lib); }
};
struct GPUInfo {
  void* handle = nullptr;
  char name[96] = "GPU";
  unsigned long long vram_mb = 0;
  int sm_maj = 0, sm_min = 0;
  int bus = 0;
};
static NVML g_nvml;
static std::vector<GPUInfo> g_gpus;

// ===========================================================================
// stats tracking
// ===========================================================================
static std::atomic<uint64_t> g_accepted{0}, g_rejected{0}, g_total_draws{0};
static std::atomic<double> g_work_per_draw{0.0};
static std::chrono::steady_clock::time_point g_start = std::chrono::steady_clock::now();
static std::chrono::steady_clock::time_point g_last_share = g_start;
static std::mutex g_stat_mu;
struct DrawTick { std::chrono::steady_clock::time_point t; uint64_t draws; };
static std::deque<DrawTick> g_ticks;
static void tick_draw(uint64_t draws) {
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(g_stat_mu);
  g_ticks.push_back({now, draws});
  while (!g_ticks.empty()) {
    auto dt = std::chrono::duration<double>(now - g_ticks.front().t).count();
    if (dt < 900.0) break;
    g_ticks.pop_front();
  }
}
static double window_ths(double win_s) {
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(g_stat_mu);
  if (g_ticks.empty()) return 0.0;
  uint64_t d0 = g_ticks.front().draws, d1 = g_ticks.back().draws;
  double t0 = std::chrono::duration<double>(now - g_ticks.front().t).count();
  if (t0 < 1.0) return 0.0;
  double wpd = g_work_per_draw.load();
  if (wpd == 0.0) return 0.0;
  return (double)(d1 - d0) * wpd / t0 / 1e12;
}
static void print_stats(const std::string& wallet, const std::string& pool_url) {
  auto now = std::chrono::steady_clock::now();
  double up_s = std::chrono::duration<double>(now - g_start).count();
  int up_d = (int)(up_s / 86400.0);
  int up_h = (int)((up_s - up_d*86400) / 3600.0);
  int up_m = (int)((up_s - up_d*86400 - up_h*3600) / 60.0);
  int up_sec = (int)(up_s - up_d*86400 - up_h*3600 - up_m*60);
  double last_s = std::chrono::duration<double>(now - g_last_share).count();
  int last_m = (int)(last_s / 60.0);
  uint64_t acc = g_accepted.load(), rej = g_rejected.load();
  double ths_10 = window_ths(10.0), ths_60 = window_ths(60.0), ths_15m = window_ths(900.0);
  char wshort[32]; snprintf(wshort, sizeof(wshort), "%.10s...%.4s", wallet.c_str(), wallet.c_str()+wallet.size()-4);
  fprintf(stderr, "\n-----%s---------------------%s-----\n", wshort, pool_url.c_str());
  fprintf(stderr, " DEVICE MODEL              HASHRATE  TEMP  FAN POWER      EFFIC       A    R  LAST\n");
  fprintf(stderr, "----------------------------------------------------------------------------------------\n");
  if (g_gpus.empty()) {
    fprintf(stderr, " GPU #0 %-18s %.2f TH/s    --   --   ---  -------- %7llu %4llu %5s\n",
            "N/A", ths_60, (unsigned long long)acc, (unsigned long long)rej,
            last_m > 0 ? (std::to_string(last_m)+"m").c_str() : "-");
  }
  for (size_t i = 0; i < g_gpus.size(); i++) {
    unsigned temp = 0, fan = 0, power_mw = 0;
    if (g_nvml.ok && g_gpus[i].handle) {
      g_nvml.DeviceGetTemperature(g_gpus[i].handle, 0, &temp);
      g_nvml.DeviceGetFanSpeed(g_gpus[i].handle, &fan);
      g_nvml.DeviceGetPowerUsage(g_gpus[i].handle, &power_mw);
    }
    double power_w = (double)power_mw / 1000.0;
    double effic = (ths_60 > 0 && power_w > 1.0) ? ths_60 * 1000.0 / power_w : 0.0;
    fprintf(stderr, " GPU #%zu %-18s %.2f TH/s   %3uC  %3u%%  %3.0fW  %5.1f GH/W %7llu %4llu %5s\n",
            i, g_gpus[i].name, ths_60, temp, fan, power_w, effic,
            (unsigned long long)acc, (unsigned long long)rej,
            last_m > 0 ? (std::to_string(last_m)+"m").c_str() : "-");
  }
  fprintf(stderr, "----------------------------------------------------------------------------------------\n");
  fprintf(stderr, " 10s                   %10.2f TH/s            %5.0fW           A: %llu\n", ths_10, 0.0, (unsigned long long)acc);
  fprintf(stderr, " 60s                   %10.2f TH/s                           R: %llu\n", ths_60, (unsigned long long)rej);
  fprintf(stderr, " 15m                   %10.2f TH/s                           S: 0\n", ths_15m);
  double acc_pct = (acc + rej > 0) ? 100.0 * (double)acc / (double)(acc + rej) : 0.0;
  fprintf(stderr, "[%d days %02d:%02d:%02d]-------------------------------------[%.1f%% accept - ver. self-built]\n",
          up_d, up_h, up_m, up_sec, acc_pct);
  fprintf(stderr, "\n");
}

// ===========================================================================
// utilities
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
static int run_capture(const std::string& cmd, std::string& out) {
  out.clear();
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return -1;
  char buf[8192];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
  int rc = pclose(p);
  if (rc == -1) return -1;
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
  if (getaddrinfo(host.c_str(), ports, &hints, &res) != 0 || !res) return -1;
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
// POOL MODE
// ===========================================================================
struct PoolOpts {
  std::string host = "prl.kryptex.network";
  int port = 7048;
  bool use_tls = false;
  std::string wallet = "prl1patz2mw7d28lqn33a768huhhsz3rg6e228m22wxh0v4pjh53x4qwsg2apmv";
  std::string worker = "pm";
  std::string agent = "lpminer/0.1.9-552bdfe";
  uint64_t batch = 1000;
  bool real_cfg = true;
  bool use_tc = true;
  bool breakdown = false;
};
struct PoolState {
  std::mutex mu;
  std::string header, target, job_id;
  long long height = 0;
  uint64_t gen = 0, seq = 0;
  bool have_job = false;
  std::atomic<bool> stop{false};
  std::atomic<std::atomic<bool>*> active_stop{nullptr};
  std::unordered_set<long long> submit_ids;
  std::unordered_map<long long, std::string> submit_resp;
};
static bool pool_send(int fd, SSL* ssl, const std::string& json) {
  std::string m = json + "\n";
  size_t sent = 0;
  while (sent < m.size()) {
    ssize_t n = ssl ? SSL_write(ssl, m.data()+sent, (int)(m.size()-sent))
                    : send(fd, m.data()+sent, m.size()-sent, 0);
    if (n <= 0) return false;
    sent += (size_t)n;
  }
  return true;
}
static void pool_reader(int fd, SSL* ssl, PoolState* st) {
  std::string buf;
  char tmp[8192];
  while (!st->stop.load()) {
    fd_set fds; FD_ZERO(&fds); FD_SET(fd, &fds);
    struct timeval tv{0, 500000};
    int sr = select(fd + 1, &fds, nullptr, nullptr, &tv);
    if (sr < 0) { st->stop = true; break; }
    if (sr == 0) continue;
    ssize_t n = ssl ? SSL_read(ssl, tmp, sizeof(tmp)) : recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) { st->stop = true; break; }
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
        size_t us = job_id.rfind('_');
        long long diff = (us != std::string::npos) ? atoll(job_id.substr(us+1).c_str()) : 0;
        std::atomic<bool>* to_abort = nullptr;
        {
          std::lock_guard<std::mutex> lk(st->mu);
          bool changed = !(st->have_job && st->job_id==job_id && st->header==header &&
                           st->target==target && st->height==height);
          st->header = header; st->target = target; st->job_id = job_id; st->height = height;
          st->have_job = true;
          if (changed) {
            st->gen++;
            st->seq++;
            to_abort = st->active_stop.load();
          }
        }
        log_linef("stratum", "new job id=%s height=%lld diff=%lld seq=%llu",
                  job_id.c_str(), height, diff, (unsigned long long)st->seq);
        if (to_abort) to_abort->store(true);
      } else {
        long long id = 0;
        bool has_id = json_int(line, "id", id);
        {
          std::lock_guard<std::mutex> lk(st->mu);
          bool is_submit = has_id && st->submit_ids.count(id);
          if (is_submit) st->submit_resp[id] = line;
        }
      }
    }
  }
}
static int run_pool(const PoolOpts& o) {
  SSL_library_init();
  int fd = tcp_connect(o.host, o.port);
  if (fd < 0) { log_line("stratum", "connect failed"); return 2; }
  SSL_CTX* ctx = nullptr;
  SSL* ssl = nullptr;
  if (o.use_tls) {
    ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) { close(fd); return 2; }
    SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
    ssl = SSL_new(ctx);
    SSL_set_fd(ssl, fd);
    SSL_set_tlsext_host_name(ssl, o.host.c_str());
    if (SSL_connect(ssl) != 1) {
      log_line("stratum", "TLS handshake failed");
      SSL_free(ssl); SSL_CTX_free(ctx); close(fd); return 2;
    }
  }
  std::string proto = o.use_tls ? "stratum+tls://" : "stratum+tcp://";
  log_linef("stratum", "connecting to %s%s:%d as %s.%s (1 GPUs, one session)",
            proto.c_str(), o.host.c_str(), o.port, o.wallet.c_str(), o.worker.c_str());
  PoolState st;
  std::thread rd(pool_reader, fd, ssl, &st);
  std::string auth = "{\"id\":1,\"method\":\"mining.authorize\",\"params\":{\"wallet\":\"" +
                     o.wallet + "." + o.worker + "\",\"worker\":\"" + o.worker +
                     "\",\"agent\":\"" + o.agent + "\"}}";
  pool_send(fd, ssl, auth);
  double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0 < 25) {
    { std::lock_guard<std::mutex> lk(st.mu); if (st.have_job) break; }
    if (st.stop.load()) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  { std::lock_guard<std::mutex> lk(st.mu);
    if (!st.have_job) { log_line("stratum", "no job within 25s"); st.stop = true; rd.join();
      if (ssl) { SSL_shutdown(ssl); SSL_free(ssl); }
      if (ctx) SSL_CTX_free(ctx);
      close(fd); return 3; }
    log_linef("stratum", "authorize: ok wallet=%s.%s agent=%s",
              o.wallet.c_str(), o.worker.c_str(), o.agent.c_str());
  }
  long long submit_id = 100;
  std::atomic<bool> stop_attempt{false};
  std::atomic<bool> stats_req{false};
  std::thread stats_th([&]() {
    auto last_print = std::chrono::steady_clock::now();
    while (!st.stop.load()) {
      auto now = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(now - last_print).count();
      if (stats_req.load() || dt >= 120.0) {
        print_stats(o.wallet, proto + o.host + ":" + std::to_string(o.port));
        last_print = now;
        stats_req.store(false);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  });
  while (!st.stop.load()) {
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
    int rc = mine_plain_proof(P, R, &stop_attempt);
    st.active_stop.store(nullptr);

    if (R.draws > 0 && R.work_per_draw > 0.0) {
      g_total_draws.fetch_add(R.draws);
      g_work_per_draw.store(R.work_per_draw);
      tick_draw(g_total_draws.load());
    }

    bool fresh;
    { std::lock_guard<std::mutex> lk(st.mu);
      fresh = (st.gen == cur_gen && st.job_id==jid && st.header==hdr &&
               st.target==tgt && st.height==height); }

    if (rc == 0 && R.found && fresh) {
      long long this_id = submit_id++;
      { std::lock_guard<std::mutex> lk(st.mu); st.submit_ids.insert(this_id); }
      long long hs_actual = (long long)(R.draws * R.work_per_draw / R.elapsed_s);
      std::string sub = "{\"id\":" + std::to_string(this_id) +
                        ",\"method\":\"mining.submit\",\"params\":{\"job_id\":\"" + jid +
                        "\",\"plain_proof\":\"" + R.proof_b64 +
                        "\",\"hs\":" + std::to_string(hs_actual) + "}}";
      pool_send(fd, ssl, sub);
      double wt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      std::string verdict;
      while (std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - wt < 30) {
        { std::lock_guard<std::mutex> lk(st.mu);
          auto it = st.submit_resp.find(this_id);
          if (it != st.submit_resp.end()) { verdict = it->second; break; } }
        if (st.stop.load()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
      }
      if (!verdict.empty() && verdict.find("\"result\":true") != std::string::npos) {
        g_accepted.fetch_add(1);
        g_last_share = std::chrono::steady_clock::now();
        log_line("GPU #0", "share accepted");
      } else if (!verdict.empty()) {
        g_rejected.fetch_add(1);
        log_linef("GPU #0", "share rejected: %s", verdict.substr(0, 100).c_str());
      }
    }
    // Continue mining regardless of result (no win, stale, or error - keep going)
  }
  st.stop = true;
  if (stats_th.joinable()) stats_th.join();
  if (rd.joinable()) rd.join();
  if (ssl) { SSL_shutdown(ssl); SSL_free(ssl); }
  if (ctx) SSL_CTX_free(ctx);
  close(fd);
  return 0;
}

// ===========================================================================
// SOLO MODE
// ===========================================================================
struct SoloOpts {
  std::string host = "127.0.0.1";
  int port = 44107;
  std::string user, pass;
  std::string addr;
  std::string zkprove = "./zkprove";
  uint64_t batch = 1000;
  bool real_cfg = true;
  bool use_tc = true;
  bool breakdown = false;
  int poll_s = 10;
};
static bool https_post(const SoloOpts& o, const std::string& body, std::string& resp_body) {
  resp_body.clear();
  int fd = tcp_connect(o.host, o.port);
  if (fd < 0) return false;
  SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
  if (!ctx) { close(fd); return false; }
  SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
  SSL* ssl = SSL_new(ctx);
  SSL_set_fd(ssl, fd);
  SSL_set_tlsext_host_name(ssl, o.host.c_str());
  if (SSL_connect(ssl) != 1) {
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
  if (o.addr.empty()) { log_line("solo", "--addr required"); return 2; }
  if (o.user.empty()) { log_line("solo", "--rpcuser required"); return 2; }
  std::string tpl_path = "/tmp/pm_tpl.json";
  std::string pp_path  = "/tmp/pm_pp.b64";
  std::mutex mu;
  std::string cur_prevhash;
  long long cur_height = -1;
  std::atomic<bool> shutdown{false};
  std::atomic<std::atomic<bool>*> active_stop{nullptr};
  auto fetch_template = [&](std::string& result) -> bool {
    std::string err;
    if (!rpc_call(o, "getblocktemplate", "[{\"rules\":[\"segwit\"]}]", result, err)) {
      log_linef("solo", "getblocktemplate error: %s", err.c_str());
      return false;
    }
    return !result.empty();
  };
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
      if (to_abort) { log_line("solo", "new block detected"); to_abort->store(true); }
    }
  });
  int rc_final = 0;
  std::atomic<bool> stop_attempt{false};
  while (!shutdown.load()) {
    std::string tpl;
    if (!fetch_template(tpl)) { std::this_thread::sleep_for(std::chrono::seconds(3)); continue; }
    std::string ph; long long h = 0;
    json_str(tpl, "previousblockhash", ph);
    json_int(tpl, "height", h);
    { std::lock_guard<std::mutex> lk(mu); cur_prevhash = ph; cur_height = h; }
    if (!write_file(tpl_path, tpl)) { log_line("solo", "cannot write template file"); rc_final = 5; break; }
    std::string hjson;
    int zrc = run_capture(o.zkprove + " header --addr " + o.addr + " --tpl " + tpl_path + " 2>/dev/null", hjson);
    std::string header_hex, target_hex;
    if (zrc != 0 || !json_str(hjson, "incomplete_header", header_hex) ||
        !json_str(hjson, "target", target_hex)) {
      log_linef("solo", "zkprove header failed (rc=%d)", zrc);
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }
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
    if (R.draws > 0 && R.work_per_draw > 0.0) {
      g_total_draws.fetch_add(R.draws);
      g_work_per_draw.store(R.work_per_draw);
      tick_draw(g_total_draws.load());
    }
    if (mrc != 0 || !R.found) continue;
    if (!write_file(pp_path, R.proof_b64)) { log_line("solo", "cannot write proof file"); continue; }
    log_line("solo", "WIN -> zkprove block (plonky2 ZK proof + block assembly)");
    std::string block_hex;
    int brc = run_capture(o.zkprove + " block --addr " + o.addr + " --tpl " + tpl_path +
                          " --ppfile " + pp_path, block_hex);
    while (!block_hex.empty() && (block_hex.back()=='\n'||block_hex.back()=='\r'||block_hex.back()==' '))
      block_hex.pop_back();
    if (brc != 0 || block_hex.empty()) {
      log_linef("solo", "zkprove block failed (rc=%d)", brc);
      continue;
    }
    std::string result, err;
    rpc_call(o, "submitblock", "[\"" + block_hex + "\"]", result, err);
    std::string r2; for (char c : result) if (c!=' '&&c!='\t'&&c!='"') r2 += c;
    if (err.empty() && (r2 == "null" || r2.empty()))
      log_linef("solo", "BLOCK ACCEPTED height=%lld", h);
    else
      log_linef("solo", "submitblock rejected: %s", result.c_str());
  }
  shutdown = true;
  if (poller.joinable()) poller.join();
  return rc_final;
}

// ===========================================================================
// main
// ===========================================================================
static void banner() {
  log_line("about", "pearl-miner/self-built");
  FILE* f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[256], model[128] = "CPU";
    int cores = 0;
    while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "model name", 10) == 0) {
        char* p = strchr(line, ':');
        if (p) {
          p++; while (*p == ' ' || *p == '\t') p++;
          char* e = strchr(p, '\n'); if (e) *e = 0;
          snprintf(model, sizeof(model), "%s", p);
        }
      } else if (strncmp(line, "processor", 9) == 0) cores++;
    }
    fclose(f);
    log_linef("cpu", "%s (%d threads)", model, cores);
  } else {
    log_line("cpu", "N/A");
  }
  log_line("algo", "pearl");
}
static void usage() {
  fprintf(stderr,
    "pearl-miner — self-built Pearl(PRL) PoUW miner (lpminer-compatible)\n\n"
    "  pearl-miner --algo pearl --pool URL --wallet ADDR[.WORKER]\n"
    "  pearl-miner --solo --node host:port --rpcuser U --rpcpass P --addr <p2tr>\n\n"
    "pool URL:  stratum+ssl://host:port  or  stratum+tcp://host:port\n"
    "commands:  s (stats now), q (quit); table prints every 120s\n");
}
int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 2; }
  bool pool = false, solo = false;
  PoolOpts po; SoloOpts so;
  std::string pool_url;
  std::string algo;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* def="") -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
    };
    if (a == "--algo") algo = next("pearl");
    else if (a == "--pool") { pool_url = next(); pool = true; }
    else if (a == "--wallet") {
      std::string w = next();
      size_t dot = w.rfind('.');
      if (dot != std::string::npos && dot > 10) {
        po.wallet = w.substr(0, dot);
        po.worker = w.substr(dot + 1);
      } else {
        po.wallet = w;
      }
    }
    else if (a == "--worker") po.worker = next();
    else if (a == "--agent") po.agent = next();
    else if (a == "--hs") po.hs = atoll(next().c_str());
    else if (a == "--batch") po.batch = so.batch = strtoull(next("1000").c_str(), nullptr, 10);
    else if (a == "--cfg") { std::string c = next("real"); po.real_cfg = so.real_cfg = (c == "real"); }
    else if (a == "--tc") po.use_tc = so.use_tc = true;
    else if (a == "--breakdown") po.breakdown = so.breakdown = true;
    else if (a == "--solo") solo = true;
    else if (a == "--node") {
      std::string hp = next();
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
  if (!algo.empty() && algo != "pearl") {
    fprintf(stderr, "only --algo pearl is supported\n");
    return 2;
  }
  if (pool == solo) { fprintf(stderr, "choose exactly one of --pool / --solo\n"); usage(); return 2; }
  if (pool) {
    if (!pool_url.empty()) {
      if (pool_url.rfind("stratum+ssl://", 0) == 0) {
        po.use_tls = true;
        std::string hp = pool_url.substr(14);
        size_t c = hp.rfind(':');
        if (c != std::string::npos) {
          po.host = hp.substr(0, c);
          po.port = atoi(hp.substr(c+1).c_str());
        } else {
          po.host = hp;
          po.port = 8048;
        }
      } else if (pool_url.rfind("stratum+tcp://", 0) == 0) {
        po.use_tls = false;
        std::string hp = pool_url.substr(14);
        size_t c = hp.rfind(':');
        if (c != std::string::npos) {
          po.host = hp.substr(0, c);
          po.port = atoi(hp.substr(c+1).c_str());
        } else {
          po.host = hp;
          po.port = 7048;
        }
      }
    }
    banner();
    std::string proto = po.use_tls ? "stratum+tls://" : "stratum+tcp://";
    log_linef("pool", "%s%s:%d", proto.c_str(), po.host.c_str(), po.port);
    log_linef("wallet", "%s.%s", po.wallet.c_str(), po.worker.c_str());
    log_linef("worker", "%s", po.worker.c_str());
    log_line("commands", "s (stats), q (quit); table every 120s");
    g_nvml.init();
    if (g_nvml.ok) {
      unsigned cnt = 0;
      g_nvml.DeviceGetCount(&cnt);
      int drv_maj = 0, drv_min = 0;
      if (cudaDriverGetVersion) {
        int v = 0;
        cudaDriverGetVersion(&v);
        drv_maj = v / 1000;
        drv_min = (v % 1000) / 10;
      }
      log_linef("detected", "%u devices - driver %d.%d", cnt, drv_maj, drv_min);
      for (unsigned i = 0; i < cnt && i < 16; i++) {
        GPUInfo gi;
        g_nvml.DeviceGetHandleByIndex(i, &gi.handle);
        if (gi.handle) {
          g_nvml.DeviceGetName(gi.handle, gi.name, sizeof(gi.name));
          struct { unsigned long long free, total, used; } mem{};
          g_nvml.DeviceGetMemoryInfo(gi.handle, &mem);
          gi.vram_mb = mem.total / (1024*1024);
          g_nvml.DeviceGetCudaComputeCapability(gi.handle, &gi.sm_maj, &gi.sm_min);
          struct { char busId[16]; unsigned domain, bus, device; } pci{};
          g_nvml.DeviceGetPciInfo(gi.handle, &pci);
          gi.bus = pci.bus;
          log_linef("GPU", "#%u %-16s %lluGB sm_%d%d bus:%02x enabled",
                    i, gi.name, gi.vram_mb/1024, gi.sm_maj, gi.sm_min, gi.bus);
        }
        g_gpus.push_back(gi);
      }
    } else {
      log_line("detected", "1 devices - driver N/A");
      log_line("GPU", "#0 GPU            N/A sm_xx bus:00 enabled");
      g_gpus.push_back({});
    }
    log_line("devfee", "0%");
    return run_pool(po);
  } else {
    banner();
    return run_solo(so);
  }
}

