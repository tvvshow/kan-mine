// miner_main.cpp — Kan: Pearl(PRL) PoUW miner
#include "prover.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdarg>
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
#include <utility>
#include <vector>
#include <fcntl.h>
#include <termios.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <sys/stat.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <cuda_runtime.h>

#ifndef KAN_VERSION
#define KAN_VERSION "dev"
#endif

extern "C" { int g_miner_verbose = 0; }  // gate per-draw prints in plainproof_gen/tc_cutlass
extern std::atomic<uint64_t> g_live_draw_count;  // incremented per-draw in plainproof_gen
extern double g_work_per_draw_export;  // set by plainproof_gen before draw loop starts

// ===========================================================================
// lpminer-compatible logger
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
  unsigned index = 0;
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
struct Sample { std::chrono::steady_clock::time_point t; uint64_t draws; };
static std::deque<Sample> g_samples;
static void sample_draws() {
  auto now = std::chrono::steady_clock::now();
  uint64_t d = g_live_draw_count.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> lk(g_stat_mu);
  g_samples.push_back({now, d});
  while (g_samples.size() > 960) g_samples.pop_front();
}
static double window_ths(double win_s) {
  auto now = std::chrono::steady_clock::now();
  auto cutoff = now - std::chrono::duration<double>(win_s);
  std::lock_guard<std::mutex> lk(g_stat_mu);
  if (g_samples.size() < 2) return 0.0;
  const Sample* s0 = nullptr;
  for (auto& s : g_samples) {
    if (s.t >= cutoff) { s0 = &s; break; }
  }
  if (!s0) s0 = &g_samples.front();
  const Sample& s1 = g_samples.back();
  double dt = std::chrono::duration<double>(now - s0->t).count();
  if (dt < 0.5) return 0.0;
  double wpd = g_work_per_draw_export > 0.0 ? g_work_per_draw_export : g_work_per_draw.load();
  if (wpd == 0.0) return 0.0;
  return (double)(s1.draws - s0->draws) * wpd / dt / 1e12;
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
  char wshort[32];
  if (wallet.size() > 14) {
    snprintf(wshort, sizeof(wshort), "%.10s...%.4s", wallet.c_str(),
             wallet.c_str() + wallet.size() - 4);
  } else {
    snprintf(wshort, sizeof(wshort), "%s", wallet.empty() ? "<wallet>" : wallet.c_str());
  }
  const std::string last_share = last_m > 0 ? std::to_string(last_m) + "m" : "-";
  fprintf(stderr, "\n-----%s---------------------%s-----\n", wshort, pool_url.c_str());
  fprintf(stderr, " DEVICE MODEL              HASHRATE  TEMP  FAN POWER      EFFIC       A    R  LAST\n");
  fprintf(stderr, "----------------------------------------------------------------------------------------\n");
  if (g_gpus.empty()) {
    fprintf(stderr, " GPU #0 %-18s %.2f TH/s    --   --   ---  -------- %7llu %4llu %5s\n",
            "N/A", ths_60, (unsigned long long)acc, (unsigned long long)rej,
            last_share.c_str());
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
    fprintf(stderr, " GPU #%u %-18s %.2f TH/s   %3uC  %3u%%  %3.0fW  %5.1f GH/W %7llu %4llu %5s\n",
            g_gpus[i].index, g_gpus[i].name, ths_60, temp, fan, power_w, effic,
            (unsigned long long)acc, (unsigned long long)rej,
            last_share.c_str());
  }
  fprintf(stderr, "----------------------------------------------------------------------------------------\n");
  fprintf(stderr, " 10s                   %10.2f TH/s            %5.0fW           A: %llu\n", ths_10, 0.0, (unsigned long long)acc);
  fprintf(stderr, " 60s                   %10.2f TH/s                           R: %llu\n", ths_60, (unsigned long long)rej);
  fprintf(stderr, " 15m                   %10.2f TH/s                           S: 0\n", ths_15m);
  double acc_pct = (acc + rej > 0) ? 100.0 * (double)acc / (double)(acc + rej) : 0.0;
  fprintf(stderr, "[%d days %02d:%02d:%02d]-------------------------------------[%.1f%% accept - ver. %s]\n",
          up_d, up_h, up_m, up_sec, acc_pct, KAN_VERSION);
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
struct PoolEndpoint {
  std::string host;
  int port = 0;
  bool use_tls = false;
};
struct PoolOpts {
  std::string host = "prl.kryptex.network";
  int port = 7048;
  bool use_tls = false;
  // Primary first, then failover backups (filled from one or more --pool URLs).
  // run_pool tries them in order and mines the first that connects+authorizes;
  // host/port/use_tls above mirror endpoints[0] for the banner / backward compat.
  std::vector<PoolEndpoint> endpoints;
  std::string wallet;
  std::string worker = "pm";
  std::string agent = std::string("Kan/") + KAN_VERSION;
  uint64_t batch = 1000;
  bool real_cfg = true;
  bool use_tc = true;
  bool breakdown = false;
  int api_port = 0;            // >0 enables the HTTP/JSON monitoring API.
  int gpu_index = 0;           // CUDA device index visible in this process.
  int physical_gpu_index = 0;  // Original machine GPU index for logs/workers.
  int gpu_count = 1;           // Total GPUs managed by the parent process.
};
struct PoolState {
  std::mutex mu;
  std::condition_variable cv;
  std::string header, target, job_id;
  long long height = 0;
  uint64_t gen = 0, seq = 0;
  bool have_job = false;
  std::atomic<bool> stop{false};
  std::atomic<std::atomic<bool>*> active_stop{nullptr};
  std::unordered_set<long long> submit_ids;
  std::unordered_map<long long, std::string> submit_resp;
  uint64_t notify_count = 0;
  uint64_t abort_count = 0;
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
    if (sr < 0) { st->stop = true; st->cv.notify_all(); break; }
    if (sr == 0) continue;
    ssize_t n = ssl ? SSL_read(ssl, tmp, sizeof(tmp)) : recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) { st->stop = true; st->cv.notify_all(); break; }
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
          st->notify_count++;
          if (changed) {
            st->gen++;
            st->seq++;
            to_abort = st->active_stop.load();
            if (to_abort) st->abort_count++;
          }
        }
        log_linef("stratum", "new job id=%s height=%lld diff=%lld seq=%llu",
                  job_id.c_str(), height, diff, (unsigned long long)st->seq);
        st->cv.notify_all();
        if (to_abort) to_abort->store(true);
      } else {
        long long id = 0;
        bool has_id = json_int(line, "id", id);
        bool is_submit = false;
        {
          std::lock_guard<std::mutex> lk(st->mu);
          is_submit = has_id && st->submit_ids.count(id);
          if (is_submit) st->submit_resp[id] = line;
        }
        if (is_submit) st->cv.notify_all();
      }
    }
  }
  st->cv.notify_all();
}
// ===========================================================================
// monitoring API (HTTP/JSON) — rig-manager friendly (HiveOS / mmpOS / curl)
// ===========================================================================
// The miner is multi-process (one lane per GPU). execvp wipes shared mmap, so
// lanes publish their stats as per-lane JSON files in KAN_API_DIR and the HTTP
// server (run by the multi-GPU parent, or by a single-GPU process for itself)
// reads + aggregates them on each request. Enabled with --api-port N.
static std::string json_escape(const std::string& s) {
  std::string o; o.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
    else if ((unsigned char)c >= 0x20) o.push_back(c);
  }
  return o;
}
// One lane's stats, taken from THIS process's globals.
static std::string api_lane_json() {
  double up_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
  uint64_t acc = g_accepted.load(), rej = g_rejected.load();
  double ths = window_ths(60.0);
  unsigned idx = g_gpus.empty() ? 0 : g_gpus[0].index;
  const char* name = g_gpus.empty() ? "GPU" : g_gpus[0].name;
  unsigned temp = 0, fan = 0, pmw = 0;
  if (g_nvml.ok && !g_gpus.empty() && g_gpus[0].handle) {
    g_nvml.DeviceGetTemperature(g_gpus[0].handle, 0, &temp);
    g_nvml.DeviceGetFanSpeed(g_gpus[0].handle, &fan);
    g_nvml.DeviceGetPowerUsage(g_gpus[0].handle, &pmw);
  }
  char buf[640];
  snprintf(buf, sizeof(buf),
    "{\"id\":%u,\"name\":\"%s\",\"hashrate_ths\":%.2f,\"temp\":%u,\"fan\":%u,"
    "\"power_w\":%.1f,\"accepted\":%llu,\"rejected\":%llu,\"uptime\":%.0f}",
    idx, json_escape(name).c_str(), ths, temp, fan, (double)pmw / 1000.0,
    (unsigned long long)acc, (unsigned long long)rej, up_s);
  return std::string(buf);
}
static void api_write_file(const std::string& dir, int physical_gpu) {
  std::string path = dir + "/gpu" + std::to_string(physical_gpu) + ".json";
  std::string tmp = path + ".tmp";
  FILE* f = fopen(tmp.c_str(), "w");
  if (!f) return;
  std::string j = api_lane_json();
  fwrite(j.data(), 1, j.size(), f);
  fclose(f);
  rename(tmp.c_str(), path.c_str());   // atomic publish
}
// Read every gpu*.json in dir and build the aggregated machine response.
static std::string api_aggregate_json(const std::string& dir, const std::string& pool,
                                      const std::string& wallet) {
  std::string gpus = "[";
  double tot_ths = 0; unsigned long long tot_acc = 0, tot_rej = 0;
  double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
  bool first = true;
  DIR* d = opendir(dir.c_str());
  if (d) {
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
      const char* nm = e->d_name;
      size_t L = strlen(nm);
      if (L < 8 || strncmp(nm, "gpu", 3) != 0 || strcmp(nm + L - 5, ".json") != 0) continue;
      FILE* f = fopen((dir + "/" + nm).c_str(), "r");
      if (!f) continue;
      char buf[1024];
      size_t n = fread(buf, 1, sizeof(buf) - 1, f);
      fclose(f);
      if (n == 0) continue;
      buf[n] = 0;
      if (!first) gpus += ",";
      first = false;
      gpus += buf;
      const char* p;
      if ((p = strstr(buf, "\"hashrate_ths\":")) != nullptr) tot_ths += atof(p + 15);
      if ((p = strstr(buf, "\"accepted\":"))     != nullptr) tot_acc += strtoull(p + 11, nullptr, 10);
      if ((p = strstr(buf, "\"rejected\":"))     != nullptr) tot_rej += strtoull(p + 11, nullptr, 10);
    }
    closedir(d);
  }
  gpus += "]";
  char head[1024];
  snprintf(head, sizeof(head),
    "{\"miner\":\"kan\",\"version\":\"%s\",\"algo\":\"pearl\",\"uptime\":%.0f,"
    "\"pool\":\"%s\",\"wallet\":\"%s\",\"total\":{\"hashrate_ths\":%.2f,"
    "\"accepted\":%llu,\"rejected\":%llu},\"gpus\":",
    KAN_VERSION, up, json_escape(pool).c_str(), json_escape(wallet).c_str(),
    tot_ths, tot_acc, tot_rej);
  return std::string(head) + gpus + "}";
}
// Minimal HTTP/1.1 server: one short-lived JSON response per request. Runs on a
// dedicated thread; select() with a 1s timeout lets it observe *stop.
static void api_serve(int port, std::string dir, std::string pool, std::string wallet,
                      std::atomic<bool>* stop) {
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  if (ls < 0) { log_line("api", "socket failed"); return; }
  int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons((uint16_t)port);
  if (bind(ls, (sockaddr*)&a, sizeof(a)) != 0) { log_linef("api", "bind :%d failed (port in use?)", port); close(ls); return; }
  if (listen(ls, 8) != 0) { log_line("api", "listen failed"); close(ls); return; }
  log_linef("api", "monitoring JSON on http://0.0.0.0:%d", port);
  while (!stop->load()) {
    fd_set rf; FD_ZERO(&rf); FD_SET(ls, &rf);
    timeval tv{1, 0};
    if (select(ls + 1, &rf, nullptr, nullptr, &tv) <= 0) continue;
    int c = accept(ls, nullptr, nullptr);
    if (c < 0) continue;
    char req[1024]; recv(c, req, sizeof(req), 0);    // read+discard request line
    std::string body = api_aggregate_json(dir, pool, wallet);
    char hdr[256];
    int hl = snprintf(hdr, sizeof(hdr),
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Access-Control-Allow-Origin: *\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
      body.size());
    send(c, hdr, hl, 0);
    send(c, body.data(), body.size(), 0);
    close(c);
  }
  close(ls);
}

static int run_pool(const PoolOpts& o) {
  SSL_library_init();
  char gpu_cat[32];
  snprintf(gpu_cat, sizeof(gpu_cat), "GPU #%d", o.physical_gpu_index);
  cudaError_t ce = cudaSetDevice(o.gpu_index);
  if (ce != cudaSuccess) {
    log_linef(gpu_cat, "cudaSetDevice(%d) failed: %s",
              o.gpu_index, cudaGetErrorString(ce));
    return 2;
  }
  // The pool-facing worker name identifies the VPS / machine, not one GPU.
  // Each GPU lane runs as its own process and opens its OWN stratum connection,
  // but every lane authorizes as the SAME worker name so pool-side accounting
  // aggregates the whole box instead of splitting it into gpu0/gpu1/... workers.
  // This is per-GPU-connection fan-in by worker name — NOT one shared stratum
  // session and NOT a parent-multiplexed single connection (that is a future
  // item, see run_pool_parent_multigpu).
  const std::string& worker = o.worker;
  log_linef(gpu_cat, "selected CUDA device %d (physical GPU #%d of %d) worker=%s",
            o.gpu_index, o.physical_gpu_index, o.gpu_count, worker.c_str());

  // Build the endpoint list: primary first, then any failover backups. (Started
  // with a single --pool => one entry; the {host,port,use_tls} fallback keeps any
  // older call path working.)
  std::vector<PoolEndpoint> eps = o.endpoints;
  if (eps.empty()) eps.push_back({o.host, o.port, o.use_tls});

  int fd = -1;
  SSL_CTX* ctx = nullptr;
  SSL* ssl = nullptr;
  PoolState st;
  std::thread rd;
  std::string proto, cur_host;
  int cur_port = 0;
  std::string auth = "{\"id\":1,\"method\":\"mining.authorize\",\"params\":{\"wallet\":\"" +
                     o.wallet + "." + worker + "\",\"worker\":\"" + worker +
                     "\",\"agent\":\"" + o.agent + "\"}}";

  // Try each endpoint in order; mine the first that connects AND delivers a job
  // (= authorize accepted). On a clean failure advance to the next backup. If ALL
  // fail we return so the outer supervisor restarts the lane, which scans again
  // from the primary (natural primary-preference once the primary recovers).
  bool connected = false;
  for (size_t ei = 0; ei < eps.size() && !connected; ei++) {
    const PoolEndpoint& ep = eps[ei];
    proto = ep.use_tls ? "stratum+tls://" : "stratum+tcp://";
    if (eps.size() > 1)
      log_linef("stratum", "endpoint %zu/%zu: %s%s:%d",
                ei + 1, eps.size(), proto.c_str(), ep.host.c_str(), ep.port);

    fd = tcp_connect(ep.host, ep.port);
    if (fd < 0) { log_linef("stratum", "connect failed (%s:%d)", ep.host.c_str(), ep.port); continue; }

    ctx = nullptr; ssl = nullptr;
    if (ep.use_tls) {
      ctx = SSL_CTX_new(TLS_client_method());
      if (!ctx) { close(fd); fd = -1; continue; }
      SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
      ssl = SSL_new(ctx);
      SSL_set_fd(ssl, fd);
      SSL_set_tlsext_host_name(ssl, ep.host.c_str());
      if (SSL_connect(ssl) != 1) {
        log_linef("stratum", "TLS handshake failed (%s:%d)", ep.host.c_str(), ep.port);
        SSL_free(ssl); SSL_CTX_free(ctx); close(fd);
        fd = -1; ssl = nullptr; ctx = nullptr; continue;
      }
    }

    log_linef("stratum", "connecting to %s%s:%d as %s.%s (GPU %d/%d, per-GPU connection)",
              proto.c_str(), ep.host.c_str(), ep.port, o.wallet.c_str(),
              worker.c_str(), o.physical_gpu_index, o.gpu_count);
    st.stop = false;
    { std::lock_guard<std::mutex> lk(st.mu); st.have_job = false; }
    rd = std::thread(pool_reader, fd, ssl, &st);
    pool_send(fd, ssl, auth);
    double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0 < 25) {
      { std::lock_guard<std::mutex> lk(st.mu); if (st.have_job) break; }
      if (st.stop.load()) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    bool got_job;
    { std::lock_guard<std::mutex> lk(st.mu); got_job = st.have_job; }
    if (got_job) {
      log_linef("stratum", "authorize: ok wallet=%s.%s agent=%s",
                o.wallet.c_str(), worker.c_str(), o.agent.c_str());
      cur_host = ep.host; cur_port = ep.port;
      connected = true;
    } else {
      log_linef("stratum", "no job within 25s (%s:%d)", ep.host.c_str(), ep.port);
      st.stop = true;
      if (rd.joinable()) rd.join();
      if (ssl) { SSL_shutdown(ssl); SSL_free(ssl); ssl = nullptr; }
      if (ctx) { SSL_CTX_free(ctx); ctx = nullptr; }
      close(fd); fd = -1;
    }
  }
  if (!connected) { log_line("stratum", "all pool endpoints failed"); return 3; }

  // ---- monitoring API: publish THIS lane's stats; a standalone single-GPU
  // process also serves the aggregate. In multi-GPU the parent runs the server
  // and children here just write their gpuN.json into the inherited KAN_API_DIR.
  std::atomic<bool> api_stop{false};
  std::thread api_writer, api_server;
  std::string api_dir;
  {
    const char* ed = getenv("KAN_API_DIR");
    bool is_child = getenv("KAN_GPU_CHILD") && *getenv("KAN_GPU_CHILD");
    if (ed && *ed) api_dir = ed;
    else if (!is_child && o.api_port > 0) {
      api_dir = "/tmp/kan-api-" + std::to_string((long)getpid());
      mkdir(api_dir.c_str(), 0755);
      setenv("KAN_API_DIR", api_dir.c_str(), 1);
    }
    if (!api_dir.empty()) {
      int pg = o.physical_gpu_index;
      std::string d = api_dir;
      api_writer = std::thread([d, pg, &api_stop]() {
        while (!api_stop.load()) {
          api_write_file(d, pg);
          for (int i = 0; i < 30 && !api_stop.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      });
    }
    if (!is_child && o.api_port > 0 && !api_dir.empty()) {
      api_server = std::thread(api_serve, o.api_port, api_dir,
                               proto + cur_host + ":" + std::to_string(cur_port),
                               o.wallet, &api_stop);
    }
  }
  struct ApiGuard {
    std::atomic<bool>* s; std::thread* w; std::thread* v;
    ~ApiGuard() { s->store(true); if (w->joinable()) w->join(); if (v->joinable()) v->join(); }
  } api_guard{&api_stop, &api_writer, &api_server};
  long long submit_id = 100;
  uint64_t attempt_id = 0;
  std::atomic<bool> stop_attempt{false};
  std::atomic<bool> stats_req{false};

  // Submit responses take ~0.7s on the observed Kryptex pool path.  Waiting for
  // that response on the mining thread leaves the GPU idle after every found
  // proof.  Keep proof generation synchronous for now (correctness-critical),
  // but move the network submit+verdict wait to one worker so the main loop can
  // immediately start the next mining attempt after queueing a fresh share.
  struct SubmitJob {
    long long id = 0;
    std::string job_id;
    std::string proof_b64;
    long long hs_actual = 0;
  };
  std::mutex submit_mu;
  std::condition_variable submit_cv;
  std::deque<SubmitJob> submit_q;
  bool submit_stop = false;
  std::mutex send_mu;
  std::thread submit_th([&]() {
    while (true) {
      SubmitJob job;
      {
        std::unique_lock<std::mutex> lk(submit_mu);
        submit_cv.wait(lk, [&]() { return submit_stop || !submit_q.empty(); });
        if (submit_stop) break;
        job = std::move(submit_q.front());
        submit_q.pop_front();
      }

      {
        std::lock_guard<std::mutex> lk(st.mu);
        st.submit_ids.insert(job.id);
      }

      std::string sub = "{\"id\":" + std::to_string(job.id) +
                        ",\"method\":\"mining.submit\",\"params\":{\"job_id\":\"" +
                        job.job_id + "\",\"plain_proof\":\"" + job.proof_b64 +
                        "\",\"hs\":" + std::to_string(job.hs_actual) + "}}";
      bool sent = false;
      {
        std::lock_guard<std::mutex> lk(send_mu);
        sent = pool_send(fd, ssl, sub);
      }
      if (!sent) {
        {
          std::lock_guard<std::mutex> lk(st.mu);
          st.submit_ids.erase(job.id);
        }
        log_line("stratum", "share submit send failed");
        st.stop = true;
        st.cv.notify_all();
        continue;
      }

      auto submit_t0 = std::chrono::steady_clock::now();
      auto deadline = submit_t0 + std::chrono::seconds(30);
      std::string verdict;
      {
        std::unique_lock<std::mutex> lk(st.mu);
        st.cv.wait_until(lk, deadline, [&]() {
          return st.stop.load() ||
                 st.submit_resp.find(job.id) != st.submit_resp.end();
        });
        auto it = st.submit_resp.find(job.id);
        if (it != st.submit_resp.end()) {
          verdict = it->second;
          st.submit_resp.erase(it);
        }
        st.submit_ids.erase(job.id);
      }
      double submit_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - submit_t0).count();
      if (!verdict.empty() && verdict.find("\"result\":true") != std::string::npos) {
        g_accepted.fetch_add(1);
        g_last_share = std::chrono::steady_clock::now();
        log_linef(gpu_cat, "share accepted submit_wait=%.3fs", submit_s);
      } else if (!verdict.empty()) {
        g_rejected.fetch_add(1);
        log_linef(gpu_cat, "share rejected submit_wait=%.3fs: %s",
                  submit_s, verdict.substr(0, 100).c_str());
      } else if (!st.stop.load()) {
        log_linef(gpu_cat, "share submit timeout submit_wait=%.3fs", submit_s);
      }
    }
  });
  log_linef("stratum", "async share submit worker active gpu=%d",
            o.physical_gpu_index);

  std::thread stats_th([&]() {
    bool first_print = true;
    auto last_print = std::chrono::steady_clock::now();
    while (!st.stop.load()) {
      auto now = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(now - last_print).count();
      sample_draws();
      if (stats_req.load() || (first_print && dt >= 15.0) || (!first_print && dt >= 120.0)) {
        print_stats(o.wallet, proto + cur_host + ":" + std::to_string(cur_port));
        last_print = now;
        first_print = false;
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
    auto attempt_t0 = std::chrono::steady_clock::now();
    st.active_stop.store(&stop_attempt);
    int rc = mine_plain_proof(P, R, &stop_attempt);
    st.active_stop.store(nullptr);
    auto attempt_t1 = std::chrono::steady_clock::now();
    double attempt_s = std::chrono::duration<double>(attempt_t1 - attempt_t0).count();

    if (R.draws > 0 && R.work_per_draw > 0.0) {
      g_total_draws.fetch_add(R.draws);
      g_work_per_draw.store(R.work_per_draw);
    }

    bool fresh;
    { std::lock_guard<std::mutex> lk(st.mu);
      fresh = (st.gen == cur_gen && st.job_id==jid && st.header==hdr &&
               st.target==tgt && st.height==height); }

    attempt_id++;
    const char* reason = "batch";
    if (stop_attempt.load(std::memory_order_relaxed)) reason = "job_abort";
    else if (rc != 0) reason = "error";
    else if (R.found && !fresh) reason = "stale_win";
    else if (R.found) reason = "found";
    if (strcmp(reason, "batch") != 0 || (attempt_id % 20) == 0) {
      double ths = (R.draws > 0 && R.work_per_draw > 0.0 && attempt_s > 0.0)
                       ? (double)R.draws * R.work_per_draw / attempt_s / 1e12
                       : 0.0;
      uint64_t notify_count = 0, abort_count = 0, seq = 0;
      { std::lock_guard<std::mutex> lk(st.mu);
        notify_count = st.notify_count;
        abort_count = st.abort_count;
        seq = st.seq; }
      log_linef("perf",
                "gpu=%d attempt=%llu reason=%s rc=%d fresh=%d "
                "draws=%llu elapsed=%.3fs %.2f TH/s seq=%llu "
                "notify=%llu abort=%llu",
                o.physical_gpu_index, (unsigned long long)attempt_id,
                reason, rc, fresh ? 1 : 0,
                (unsigned long long)R.draws, attempt_s, ths,
                (unsigned long long)seq, (unsigned long long)notify_count,
                (unsigned long long)abort_count);
    }

    if (rc == 0 && R.found && fresh) {
      SubmitJob job;
      job.id = submit_id++;
      job.job_id = jid;
      job.proof_b64 = std::move(R.proof_b64);
      job.hs_actual = (long long)(R.draws * R.work_per_draw / R.elapsed_s);
      {
        std::lock_guard<std::mutex> lk(submit_mu);
        submit_q.push_back(std::move(job));
      }
      submit_cv.notify_one();
    }
    // Continue mining regardless of result (no win, stale, or error - keep going)
  }
  st.stop = true;
  st.cv.notify_all();
  {
    std::lock_guard<std::mutex> lk(submit_mu);
    submit_stop = true;
  }
  submit_cv.notify_all();
  if (stats_th.joinable()) stats_th.join();
  if (submit_th.joinable()) submit_th.join();
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
  std::string about = std::string("Kan/") + KAN_VERSION;
  log_line("about", about.c_str());
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
    "kan — Kan Pearl(PRL) PoUW miner\n\n"
    "  kan --algo pearl --pool URL --wallet ADDR[.WORKER]\n"
    "  kan --solo --node host:port --rpcuser U --rpcpass P --addr <p2tr>\n\n"
    "pool URL:  stratum+ssl://host:port  or  stratum+tcp://host:port\n"
    "           repeat --pool for failover backups (primary first; the miner\n"
    "           advances to the next pool when one is unreachable).\n\n"
    "GPU selection (pool mode):\n"
    "  default          use ALL detected GPUs automatically (one isolated lane\n"
    "                   process per GPU; all lanes share the same worker name so\n"
    "                   the pool aggregates the whole machine).\n"
    "  --devices 0,1,3  use only these physical GPU indices (subset auto-fanout).\n"
    "  CUDA_VISIBLE_DEVICES=...\n"
    "                   if set in the environment, the miner RESPECTS it and runs\n"
    "                   a single lane on whatever it exposes; auto-fanout is then\n"
    "                   DISABLED (use this to pin one GPU or integrate with an\n"
    "                   external scheduler).\n\n"
    "options:   --worker NAME   pool worker name (default pm)\n"
    "           --batch N       max draws per attempt before re-checking job\n"
    "           --api-port N    serve HTTP/JSON stats on port N (HiveOS/mmpOS)\n"
    "           --breakdown     per-draw timing\n"
    "commands:  s (stats now), q (quit); table prints every 120s\n");
}

static volatile sig_atomic_t g_parent_signal = 0;
static void parent_signal_handler(int sig) {
  g_parent_signal = sig;
}

static int env_int(const char* name, int def) {
  const char* v = getenv(name);
  if (!v || !*v) return def;
  char* end = nullptr;
  long x = strtol(v, &end, 10);
  if (end == v) return def;
  return (int)x;
}

static void normalize_cuda_device_order() {
  // CUDA defaults to FASTEST_FIRST ordering, while NVML enumerates PCI-bus order.
  // The multi-GPU code treats user-facing indices as NVML/physical indices, so
  // force CUDA ordinal N to mean the same physical GPU N before any CUDA query.
  setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
}

static int cpu_thread_count() {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
}

static bool parse_device_list(const std::string& s, std::vector<unsigned>& out) {
  out.clear();
  size_t pos = 0;
  while (pos < s.size()) {
    while (pos < s.size() && (s[pos] == ',' || s[pos] == ' ' || s[pos] == '\t')) pos++;
    if (pos >= s.size()) break;
    char* end = nullptr;
    unsigned long v = strtoul(s.c_str() + pos, &end, 10);
    if (end == s.c_str() + pos) return false;
    out.push_back((unsigned)v);
    pos = (size_t)(end - s.c_str());
    while (pos < s.size() && s[pos] != ',') {
      if (s[pos] != ' ' && s[pos] != '\t') return false;
      pos++;
    }
  }
  for (size_t i = 0; i < out.size(); i++) {
    for (size_t j = i + 1; j < out.size(); j++) {
      if (out[i] == out[j]) return false;
    }
  }
  return !out.empty();
}

static bool device_requested(const std::vector<unsigned>& list, unsigned idx) {
  if (list.empty()) return true;
  for (unsigned v : list) if (v == idx) return true;
  return false;
}

static bool should_enable_physical_gpu(bool gpu_child, int child_phys_gpu,
                                       const std::vector<unsigned>& requested,
                                       unsigned idx) {
  if (gpu_child) return (int)idx == child_phys_gpu;
  return device_requested(requested, idx);
}

static bool physical_gpu_available(const std::vector<GPUInfo>& gpus, unsigned idx) {
  for (const auto& gi : gpus) if (gi.index == idx) return true;
  return false;
}

static bool parse_first_visible_device_index(const char* cvd, unsigned& out) {
  if (!cvd || !*cvd) return false;
  char* end = nullptr;
  unsigned long v = strtoul(cvd, &end, 10);
  if (end == cvd) return false;
  if (*end != '\0' && *end != ',' && *end != ' ' && *end != '\t') return false;
  out = (unsigned)v;
  return true;
}

static void select_single_visible_gpu(unsigned physical_gpu) {
  // Reuse the same isolation model as forked lanes for an explicit one-GPU
  // selection.  After this, cudaSetDevice(0) targets the selected physical GPU.
  std::string gpu = std::to_string(physical_gpu);
  setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1);
}

// Auto multi-GPU presents one command and one pool worker name to the user and
// the pool.  Internally the parent acts as a supervisor and spawns one isolated
// GPU lane PROCESS per physical GPU; each lane opens its OWN stratum connection
// and authorizes under the shared worker name (pool-side aggregation by worker,
// not a single multiplexed session).
// This intentionally avoids running multiple prover instances as threads in one
// address space: tc_cutlass_v2.cu / gpu_prep.cu keep persistent device buffers
// in file-static globals, so multi-threading would risk sharing GPU pointers
// across devices.  Per-GPU lane processes keep those globals isolated.
// A single parent-multiplexed stratum session with unified stats is a future
// item and is intentionally NOT implemented here.
static int run_pool_parent_multigpu(char** argv, const PoolOpts& base,
                                    const std::vector<unsigned>& gpu_indices) {
  unsigned gpu_count = (unsigned)gpu_indices.size();
  log_linef("multigpu",
            "supervisor: starting %u per-GPU lane(s) under one worker name "
            "(each lane = isolated process + own stratum connection)",
            gpu_count);
  int host_threads = cpu_thread_count();
  int lane_threads = (host_threads + (int)gpu_count - 1) / (int)gpu_count;
  if (lane_threads < 1) lane_threads = 1;
  bool user_omp = getenv("OMP_NUM_THREADS") && *getenv("OMP_NUM_THREADS");
  if (user_omp) {
    log_linef("multigpu", "OMP_NUM_THREADS=%s supplied by user",
              getenv("OMP_NUM_THREADS"));
  } else {
    log_linef("multigpu", "auto CPU split: %d host threads / %u GPUs -> OMP_NUM_THREADS=%d per lane",
              host_threads, gpu_count, lane_threads);
  }

  std::vector<pid_t> pids;
  pids.reserve(gpu_count);

  // monitoring API: the parent owns the HTTP server; children inherit KAN_API_DIR
  // (set BEFORE fork) and publish their per-lane JSON there for aggregation.
  std::atomic<bool> api_stop{false};
  std::thread api_server;
  if (base.api_port > 0) {
    std::string api_dir = "/tmp/kan-api-" + std::to_string((long)getpid());
    mkdir(api_dir.c_str(), 0755);
    setenv("KAN_API_DIR", api_dir.c_str(), 1);
    std::string pool = std::string(base.use_tls ? "stratum+tls://" : "stratum+tcp://") +
                       base.host + ":" + std::to_string(base.port);
    api_server = std::thread(api_serve, base.api_port, api_dir, pool, base.wallet, &api_stop);
  }
  struct PApiGuard {
    std::atomic<bool>* s; std::thread* v;
    ~PApiGuard() { s->store(true); if (v->joinable()) v->join(); }
  } papi_guard{&api_stop, &api_server};

  struct sigaction sa{};
  sa.sa_handler = parent_signal_handler;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  for (unsigned lane = 0; lane < gpu_count; lane++) {
    unsigned physical_gpu = gpu_indices[lane];
    pid_t pid = fork();
    if (pid < 0) {
      log_linef("multigpu", "fork failed for GPU #%u: errno=%d",
                physical_gpu, errno);
      continue;
    }
    if (pid == 0) {
      std::string gpu = std::to_string(physical_gpu);
      std::string total = std::to_string(gpu_count);
      setenv("KAN_GPU_CHILD", "1", 1);
      setenv("KAN_GPU_PHYSICAL_INDEX", gpu.c_str(), 1);
      setenv("KAN_GPU_COUNT", total.c_str(), 1);
      setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
      setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1);
      if (!user_omp) {
        std::string omp = std::to_string(lane_threads);
        setenv("OMP_NUM_THREADS", omp.c_str(), 1);
        setenv("OMP_DYNAMIC", "FALSE", 1);
      }
      execvp(argv[0], argv);
      fprintf(stderr, "execvp(%s) failed: errno=%d\n", argv[0], errno);
      _exit(127);
    }
    pids.push_back(pid);
    log_linef("multigpu", "GPU #%u lane %u/%u pid=%d worker=%s",
              physical_gpu, lane + 1, gpu_count, (int)pid,
              base.worker.c_str());
  }

  if (pids.empty()) {
    log_line("multigpu", "no GPU lanes started");
    return 2;
  }

  int rc_final = 0;
  size_t alive = pids.size();
  bool terminating = false;
  while (alive > 0) {
    if (g_parent_signal && !terminating) {
      log_linef("multigpu", "signal %d, stopping GPU lanes",
                (int)g_parent_signal);
      for (pid_t p : pids) kill(p, SIGTERM);
      terminating = true;
      g_parent_signal = 0;
    }
    int status = 0;
    pid_t w = wait(&status);
    if (w < 0) {
      if (errno == EINTR && g_parent_signal) {
        log_linef("multigpu", "signal %d, stopping GPU lanes",
                  (int)g_parent_signal);
        for (pid_t p : pids) kill(p, SIGTERM);
        terminating = true;
        g_parent_signal = 0;
        continue;
      }
      if (errno == ECHILD) break;
      continue;
    }
    alive--;
    if (WIFEXITED(status)) {
      int rc = WEXITSTATUS(status);
      log_linef("multigpu", "lane pid=%d exited rc=%d", (int)w, rc);
      if (rc != 0 && rc_final == 0) rc_final = rc;
    } else if (WIFSIGNALED(status)) {
      int sig = WTERMSIG(status);
      log_linef("multigpu", "lane pid=%d killed by signal %d", (int)w, sig);
      if (rc_final == 0) rc_final = 128 + sig;
    }
    // A single dead lane means the VPS is no longer mining at full capacity.
    // Stop the remaining lanes and let the external supervisor restart the
    // whole miner cleanly instead of silently running degraded forever.
    if (!terminating && alive > 0) {
      log_line("multigpu", "lane stopped; stopping remaining GPU lanes");
      for (pid_t p : pids) {
        if (p != w) kill(p, SIGTERM);
      }
      terminating = true;
    }
  }
  return rc_final;
}

int main(int argc, char** argv) {
  normalize_cuda_device_order();
  if (argc < 2) { usage(); return 2; }
  bool pool = false, solo = false;
  PoolOpts po; SoloOpts so;
  bool gpu_child = env_int("KAN_GPU_CHILD", 0) != 0;
  int child_phys_gpu = env_int("KAN_GPU_PHYSICAL_INDEX", 0);
  int child_gpu_count = env_int("KAN_GPU_COUNT", 1);
  std::vector<unsigned> requested_devices;
  std::vector<std::string> pool_urls;   // one or more --pool (primary + backups)
  std::string algo;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* def="") -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
    };
    if (a == "--algo") algo = next("pearl");
    else if (a == "--pool") { pool_urls.push_back(next()); pool = true; }
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
    else if (a == "--devices") {
      std::string d = next();
      if (!parse_device_list(d, requested_devices)) {
        fprintf(stderr, "invalid --devices list: %s\n", d.c_str());
        return 2;
      }
    }
    else if (a == "--batch") po.batch = so.batch = strtoull(next("1000").c_str(), nullptr, 10);
    else if (a == "--cfg") { std::string c = next("real"); po.real_cfg = so.real_cfg = (c == "real"); }
    else if (a == "--tc") po.use_tc = so.use_tc = true;
    else if (a == "--api-port") po.api_port = atoi(next("0").c_str());
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
  // --devices only affects pool-mode auto-fanout. Solo mode runs a single
  // prover lane (one node template at a time), so reject the flag there instead
  // of silently ignoring it.
  if (solo && !requested_devices.empty()) {
    fprintf(stderr, "--devices is only valid in --pool mode "
                    "(solo mode runs a single GPU lane)\n");
    return 2;
  }
  // If the operator pins GPUs through the environment, respect it and refuse to
  // also auto-fanout: CUDA_VISIBLE_DEVICES remaps CUDA ordinals, so combining it
  // with --devices (which filters NVML/physical indices) is ambiguous. Pick one.
  {
    const char* cvd_arg = getenv("CUDA_VISIBLE_DEVICES");
    if (!gpu_child && cvd_arg && *cvd_arg && !requested_devices.empty()) {
      fprintf(stderr,
              "--devices and CUDA_VISIBLE_DEVICES are mutually exclusive; "
              "CUDA_VISIBLE_DEVICES=%s is already set so auto-fanout is "
              "disabled. Drop --devices or unset CUDA_VISIBLE_DEVICES.\n",
              cvd_arg);
      return 2;
    }
  }
  if (pool) {
    if (po.wallet.empty()) {
      fprintf(stderr, "pool mode requires --wallet ADDR[.WORKER]\n");
      usage();
      return 2;
    }
    for (const std::string& purl : pool_urls) {
      PoolEndpoint ep;
      std::string hp;
      if (purl.rfind("stratum+ssl://", 0) == 0 || purl.rfind("stratum+tls://", 0) == 0) {
        ep.use_tls = true; ep.port = 8048; hp = purl.substr(14);
      } else if (purl.rfind("stratum+tcp://", 0) == 0) {
        ep.use_tls = false; ep.port = 7048; hp = purl.substr(14);
      } else {
        fprintf(stderr, "unrecognized pool URL (need stratum+ssl:// or stratum+tcp://): %s\n", purl.c_str());
        usage(); return 2;
      }
      size_t c = hp.rfind(':');
      if (c != std::string::npos) { ep.host = hp.substr(0, c); ep.port = atoi(hp.substr(c+1).c_str()); }
      else { ep.host = hp; }
      if (ep.host.empty()) { fprintf(stderr, "empty pool host in URL: %s\n", purl.c_str()); usage(); return 2; }
      po.endpoints.push_back(ep);
    }
    if (!po.endpoints.empty()) {
      po.host = po.endpoints[0].host;
      po.port = po.endpoints[0].port;
      po.use_tls = po.endpoints[0].use_tls;
    }
    banner();
    std::string proto = po.use_tls ? "stratum+tls://" : "stratum+tcp://";
    log_linef("pool", "%s%s:%d", proto.c_str(), po.host.c_str(), po.port);
    for (size_t ei = 1; ei < po.endpoints.size(); ei++) {
      const PoolEndpoint& bep = po.endpoints[ei];
      log_linef("pool", "backup #%zu %s%s:%d", ei,
                bep.use_tls ? "stratum+tls://" : "stratum+tcp://",
                bep.host.c_str(), bep.port);
    }
    log_linef("wallet", "%s.%s", po.wallet.c_str(), po.worker.c_str());
    log_linef("worker", "%s", po.worker.c_str());
    log_line("commands", "s (stats), q (quit); table every 120s");
    if (gpu_child) {
      log_linef("multigpu", "GPU lane mode physical GPU #%d of %d",
                child_phys_gpu, child_gpu_count);
    }
    g_nvml.init();
    if (g_nvml.ok) {
      unsigned cnt = 0;
      g_nvml.DeviceGetCount(&cnt);
      // Full driver version string from NVML
      char drv_str[32] = "N/A";
      // Try to get driver version from CUDA API as fallback
      int cuda_drv = 0;
      cudaDriverGetVersion(&cuda_drv);
      snprintf(drv_str, sizeof(drv_str), "%d.%d", cuda_drv / 1000, (cuda_drv % 1000) / 10);
      log_linef("detected", "%u devices - driver %s", cnt, drv_str);
      for (unsigned i = 0; i < cnt && i < 16; i++) {
        if (!should_enable_physical_gpu(gpu_child, child_phys_gpu,
                                        requested_devices, i)) continue;
        GPUInfo gi;
        gi.index = i;
        g_nvml.DeviceGetHandleByIndex(i, &gi.handle);
        if (gi.handle) {
          g_nvml.DeviceGetName(gi.handle, gi.name, sizeof(gi.name));
          // Shorten GPU name: "NVIDIA GeForce RTX 4090" → "RTX 4090"
          const char* shortname = gi.name;
          if (strncmp(shortname, "NVIDIA GeForce ", 15) == 0) shortname += 15;
          else if (strncmp(shortname, "NVIDIA ", 7) == 0) shortname += 7;
          // Overwrite gi.name with short version for stats table
          if (shortname != gi.name) memmove(gi.name, shortname, strlen(shortname)+1);
          struct { unsigned long long free, total, used; } mem{};
          g_nvml.DeviceGetMemoryInfo(gi.handle, &mem);
          gi.vram_mb = mem.total / (1024*1024);
          g_nvml.DeviceGetCudaComputeCapability(gi.handle, &gi.sm_maj, &gi.sm_min);
          struct { char busId[16]; unsigned domain, bus, device; } pci{};
          g_nvml.DeviceGetPciInfo(gi.handle, &pci);
          gi.bus = pci.bus;
          log_linef("GPU", "#%u %-18s %lluGB sm_%d%d bus:%02x enabled",
                    i, gi.name, gi.vram_mb/1024, gi.sm_maj, gi.sm_min, gi.bus);
        }
        g_gpus.push_back(gi);
      }
    } else {
      int cuda_cnt = 0;
      if (cudaGetDeviceCount(&cuda_cnt) != cudaSuccess || cuda_cnt < 1) cuda_cnt = 1;
      unsigned detected_count = gpu_child ? (unsigned)child_gpu_count : (unsigned)cuda_cnt;
      log_linef("detected", "%u devices - driver N/A", detected_count);
      unsigned begin = gpu_child ? (unsigned)child_phys_gpu : 0;
      unsigned end = gpu_child ? begin + 1 : detected_count;
      for (unsigned i = begin; i < end && i < 16; i++) {
        if (!should_enable_physical_gpu(gpu_child, child_phys_gpu,
                                        requested_devices, i)) continue;
        GPUInfo gi;
        gi.index = i;
        log_linef("GPU", "#%u GPU              N/A sm_xx bus:00 enabled", i);
        g_gpus.push_back(gi);
      }
    }
    if (!gpu_child && !requested_devices.empty()) {
      for (unsigned req : requested_devices) {
        if (!physical_gpu_available(g_gpus, req)) {
          log_linef("multigpu", "requested --devices GPU #%u is not available", req);
          return 2;
        }
      }
    }
    if (!gpu_child && !requested_devices.empty() && g_gpus.empty()) {
      log_line("multigpu", "none of the requested --devices are available");
      return 2;
    }
    if (g_gpus.empty()) {
      GPUInfo gi;
      gi.index = gpu_child ? (unsigned)child_phys_gpu : 0;
      log_linef("GPU", "#%u GPU              N/A sm_xx bus:00 enabled", gi.index);
      g_gpus.push_back(gi);
    }
    log_line("devfee", "0%");
    if (!requested_devices.empty() && !gpu_child) {
      std::string s;
      for (size_t i = 0; i < requested_devices.size(); i++) {
        if (i) s += ",";
        s += std::to_string(requested_devices[i]);
      }
      log_linef("multigpu", "device filter requested: %s", s.c_str());
    }
    po.gpu_index = 0;
    po.physical_gpu_index = gpu_child ? child_phys_gpu : (int)g_gpus[0].index;
    po.gpu_count = gpu_child ? child_gpu_count : (int)g_gpus.size();
    const char* cvd = getenv("CUDA_VISIBLE_DEVICES");
    bool external_cvd = (!gpu_child && cvd && *cvd);
    if (gpu_child) {
      // Child lanes are already scoped by the parent's CUDA_VISIBLE_DEVICES.
    } else if (external_cvd) {
      // Under an external CUDA_VISIBLE_DEVICES this process only uses CUDA
      // ordinal 0 (the first GPU the env var exposes). CUDA_DEVICE_ORDER=PCI_BUS_ID
      // lets simple numeric masks map back to the same NVML physical index.
      unsigned visible0 = 0;
      if (parse_first_visible_device_index(cvd, visible0) &&
          physical_gpu_available(g_gpus, visible0)) {
        for (const auto& gi : g_gpus) {
          if (gi.index == visible0) {
            GPUInfo selected = gi;
            g_gpus.assign(1, selected);
            break;
          }
        }
        po.physical_gpu_index = (int)visible0;
      } else {
        // UUID/MIG masks cannot be mapped to an NVML index here; keep telemetry
        // honest instead of reporting the wrong physical GPU as active.
        GPUInfo gi;
        gi.index = 0;
        snprintf(gi.name, sizeof(gi.name), "CUDA-visible");
        g_gpus.assign(1, gi);
        po.physical_gpu_index = 0;
      }
      po.gpu_count = 1;
      log_linef("multigpu",
                "CUDA_VISIBLE_DEVICES=%s set; respecting it, auto fanout disabled "
                "(single lane on CUDA ordinal 0)",
                cvd);
    } else if (po.gpu_count > 1) {
      std::vector<unsigned> indices;
      for (const auto& gi : g_gpus) indices.push_back(gi.index);
      log_linef("multigpu", "%s%u GPUs -> auto fanout (one isolated lane per GPU, shared worker=%s)",
                requested_devices.empty() ? "auto-detected " : "selected ",
                (unsigned)indices.size(), po.worker.c_str());
      return run_pool_parent_multigpu(argv, po, indices);
    } else {
      select_single_visible_gpu((unsigned)po.physical_gpu_index);
      log_linef("multigpu", "single GPU #%d -> one lane (no fanout)%s",
                po.physical_gpu_index,
                requested_devices.empty() ? "" : " (selected via --devices)");
    }
    return run_pool(po);
  } else {
    banner();
    return run_solo(so);
  }
}
