// platform.h — thin cross-platform shim so the miner builds on both Linux (the
// production target) and Windows (desktop miners). The POSIX branch just pulls in
// the headers miner_main.cpp already used, so including this is a no-op there; the
// Windows branch maps the handful of POSIX calls the miner makes (sockets, env,
// mkdir, getpid, ssize_t, the NVML dynamic loader) onto Win32 equivalents.
//
// Scope note: Windows v1 targets SINGLE-GPU mining. The multi-GPU supervisor uses
// fork()+execvp()+wait()+sigaction (no Win32 equivalent without a CreateProcess
// rewrite), so that path stays #ifndef _WIN32 and Windows dispatches to the
// single-lane run_pool. Multi-GPU on Windows is a follow-up.
#ifndef KAN_PLATFORM_H
#define KAN_PLATFORM_H

// Portable attribute macros. MSVC has no GNU weak and no format-checking attribute;
// Linux/clang get the real ones. (The optional-link gpu_prep/tc_search_launch symbols
// are additionally gated by compile-time flags KAN_HAS_ASYNC_SEARCH /
// KAN_HAS_GATHER_EVENT so WMMA builds don't reference CUTLASS-only symbols at all.)
#if defined(_WIN32)
  #define KAN_WEAK
  #define KAN_FORMAT_PRINTF(a,b)
#else
  #define KAN_WEAK              __attribute__((weak))
  #define KAN_FORMAT_PRINTF(a,b) __attribute__((format(printf,a,b)))
#endif

#if defined(_WIN32)
// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>   // getaddrinfo / inet_ntop
#include <windows.h>    // LoadLibrary / GetProcAddress (NVML)
#include <process.h>    // _getpid
#include <direct.h>     // _mkdir
#include <io.h>
#include <cstdint>
#include <cstdlib>

// POSIX ssize_t is not provided by MSVC.
#ifndef _SSIZE_T_DEFINED
typedef long long ssize_t;
#define _SSIZE_T_DEFINED
#endif

// Sockets: the miner stores descriptors in `int`. SOCKET is wider but real fds
// fit, so keep the int-based code and only remap the calls that differ.
#ifndef close
#define close(s) closesocket(s)
#endif
#ifndef setenv
// _putenv_s ignores the overwrite flag (it always overwrites) — matches our use.
#define setenv(k, v, overwrite) _putenv_s((k), (v))
#endif
#ifndef unsetenv
#define unsetenv(k) _putenv_s((k), "")
#endif
#ifndef mkdir
#define mkdir(path, mode) _mkdir(path)
#endif
#ifndef getpid
#define getpid _getpid
#endif
// MSVC spells the pipe-to-command helpers _popen/_pclose. run_capture() (solo
// zkprove path) is compiled unconditionally, so map the POSIX names even though
// Windows v1 rarely exercises them.
#ifndef popen
#define popen(cmd, mode) _popen((cmd), (mode))
#endif
#ifndef pclose
#define pclose(p) _pclose((p))
#endif

// MSVC setsockopt takes `const char*` for optval; the miner passes `int*`
// (TCP_NODELAY / SO_REUSEADDR). Wrap it so the int-based call sites compile
// unchanged on both platforms.
#define setsockopt(s, lvl, opt, val, len) \
  setsockopt((s), (lvl), (opt), (const char*)(val), (len))

// Minimal POSIX dirent shim over the Win32 FindFirstFile API. The miner only
// enumerates one flat directory (the per-lane API dir) reading file names, so a
// thin wrapper covering opendir/readdir/closedir + d_name is enough. On Windows
// v1 (single-GPU) this dir holds at most one gpu*.json, but the code path must
// still compile and behave correctly.
#include <string>
struct dirent { char d_name[260]; };
struct DIR {
  HANDLE h = INVALID_HANDLE_VALUE;
  WIN32_FIND_DATAA fd{};
  bool first = true;
  dirent ent{};
};
static inline DIR* opendir(const char* path) {
  std::string pat = std::string(path) + "\\*";
  DIR* d = new DIR();
  d->h = FindFirstFileA(pat.c_str(), &d->fd);
  if (d->h == INVALID_HANDLE_VALUE) { delete d; return nullptr; }
  return d;
}
static inline struct dirent* readdir(DIR* d) {
  if (!d) return nullptr;
  if (!d->first) {
    if (!FindNextFileA(d->h, &d->fd)) return nullptr;
  }
  d->first = false;
  size_t n = sizeof(d->ent.d_name) - 1;
  strncpy(d->ent.d_name, d->fd.cFileName, n);
  d->ent.d_name[n] = 0;
  return &d->ent;
}
static inline int closedir(DIR* d) {
  if (!d) return -1;
  if (d->h != INVALID_HANDLE_VALUE) FindClose(d->h);
  delete d;
  return 0;
}

// The NVML loader uses dlopen/dlsym on POSIX; map to LoadLibrary/GetProcAddress.
typedef HMODULE kan_dl_t;
static inline kan_dl_t kan_dlopen_nvml(void) {
  return LoadLibraryA("nvml.dll");
}
static inline void* kan_dlsym(kan_dl_t h, const char* name) {
  return h ? (void*)GetProcAddress(h, name) : nullptr;
}

// Winsock needs one-time init before any socket call.
static inline void kan_net_init(void) {
  static bool done = false;
  if (!done) { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); done = true; }
}

#else
// ---------------------------------------------------------------------------
// POSIX (Linux) — the production build. Same headers miner_main.cpp used before.
// ---------------------------------------------------------------------------
#include <arpa/inet.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

typedef void* kan_dl_t;
static inline kan_dl_t kan_dlopen_nvml(void) {
  void* h = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  if (!h) h = dlopen("libnvidia-ml.so", RTLD_LAZY);
  return h;
}
static inline void* kan_dlsym(kan_dl_t h, const char* name) {
  return h ? dlsym(h, name) : nullptr;
}
static inline void kan_net_init(void) {}   // nothing to do on POSIX

#endif  // _WIN32
#endif  // KAN_PLATFORM_H
