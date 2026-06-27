# build_windows.ps1 — Windows build for kan (Pearl PRL PoUW miner)
#
# Mirrors peral/build.sh for the Windows CI / MSVC + CUDA toolchain.
# Produces build/{kan.exe, plainproof_gen.exe, pearl-miner.exe}
#
# Prerequisites (set by the GitHub Actions workflow):
#   - CUDA_PATH       (Jimver/cuda-toolkit)
#   - VCPKG_ROOT      (GitHub runner default: C:\vcpkg)
#   - VCPKG_DEFAULT_TRIPLET = x64-windows
#
# This builds the WMMA (tc_block.cu) path — no CUTLASS on Windows v1.
# BLAKE3 uses portable scalar only (no SIMD asm on Windows).

param(
  [string]$ARCH = "",
  [switch]$Help
)

if ($Help) {
  Write-Host "Usage: .\build_windows.ps1 [-ARCH sm_XX]"
  Write-Host "  -ARCH  Override GPU arch (default: portable sm_75+sm_86)"
  Write-Host "Environment: CUDA_PATH, VCPKG_ROOT"
  exit 0
}

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$BUILD = "$ROOT\build"

# ---- version ---------------------------------------------------------------
Push-Location $ROOT
$version = & git describe --tags --dirty --always 2>$null
if (-not $version) { $version = "dev" }
Pop-Location
$kanVersionDef = "/DKAN_VERSION=`"$version`""

Write-Host "=== kan Windows build ==="
Write-Host "  ROOT:        $ROOT"
Write-Host "  BUILD:       $BUILD"
Write-Host "  VERSION:     $version"
Write-Host "  CUDA_PATH:   $env:CUDA_PATH"
Write-Host "  VCPKG_ROOT:  $env:VCPKG_ROOT"

# ---- toolchain check -------------------------------------------------------
$clPath = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
if (-not $clPath) { throw "MSVC cl.exe not found — open a Developer Command Prompt or run from GitHub Actions windows-2022" }
Write-Host "  cl.exe:      $clPath"

$nvccPath = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
if (-not $nvccPath) { throw "nvcc.exe not found — set CUDA_PATH" }
Write-Host "  nvcc.exe:    $nvccPath"

# ---- arch selection --------------------------------------------------------
$GENCODE = @()
if ($ARCH) {
  $GENCODE = @("-arch=$ARCH")
  Write-Host "  arch:        ENV override -> $ARCH"
} else {
  $GENCODE = @(
    "-gencode", "arch=compute_75,code=sm_75",
    "-gencode", "arch=compute_86,code=sm_86"
  )
  Write-Host "  arch:        portable -> sm_75 + sm_86 (WMMA v1)"
}

# ---- create build dir ------------------------------------------------------
New-Item -Path $BUILD -ItemType Directory -Force | Out-Null

# ---- BLAKE3 (scalar only, no SIMD asm on Windows) --------------------------
Write-Host "=== blake3 (scalar) ==="
$B3 = "$ROOT\blake3"
Push-Location $B3
$b3Flags = @(
  "/O2", "/I.", "/DBLAKE3_NO_SSE2", "/DBLAKE3_NO_SSE41",
  "/DBLAKE3_NO_AVX2", "/DBLAKE3_NO_AVX512", "/c"
)
cl $b3Flags blake3.c 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "blake3.c failed" }
cl $b3Flags blake3_dispatch.c 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "blake3_dispatch.c failed" }
cl $b3Flags blake3_portable.c 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "blake3_portable.c failed" }
Move-Item *.obj $BUILD\ -Force
Pop-Location
Write-Host "  blake3: portable scalar OK"

# ---- locate the vcpkg root that actually has OpenSSL -----------------------
# msvc-dev-cmd repoints VCPKG_ROOT at the VS-bundled vcpkg, but the workflow
# installs openssl into the GitHub-hosted vcpkg (VCPKG_INSTALLATION_ROOT, =C:\vcpkg).
# Probe the known roots and pick whichever one actually contains openssl/err.h.
$vcpkgRoot = $null
foreach ($r in @($env:VCPKG_INSTALLATION_ROOT, $env:VCPKG_ROOT, "C:\vcpkg")) {
  if ($r -and (Test-Path "$r\installed\x64-windows\include\openssl\err.h")) { $vcpkgRoot = $r; break }
}
if (-not $vcpkgRoot) { throw "vcpkg x64-windows OpenSSL not found (looked in VCPKG_INSTALLATION_ROOT, VCPKG_ROOT, C:\vcpkg)" }
Write-Host "  vcpkg root:  $vcpkgRoot"

# ---- include paths for MSVC ------------------------------------------------
$cuInclude = "-I$env:CUDA_PATH\include"
$vcpkgInclude = "-I$vcpkgRoot\installed\x64-windows\include"
$srcInc = "-I$ROOT\src"
$b3Inc = "-I$ROOT\blake3"

# ---- host: prover core (plainproof_gen + prover_lib) -----------------------
Write-Host "=== host: prover core ==="
$hostFlags = @(
  "/O2", "/std:c++17", "/openmp", "/DKAN_NO_ASYNC_SEARCH",
  $srcInc, $b3Inc, $cuInclude, $vcpkgInclude, "/c"
)
cl $hostFlags "$ROOT\src\plainproof_gen.cpp" "/Fo$BUILD\plainproof_gen.obj" 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "plainproof_gen.cpp (CLI) failed" }
cl $hostFlags "/DPROVER_LIB" "$ROOT\src\plainproof_gen.cpp" "/Fo$BUILD\prover_lib.obj" 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "plainproof_gen.cpp (PROVER_LIB) failed" }
Write-Host "  prover core: OK"

# ---- host: unified miner driver --------------------------------------------
Write-Host "=== host: miner driver ==="
cl $hostFlags $kanVersionDef "$ROOT\src\miner_main.cpp" "/Fo$BUILD\miner_main.obj" 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "miner_main.cpp failed" }
Write-Host "  miner driver: OK"

# ---- CUDA kernel: tc_block.cu (WMMA) + gpu_prep.cu -------------------------
Write-Host "=== CUDA kernels (WMMA) ==="
Push-Location $BUILD
# IMPORTANT: build the nvcc arg list as a FLAT array (use + to concatenate, not
# nesting $GENCODE inside @(...)) and pass via splat @. PowerShell otherwise
# joins nested-array elements into one quoted token and nvcc rejects it with
# "Unknown option '-gencode arch=... -gencode arch=...'".
$cuFlags = @("-O3", "-std=c++17", "-DKAN_NO_ASYNC_SEARCH", "-I$ROOT\src") + $GENCODE
nvcc @cuFlags -c "$ROOT\src\tc_block.cu" -o tc_kernel.obj 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "tc_block.cu failed" }
nvcc @cuFlags -c "$ROOT\src\gpu_prep.cu" -o gpu_prep.obj 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "gpu_prep.cu failed" }
Pop-Location
Write-Host "  CUDA kernels: OK"

# ---- link: plainproof_gen.exe ----------------------------------------------
Write-Host "=== link: plainproof_gen.exe ==="
$vcpkgLib = "$vcpkgRoot\installed\x64-windows\lib"
$cudaLib = "$env:CUDA_PATH\lib\x64"
$b3Obj = @(
  "$BUILD\blake3.obj", "$BUILD\blake3_dispatch.obj", "$BUILD\blake3_portable.obj"
)
# FLAT concatenation (+) + splat (@) — nesting $b3Obj inside @(...) makes
# PowerShell join the obj paths into one token and link fails with LNK1104.
$ppObj = @("$BUILD\plainproof_gen.obj", "$BUILD\tc_kernel.obj", "$BUILD\gpu_prep.obj") + $b3Obj
link /OUT:"$BUILD\plainproof_gen.exe" @ppObj `
  cudart.lib ws2_32.lib `
  "/LIBPATH:$cudaLib" 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "plainproof_gen.exe link failed" }
Write-Host "  plainproof_gen.exe: OK ($((Get-Item "$BUILD\plainproof_gen.exe").Length / 1KB) KB)"

# ---- link: kan.exe ---------------------------------------------------------
Write-Host "=== link: kan.exe ==="
$kanObj = @("$BUILD\miner_main.obj", "$BUILD\prover_lib.obj", "$BUILD\tc_kernel.obj", "$BUILD\gpu_prep.obj") + $b3Obj
link /OUT:"$BUILD\kan.exe" @kanObj `
  "libssl.lib" "libcrypto.lib" cudart.lib ws2_32.lib `
  "/LIBPATH:$vcpkgLib" "/LIBPATH:$cudaLib" 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "kan.exe link failed" }
Write-Host "  kan.exe: OK ($((Get-Item "$BUILD\kan.exe").Length / 1KB) KB)"

# ---- compat alias ----------------------------------------------------------
Copy-Item "$BUILD\kan.exe" "$BUILD\pearl-miner.exe" -Force
Write-Host "  pearl-miner.exe: compat alias"

Write-Host ""
Write-Host "BUILD OK:"
Write-Host "  $BUILD\plainproof_gen.exe"
Write-Host "  $BUILD\kan.exe"
Write-Host "  $BUILD\pearl-miner.exe"
