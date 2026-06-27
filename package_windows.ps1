# package_windows.ps1 — assemble the portable Windows package.
#
# Mirrors package_portable.sh (Linux): takes the built exes from build\ and
# produces a self-contained dist\kan-portable-windows-x64.zip whose only
# runtime dependency is the NVIDIA driver. Bundles the OpenSSL, CUDA runtime
# and MSVC C++ runtime DLLs next to the exes so it runs on any Windows box
# without a CUDA toolkit / vcpkg / VS redist installed.
#
# Run AFTER build_windows.ps1 (expects build\kan.exe etc. to exist).

param([string]$StageOnly = "0")

$ErrorActionPreference = "Stop"
$ROOT  = Split-Path -Parent $MyInvocation.MyCommand.Path
$BUILD = "$ROOT\build"
$DIST  = "$ROOT\dist"
$PKG   = "kan-portable-windows-x64"
$STAGE = "$DIST\$PKG"

if (-not (Test-Path "$BUILD\kan.exe")) {
  throw "build\kan.exe not found — run .\build_windows.ps1 first"
}

# ---- version ---------------------------------------------------------------
Push-Location $ROOT
$version = & git describe --tags --dirty --always 2>$null
if (-not $version) { $version = "dev" }
$commit = & git rev-parse --short HEAD 2>$null
Pop-Location
if (-not $commit) { $commit = "unknown" }

Write-Host "=== kan Windows portable package ==="
Write-Host "  version: $version  commit: $commit"

# ---- locate vcpkg (same probe as build_windows.ps1) -----------------------
$vcpkgRoot = $null
foreach ($r in @($env:VCPKG_INSTALLATION_ROOT, $env:VCPKG_ROOT, "C:\vcpkg")) {
  if ($r -and (Test-Path "$r\installed\x64-windows\include\openssl\err.h")) { $vcpkgRoot = $r; break }
}
if (-not $vcpkgRoot) { throw "vcpkg OpenSSL root not found" }

# ---- stage dir ------------------------------------------------------------
Remove-Item -Recurse -Force $STAGE -ErrorAction SilentlyContinue
New-Item -Path $STAGE -ItemType Directory -Force | Out-Null

# ---- exes ------------------------------------------------------------------
Copy-Item "$BUILD\kan.exe", "$BUILD\plainproof_gen.exe", "$BUILD\pearl-miner.exe" $STAGE
Write-Host "  exes: kan.exe, plainproof_gen.exe, pearl-miner.exe"

# ---- runtime DLLs (so the zip runs without toolkit / redist installed) ----
$copied = 0

# OpenSSL (libssl-3-x64.dll, libcrypto-3-x64.dll)
foreach ($pat in "libssl-*.dll", "libcrypto-*.dll") {
  Get-ChildItem "$vcpkgRoot\installed\x64-windows\bin\$pat" -ErrorAction SilentlyContinue |
    ForEach-Object { Copy-Item $_.FullName $STAGE; $copied++ }
}

# CUDA runtime (cudart64_12.dll)
foreach ($pat in "cudart64_*.dll") {
  Get-ChildItem "$env:CUDA_PATH\bin\$pat" -ErrorAction SilentlyContinue |
    ForEach-Object { Copy-Item $_.FullName $STAGE; $copied++ }
}

# MSVC C++ runtime (vcruntime140.dll, vcruntime140_1.dll, msvcp140.dll)
$crtDir = $null
if ($env:VCToolsRedistDir) {
  $crtDir = Get-ChildItem "$env:VCToolsRedistDir\x64\Microsoft.VC*.CRT" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
}
if ($crtDir) {
  foreach ($pat in "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll") {
    Get-ChildItem "$($crtDir.FullName)\$pat" -ErrorAction SilentlyContinue |
      ForEach-Object { Copy-Item $_.FullName $STAGE; $copied++ }
  }
} else {
  Write-Host "  WARNING: VCToolsRedistDir not set — MSVC CRT DLLs not bundled (most Windows boxes have them)" -ForegroundColor Yellow
}
Write-Host "  runtime DLLs bundled: $copied"

# ---- run.bat launcher (restart loop, mirrors run.sh) ----------------------
$runBat = @'
@echo off
REM Kan portable Windows launcher.
REM Usage:
REM   run.bat --algo pearl --pool stratum+tcp://host:port --wallet ADDR[.WORKER] --batch 1000 --cfg real --tc
REM
REM Restarts Kan after a disconnect/exit (except Ctrl+C / terminated by signal).
setlocal
set "KAN_EXE=%~dp0kan.exe"
:loop
"%KAN_EXE%" %*
set RC=%errorlevel%
if "%RC%"=="130" goto done
if "%RC%"=="143" goto done
echo [%date% %time%] Kan exited (rc=%RC%), restarting in 15s... (Ctrl+C to stop)
timeout /t 15 /nobreak >nul
goto loop
:done
endlocal
'@
Set-Content -Path "$STAGE\run.bat" -Value $runBat -Encoding ASCII

# ---- run.ps1 (PowerShell restart launcher, same logic) --------------------
$runPs1 = @'
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $here "kan.exe"
while ($true) {
  & $exe @Args
  $rc = $LASTEXITCODE
  if ($rc -eq 130 -or $rc -eq 143) { exit $rc }
  Write-Host "[$(Get-Date)] Kan exited (rc=$rc), restarting in 15s..." -ForegroundColor Yellow
  Start-Sleep -Seconds 15
}
'@
Set-Content -Path "$STAGE\run.ps1" -Value $runPs1 -Encoding ASCII

# ---- docs ------------------------------------------------------------------
$docs = @("GPU_PROFILES.md", "CHANGELOG.md")
foreach ($d in $docs) {
  if (Test-Path "$ROOT\$d") { Copy-Item "$ROOT\$d" $STAGE }
}

Set-Content -Path "$STAGE\VERSION" -Value $version -Encoding ASCII
Set-Content -Path "$STAGE\BUILD_INFO.txt" -Encoding ASCII -Value @"
version: $version
commit: $commit
platform: windows-x64
built_utc: $(Get-Date -AsUTC -Format 'yyyy-MM-ddTHH:mm:ssZ')
cuda_version: $($env:CUDA_VERSION ?? '12.5')
runtime_dependency: NVIDIA driver + Windows 10/11 x64
contents: kan.exe (miner), plainproof_gen.exe (proof CLI), pearl-miner.exe (alias), run.bat (restart launcher), bundled OpenSSL + CUDA runtime + MSVC CRT DLLs
"@

Set-Content -Path "$STAGE\README.txt" -Encoding ASCII -Value @"
Kan Pearl (PRL) PoUW miner - Windows portable package v$version

UNPACK AND RUN:
  1. Unzip this archive to a folder.
  2. Open a Command Prompt in that folder.
  3. Run:
       run.bat --algo pearl --pool stratum+tcp://prl.kryptex.network:7048 --wallet YOUR_PEARL_ADDRESS.WORKER --batch 1000 --cfg real --tc

REQUIREMENTS:
  - An NVIDIA GPU with a recent driver (Ada/Ampere/Hopper/Blackwell supported).
  - No CUDA toolkit, no vcpkg, no Visual Studio redist needed - all runtime
    DLLs are bundled next to kan.exe.

CONTENTS:
  kan.exe             the miner (pool + solo modes)
  plainproof_gen.exe  standalone proof generator / CLI
  pearl-miner.exe     alias of kan.exe
  run.bat / run.ps1   restart-loop launchers
  libssl-3-x64.dll, libcrypto-3-x64.dll, cudart64_*.dll, vcruntime140*.dll, msvcp140.dll
                      bundled runtime libraries
  GPU_PROFILES.md, CHANGELOG.md, BUILD_INFO.txt, VERSION

POOL MODE OPTIONS:
  --pool URL      stratum+tcp:// or stratum+ssl:// host:port (repeatable for failover)
  --wallet ADDR[.WORKER]
  --devices 0,1   GPU selection (default: all detected GPUs)
  --batch N       draws per attempt
  --cfg real      use the real Pearl network config (131072/131072/4096/256)
  --tc            tensor-core search kernel
  --api-port N    HTTP/JSON stats on port N

See run.bat header for the restart-loop behaviour. Shares are submitted
asynchronously so submit_wait never blocks the next mining attempt.
"@

# ---- zip -------------------------------------------------------------------
$zip = "$DIST\$PKG.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "$STAGE\*" -DestinationPath $zip -CompressionLevel Optimal
$zipSize = [math]::Round((Get-Item $zip).Length / 1MB, 2)
Write-Host ""
Write-Host "PACKAGE OK -> $zip ($zipSize MB)"
Write-Host "STAGE DIR  -> $STAGE"
Get-ChildItem $STAGE | Format-Table Name, Length -AutoSize
