# ensure_electron.ps1 - make sure the Electron binary is in place.
#
# Why this script: electron npm package's install (@electron/get) can silently
# no-op on newer Node, leaving node_modules/electron/dist with only `locales`
# and no electron.exe. Then `electron .` throws "failed to install correctly"
# and the window flashes and disappears. This script bypasses @electron/get:
# download the official zip from npmmirror, extract to dist, write path.txt
# (electron's index.js uses it to locate the binary).
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File ensure_electron.ps1
# Exit codes: 0 = ready (was present or fixed now); 1 = fix failed.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$electronPkg = Join-Path $repoRoot "electron-app\node_modules\electron"

if (-not (Test-Path (Join-Path $electronPkg "package.json"))) {
    Write-Output "[electron] package not installed (node_modules\electron missing); run npm install first."
    exit 1
}

$version = (Get-Content (Join-Path $electronPkg "package.json") | ConvertFrom-Json).version
$dist = Join-Path $electronPkg "dist"
$exe = Join-Path $dist "electron.exe"
$pathTxt = Join-Path $electronPkg "path.txt"

if ((Test-Path $exe) -and (Test-Path $pathTxt)) {
    Write-Output "[electron] ready v$version (electron.exe present), skip."
    exit 0
}

Write-Output "[electron] binary missing, installing v$version ..."
if (-not (Test-Path $dist)) { New-Item -ItemType Directory -Path $dist | Out-Null }

$mirror = $env:ELECTRON_MIRROR
if (-not $mirror) { $mirror = "https://npmmirror.com/mirrors/electron/" }
$mirror = $mirror.TrimEnd("/") + "/"
$url = "${mirror}${version}/electron-v${version}-win32-x64.zip"
$zip = Join-Path $electronPkg "electron.zip"

Write-Output "[electron] downloading $url"
try {
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 600
} catch {
    Write-Output "[electron] download failed: $($_.Exception.Message)"
    exit 1
}
$len = (Get-Item $zip).Length
if ($len -lt 10MB) {
    $mb = [math]::Round($len/1MB, 2)
    Write-Output "[electron] downloaded file too small (${mb}MB), aborting."
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    exit 1
}
$mb = [math]::Round($len/1MB, 1)
Write-Output "[electron] downloaded ${mb}MB, extracting to dist ..."

Expand-Archive -Path $zip -DestinationPath $dist -Force
Set-Content -Path $pathTxt -Value "electron.exe" -NoNewline -Encoding ASCII
Remove-Item $zip -Force -ErrorAction SilentlyContinue

if (-not (Test-Path $exe)) {
    Write-Output "[electron] electron.exe still missing after extract, install failed."
    exit 1
}
Write-Output "[electron] install ok: $exe"
exit 0
