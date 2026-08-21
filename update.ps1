# Pulls the latest worker and restarts it. Safe to run any time.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# uv installs itself under %USERPROFILE%\.local\bin, which is not on PATH for a
# PowerShell window that was already open when setup ran. Looking where it
# actually lives is the difference between this working and a bare
# "uv is not recognized" that reads like the update itself is broken.
function Resolve-Uv {
  $onPath = Get-Command uv -ErrorAction SilentlyContinue
  if ($onPath) { return $onPath.Source }
  $candidates = @(
    (Join-Path $env:UV_INSTALL_DIR "uv.exe"),
    (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
    (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe")
  )
  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path $candidate)) { return $candidate }
  }
  return $null
}

$uv = Resolve-Uv
if (-not $uv) {
  Write-Host "uv is not installed. Installing it now..."
  irm https://astral.sh/uv/install.ps1 | iex
  $uv = Resolve-Uv
  if (-not $uv) {
    Write-Host ""
    Write-Host "uv still could not be found after installing it."
    Write-Host "Run .\setup.ps1 instead, or install uv yourself: https://docs.astral.sh/uv/"
    exit 1
  }
}

git pull --ff-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $uv sync --python 3.12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $uv run peerpixel dashboard
