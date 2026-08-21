# Builds the environment inside this folder: a standalone Python plus every
# library, in .\.venv. Nothing is installed system-wide.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# See the note in update.ps1: uv's installer does not reliably reach the PATH of
# a shell that is already running, so look where it actually puts itself.
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
  Write-Host "Installing uv..."
  irm https://astral.sh/uv/install.ps1 | iex
  $uv = Resolve-Uv
  if (-not $uv) {
    Write-Host ""
    Write-Host "uv installed but could not be found afterwards."
    Write-Host "Install it yourself and run this again: https://docs.astral.sh/uv/"
    exit 1
  }
}

& $uv python install 3.12
& $uv sync --python 3.12

Write-Host ""
Write-Host "Done. Opening the worker dashboard..."
& $uv run peerpixel dashboard
