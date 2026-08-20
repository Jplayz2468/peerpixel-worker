# Builds the environment inside this folder: a standalone Python plus every
# library, in .\.venv. Nothing is installed system-wide.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv..."
  irm https://astral.sh/uv/install.ps1 | iex
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv python install 3.12
uv sync --python 3.12

Write-Host ""
Write-Host "Done. Opening the worker dashboard..."
uv run peerpixel dashboard
