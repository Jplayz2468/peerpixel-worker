# Windows equivalent of setup.sh. Same idea: the interpreter and the libraries
# land in .venv inside this folder, and nothing is installed system-wide.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv (fetches a standalone Python, no system changes)..."
  irm https://astral.sh/uv/install.ps1 | iex
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv python install 3.12
uv sync --python 3.12

Write-Host ""
Write-Host "Done. The environment is in .\.venv - interpreter and all."
Write-Host "  node bin\peerpixel.mjs pair <CODE>"
Write-Host "  node bin\peerpixel.mjs bench"
Write-Host "  node bin\peerpixel.mjs run"
