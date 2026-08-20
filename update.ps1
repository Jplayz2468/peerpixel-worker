$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
git pull --ff-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv sync --python 3.12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run peerpixel dashboard
