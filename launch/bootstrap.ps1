# Everything that has to happen before the app can draw its own window.
#
# The twin of launch/bootstrap.sh, and it does the same two things: get uv, get
# a standalone Python, hand over. Windows gets a real progress bar for it,
# because PowerShell has one, and the rule does not have an exception for the
# part before the app starts -- that is the part somebody is most likely to
# think has hung.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# A VIRTUAL_ENV inherited from the shell would be used in preference to this
# folder's own, and PeerPixel would start on an interpreter with none of its
# libraries. Whatever somebody has activated elsewhere is not this.
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
$log = Join-Path $root ".peerpixel-setup.log"
Set-Content -Path $log -Value "" -Encoding utf8

function Write-Banner {
  Write-Host ""
  Write-Host "  PeerPixel" -ForegroundColor Yellow
  Write-Host "  getting this machine ready. it only happens once." -ForegroundColor DarkGray
  Write-Host ""
}

# uv installs itself under %USERPROFILE%\.local\bin, which is not on the PATH of
# a window that was already open. Looking where it actually lives is the
# difference between this working and a bare "uv is not recognized" that reads
# like PeerPixel itself is broken.
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

# See creep() in peerpixel/progress.py. Constant speed up to the estimate, then
# ever slower, approaching the end without arriving: an estimate that turns out
# short must not leave a frozen bar behind it.
function Invoke-Step {
  param([string]$Label, [double]$Estimate, [scriptblock]$Work)

  $job = Start-Job -ScriptBlock $Work
  $started = Get-Date
  while ($job.State -eq "Running") {
    $t = ((Get-Date) - $started).TotalSeconds
    if ($t -le $Estimate) { $p = 0.9 * $t / $Estimate; $left = $Estimate - $t }
    else { $o = ($t - $Estimate) / $Estimate; $p = 0.995 - 0.095 / (1 + $o); $left = $t * (0.995 - $p) / $p }
    Write-Progress -Activity "PeerPixel" -Status $Label `
      -PercentComplete ([Math]::Min(99, [Math]::Round($p * 100))) `
      -SecondsRemaining ([Math]::Max(0, [Math]::Round($left)))
    Start-Sleep -Milliseconds 250
  }
  Receive-Job $job -ErrorAction SilentlyContinue | Out-File -Append -FilePath $log -Encoding utf8
  $ok = $job.State -eq "Completed"
  Remove-Job $job -Force
  Write-Progress -Activity "PeerPixel" -Status $Label -Completed
  Write-Host ("  {0}: done" -f $Label) -ForegroundColor DarkGray
  return $ok
}

function Stop-Here($message) {
  Write-Host ""
  Write-Host "  That did not work." -ForegroundColor Red
  Write-Host "  $message"
  Write-Host ""
  Write-Host "  The last few lines of $log :"
  Get-Content $log -Tail 12 | ForEach-Object { Write-Host "    $_" }
  Write-Host ""
  Read-Host "  Press return to close"
  exit 1
}

Write-Banner

$uv = Resolve-Uv
if (-not $uv) {
  if (-not (Invoke-Step "Fetching the installer" 25 { irm https://astral.sh/uv/install.ps1 | iex })) {
    Stop-Here "Could not download uv. Check the connection and try again."
  }
  $uv = Resolve-Uv
  if (-not $uv) { Stop-Here "uv installed but could not be found. Install it yourself: https://docs.astral.sh/uv/" }
}

& $uv python find 3.12 *> $null
if ($LASTEXITCODE -ne 0) {
  $found = $uv
  if (-not (Invoke-Step "Fetching Python" 35 { & $using:found python install 3.12 })) {
    Stop-Here "Could not install Python 3.12."
  }
}

# --no-project on purpose. This interpreter only has to be able to *start*
# PeerPixel; the rendering libraries go into .venv, installed by PeerPixel
# itself with a progress bar on it, and it moves onto that interpreter the
# moment they are there. See runtime.use_venv.
$env:PEERPIXEL_UV = $uv
& $uv run --no-project --python 3.12 python -m peerpixel @args
