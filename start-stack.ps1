# start-stack.ps1  —  bring the whole research stack UP, detached from any shell.
# Order: OmniRoute (:20128) -> Balance Broker (:8902) -> LightGent (:8100).
# Each runs as an independent hidden process that SURVIVES this shell/session
# closing, so another repo can keep calling http://127.0.0.1:8100/research.
#
# Usage:   powershell -ExecutionPolicy Bypass -File start-stack.ps1
# Stop:    powershell -ExecutionPolicy Bypass -File stop-stack.ps1
# Logs:    C:\Users\hi\Desktop\lightgent\logs\*.log

$ErrorActionPreference = 'SilentlyContinue'
$BROKER = 'C:\Users\hi\Desktop\balance-broker'
$LIGHT  = 'C:\Users\hi\Desktop\lightgent'
$LOG    = Join-Path $LIGHT 'logs'
if (-not (Test-Path $LOG)) { New-Item -ItemType Directory -Path $LOG | Out-Null }

function Test-Up($port) {
  [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}
function Wait-Up($port, $secs) {
  for ($i=0; $i -lt ($secs*2); $i++) { if (Test-Up $port) { return $true }; Start-Sleep -Milliseconds 500 }
  return $false
}

# --- 1. OmniRoute (:20128) — free DeepSeek floor lanes -----------------------
if (Test-Up 20128) {
  Write-Output "OmniRoute  : already up"
} else {
  $omni = 'C:\Users\hi\AppData\Roaming\npm\omniroute.ps1'
  Start-Process -WindowStyle Hidden powershell.exe `
    -ArgumentList @('-ExecutionPolicy','Bypass','-File',$omni) `
    -RedirectStandardOutput (Join-Path $LOG 'omniroute.out.log') `
    -RedirectStandardError  (Join-Path $LOG 'omniroute.err.log')
  if (Wait-Up 20128 40) { Write-Output "OmniRoute  : UP" } else { Write-Output "OmniRoute  : did NOT come up (check logs\omniroute.*.log)" }
}

# --- 2. Balance Broker (:8902) — the brain (Mistral x2 + DeepSeek floor) ------
if (Test-Up 8902) {
  Write-Output "Broker     : already up"
} else {
  # Load Mistral keys from gitignored .env.local into this shell's env so the
  # broker child inherits them (upstream.py reads api_key_env from os.environ).
  Get-Content (Join-Path $BROKER '.env.local') | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)\s*$') {
      Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
  }
  $env:BROKER_LANES = 'lanes-endurance.yaml'
  $env:BROKER_STATE = 'state-endurance.sqlite'
  Start-Process -WindowStyle Hidden python.exe `
    -WorkingDirectory $BROKER `
    -ArgumentList @('-m','uvicorn','broker.router:app','--host','127.0.0.1','--port','8902','--log-level','warning') `
    -RedirectStandardOutput (Join-Path $LOG 'broker.out.log') `
    -RedirectStandardError  (Join-Path $LOG 'broker.err.log')
  if (Wait-Up 8902 25) { Write-Output "Broker     : UP" } else { Write-Output "Broker     : did NOT come up (check logs\broker.*.log)" }
}

# --- 3. LightGent (:8100) — the research service (reads .env: broker + Jina) --
if (Test-Up 8100) {
  Write-Output "LightGent  : already up"
} else {
  Start-Process -WindowStyle Hidden python.exe `
    -WorkingDirectory $LIGHT `
    -ArgumentList @('-m','uvicorn','lightgent_service:app','--host','127.0.0.1','--port','8100','--log-level','warning') `
    -RedirectStandardOutput (Join-Path $LOG 'lightgent.out.log') `
    -RedirectStandardError  (Join-Path $LOG 'lightgent.err.log')
  if (Wait-Up 8100 25) { Write-Output "LightGent  : UP" } else { Write-Output "LightGent  : did NOT come up (check logs\lightgent.*.log)" }
}

Write-Output "---"
Write-Output "Research endpoint for other repos:  POST http://127.0.0.1:8100/research"
Write-Output "Health:                             GET  http://127.0.0.1:8100/health"
