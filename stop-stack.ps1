# stop-stack.ps1  —  stop the research stack (LightGent :8100, Broker :8902).
# Leaves OmniRoute (:20128) alone by default since other tools may use it;
# pass -All to stop OmniRoute too.
param([switch]$All)
$ErrorActionPreference = 'SilentlyContinue'
$ports = @(8100, 8902)
if ($All) { $ports += 20128 }
foreach ($p in $ports) {
  $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
  if ($conn) {
    $procId = $conn[0].OwningProcess
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    Write-Output "port $p : stopped (pid $procId)"
  } else {
    Write-Output "port $p : not running"
  }
}
