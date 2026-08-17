# Overnight status for both Eternity solvers: what's running, what was found, best conflicts.
# Read-only -- deletes nothing, changes nothing.

$log = "C:\Users\knuda\IdeaProjects\Eternity\logs\eternity_solver.log"

function Best-Conflicts($dir) {
    if (-not (Test-Path $dir)) { return $null }
    $vals = Get-ChildItem $dir -Filter "Errors*_RawBoard.txt" -ErrorAction SilentlyContinue |
            ForEach-Object { if ($_.Name -match '^Errors(\d+)_') { [int]$Matches[1] } }
    if ($vals) { ($vals | Measure-Object -Minimum).Minimum } else { $null }
}

Write-Output "===== PROCESSES ====="
$gpu = Get-CimInstance Win32_Process -Filter "Name = 'java.exe'" | Where-Object { $_.CommandLine -like "*BlackwoodGpuRunner*" }
$csharp = Get-CimInstance Win32_Process -Filter "Name = 'dotnet.exe'" | Where-Object { $_.CommandLine -like "*EternityII_Solver*" }
if ($gpu) { Write-Output "  GPU solver   : RUNNING (PID $($gpu.ProcessId), since $($gpu.CreationDate))" }
else       { Write-Output "  GPU solver   : NOT RUNNING" }
if ($csharp) { Write-Output "  C# solver    : RUNNING (PID $($csharp.ProcessId), since $($csharp.CreationDate))" }
else         { Write-Output "  C# solver    : NOT RUNNING" }

Write-Output ""
Write-Output "===== BEST CONFLICTS ON DISK ====="
foreach ($d in @("EternitySolutions_GpuBlackwood", "EternitySolutions_drop239")) {
    $path = Join-Path $env:USERPROFILE $d
    $best = Best-Conflicts $path
    $count = if (Test-Path $path) { (Get-ChildItem $path -File).Count } else { 0 }
    $shown = if ($null -eq $best) { "none labelled" } else { "$best" }
    Write-Output ("  {0,-32} best={1,-14} files={2}" -f $d, $shown, $count)
}
Write-Output "  (all-time record to beat: 12)"

Write-Output ""
Write-Output "===== GPU SAVES (newest last) ====="
$saves = Select-String -Path $log -Pattern "SAVED \[" -ErrorAction SilentlyContinue | Select-Object -Last 15
if ($saves) { $saves | ForEach-Object { "  " + ($_.Line -replace '^.*BlackwoodGpuRunner - ', '') } }
else { Write-Output "  (none yet)" }

Write-Output ""
Write-Output "===== GPU HARVEST ACTIVITY ====="
$h = Select-String -Path $log -Pattern "Harvest:|Population harvest failed" -ErrorAction SilentlyContinue
Write-Output "  harvest rounds logged: $($h.Count)"
if ($h) { $h | Select-Object -Last 3 | ForEach-Object { "  " + ($_.Line -replace '^.*BlackwoodGpuRunner - ', '') } }

Write-Output ""
Write-Output "===== LATEST GPU LAUNCH ====="
$last = Select-String -Path $log -Pattern "BlackwoodGpuRunner - Launch" -ErrorAction SilentlyContinue | Select-Object -Last 1
if ($last) { Write-Output ("  " + ($last.Line -replace '^.*BlackwoodGpuRunner - ', '')) }

Write-Output ""
Write-Output "===== C# COMPLETED LINKS (last 3) ====="
$cl = Select-String -Path "C:\Users\knuda\IdeaProjects\Eternity\logs\drop239_current.log" -Pattern "COMPLETED_LINK" -ErrorAction SilentlyContinue | Select-Object -Last 3
if ($cl) { $cl | ForEach-Object { "  " + $_.Line.Substring(0, [Math]::Min(110, $_.Line.Length)) + "..." } }
else { Write-Output "  (none yet)" }
