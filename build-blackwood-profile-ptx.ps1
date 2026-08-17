# Builds the INSTRUMENTED Blackwood kernel (SolveBlackwoodKernel.profile.ptx).
#
# The production SolveBlackwoodKernel.ptx is NOT touched by this script -- the
# profiling counters live behind -DBW_PROFILE_COUNTERS, so the shipped kernel
# compiles from the same source with an identical signature and zero added cost.
# Only BlackwoodGpuProfileHarness loads the .profile.ptx produced here.
#
# nvcc needs cl.exe on PATH even for -ptx-only output.

$ErrorActionPreference = "Stop"

$clDir = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
if (-not (Test-Path (Join-Path $clDir "cl.exe"))) {
    # Fall back to whatever MSVC version is actually installed rather than failing on a hardcoded path.
    $found = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "cl.exe" -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -like "*Hostx64\x64*" } | Select-Object -First 1
    if ($null -eq $found) { throw "cl.exe not found -- nvcc cannot compile without it." }
    $clDir = $found.DirectoryName
}
$env:PATH = "$clDir;$env:PATH"

$root = "C:\Users\knuda\IdeaProjects\Eternity"
nvcc -ptx -O3 -arch=compute_120 -DBW_PROFILE_COUNTERS `
    (Join-Path $root "SolveBlackwoodKernel.cu") `
    -o (Join-Path $root "SolveBlackwoodKernel.profile.ptx")

if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
Write-Output "Built SolveBlackwoodKernel.profile.ptx"
