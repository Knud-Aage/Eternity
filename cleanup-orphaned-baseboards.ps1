<#
.SYNOPSIS
Deletes orphaned "..._baseboard.txt" save files that have no matching "..._RawBoard.txt" sibling.

.DESCRIPTION
Util.cs's PruneAboveThreshold() deletes a completed board's three files (RawBoard, physical_layout,
baseboard) in sequence with no locking, called from inside the solver's own Parallel.For search.
At high save throughput, concurrent calls from different worker threads can race on the same
directory: whichever thread loses hits an exception partway through its delete sequence (RawBoard
is deleted first, with no exists-check), leaving the baseboard -- deleted last -- behind as an
orphan that nothing ever revisits (later scans only look for "Errors*_RawBoard.txt").

Harmless -- the boards involved were already below the keep threshold either way, so only a
~2.7KB stray file survives, never board data -- but accumulates over time. Runs independently of
the solver process; does not require stopping or restarting it.

Only ever deletes a "_baseboard.txt" file, and only when its exact "_RawBoard.txt" counterpart
(same prefix) does not exist. Never touches RawBoard.txt or physical_layout.txt.

.PARAMETER SolutionsDir
Folder to sweep. Defaults to EternitySolutions_CSharpCPU (the only save folder confirmed to show
this pattern as of 2026-09-04 -- Eternity2_GPU's and Eternity2_CPU's save folders use the same
write-then-prune pattern in Java but were clean when checked).
#>
param(
    [string]$SolutionsDir = (Join-Path $env:USERPROFILE "EternitySolutions_CSharpCPU")
)

if (-not (Test-Path $SolutionsDir)) {
    Write-Output "SolutionsDir not found: $SolutionsDir"
    exit 1
}

$removed = 0
Get-ChildItem -Path $SolutionsDir -Filter "Errors*_baseboard.txt" -File | ForEach-Object {
    $rawBoardSibling = $_.FullName -replace '_baseboard\.txt$', '_RawBoard.txt'
    if (-not (Test-Path -LiteralPath $rawBoardSibling)) {
        Remove-Item -LiteralPath $_.FullName -Force
        Write-Output "Removed orphan: $($_.Name)"
        $removed++
    }
}

Write-Output "Done. Removed $removed orphaned baseboard file(s) from $SolutionsDir."
