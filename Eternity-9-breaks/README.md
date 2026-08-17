# Eternity-9-breaks

Joshua Blackwood's C# Eternity II solver (`github.com/jblackwood345/EternityII_Solver`), modified to
run a **9-entry break schedule** instead of his shipped 10, plus tooling so results are labelled and
pruned automatically instead of piling up unexamined.

This is the copy that actually produced this project's best `drop_239` results (13 conflicts as of
2026-08-17). It previously existed only in a temp directory; it is in the repo so the modifications
aren't lost with the session that made them.

## Why 9 breaks

`break_indexes_allowed` is a cumulative budget: an edge mismatch ("break") is forbidden below depth
201, and one more becomes permitted at each listed depth. Blackwood's shipped schedule has 10
entries, and the array's length is the maximum final conflict count reachable — so targeting 471
matched edges (11 conflicts) instead of his record 470 (12) means dropping to 9.

Which 9 was settled by measurement, not argument. A leave-one-out sweep (all 10 single-drop
variants plus the 10-entry baseline, 500M nodes each, ~2.3h total) found reach-rate climbing
near-monotonically the LATER the dropped entry sits:

| config | attempts | reach 245+ | best depth |
|---|---:|---:|---:|
| baseline (10 entries) | 540 | 98.7% | 250 |
| drop_201 | 540 | 15.6% | 248 |
| ... | | | |
| drop_237 | 810 | 57.5% | 248 |
| **drop_239** | 945 | **63.0%** | 248 |

The budget is cumulative, so dropping an early entry leaves the search one break short at *every*
depth from there on — a deficit compounding across nearly the whole search. Dropping the last entry
only creates that deficit for the final 17 positions. Hence `drop_239`:

```
201, 206, 211, 216, 221, 225, 229, 233, 237
```

Every 9-break config trails the 10-break baseline heavily on reach rate — that is the intrinsic
difficulty step of demanding one fewer break, not a defect in the schedule.

## Changes vs. upstream

| Area | Change |
|---|---|
| `Util.cs` | `break_indexes_allowed` overridable via `ETERNITY_BREAK_INDEXES` (falls back to Blackwood's own 10-entry schedule when unset) |
| `Util.cs` | After each save, runs `dk.puzzle.tools.HoleSolver` to label the board with its real conflict count |
| `Util.cs` | Prints `COMPLETED_LINK <file>: <bucas url>` for the *completed* (hole-filled) board |
| `Util.cs` | Retention: keeps only boards within 1 conflict of the best on disk; deletes the rest |
| `Util.cs` | Deletes the GUID-named baseboard once labelling succeeds |
| `Util.cs` | Saves to `%USERPROFILE%` (override: `ETERNITY_SOLUTIONS_DIR`), never OneDrive-redirected `Documents` |
| `Program.cs` | `node_cap`, thread count, run label, wall-clock budget all env-configurable |
| `Program.cs` | Per-generation `GEN_REACH` depth histogram |

Search logic itself is untouched — per-attempt depth and node counts are derived from
`SolvePuzzle()`'s existing return value.

## Running

```
ETERNITY_BREAK_INDEXES=201,206,211,216,221,225,229,233,237
ETERNITY_SOLUTIONS_DIR=%USERPROFILE%\EternitySolutions_drop239
ETERNITY_RUN_LABEL=drop239
ETERNITY_NODE_CAP=50000000000
dotnet build EternityII_Solver/EternityII_Solver.csproj -c Release
dotnet EternityII_Solver/bin/Release/net10.0/EternityII_Solver.dll
```

Unset `ETERNITY_BREAK_INDEXES` to get Blackwood's original 10-break behaviour.

## Note on the labelling hook

`TryLabelWithConflictCount` shells out to this repo's Java `HoleSolver` with a hardcoded
`eternityRoot` and JDK path. It must run with `WorkingDirectory` set to the Eternity project root —
`HoleSolver` loads `pieces.csv` by bare relative path and silently falls back to mock data
otherwise (logged at INFO, not thrown). Every failure path is wrapped: labelling can fail without
affecting the search, which only ever runs after the raw save has already succeeded.
