@echo off
REM Runs the C# solver from the repository root (pieces.csv is read by relative path --
REM same requirement as the Java ports; see HoleSolver's own note on this).
REM
REM Prerequisite, once (and again after any Program.cs/Util.cs change):
REM   cd Eternity-9-breaks\EternityII_Solver
REM   dotnet build -c Release
REM
REM Env vars honoured (all optional, see Program.cs/Util.cs for what each does):
REM   ETERNITY_NON_CENTER_HINTS   "true" pins the 4 non-center official clues (default off)
REM   ETERNITY_RUN_LABEL          tag printed in the startup banner, e.g. for A/B runs
REM   ETERNITY_NODE_CAP           override the 50e9-node cap
REM   ETERNITY_TUNE_SECONDS       bounded tuning run instead of running forever
REM   ETERNITY_VIRTUAL_CORES      override the 28-thread default (Parallel.For uses N-1)
REM   ETERNITY_BREAK_INDEXES      override the 10-entry general break schedule
REM   ETERNITY_SOLUTIONS_DIR      override the default EternitySolutions_CSharpCPU save folder
REM   ETERNITY_SAVE_FLOOR         override the 12-conflict permanent-save floor

cd /d "%~dp0"

set "DLL=Eternity-9-breaks\EternityII_Solver\bin\Release\net10.0\EternityII_Solver.dll"
if not exist "%DLL%" (
    echo ERROR: %DLL% not found. Run from Eternity-9-breaks\EternityII_Solver:  dotnet build -c Release
    exit /b 1
)

dotnet "%DLL%"
