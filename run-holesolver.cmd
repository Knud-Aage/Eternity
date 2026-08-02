@echo off
cd /d "%~dp0"
for /f "usebackq delims=" %%A in ("cp.txt") do set CPFILE=%%A
set LINKFILE=
set TRIALS=
set BASELABEL=
set /p LINKFILE=Path to a text file containing the bucas link:
set /p TRIALS=Trials (press Enter for default 200000):
if "%TRIALS%"=="" set TRIALS=200000
set /p BASELABEL=Base label / piece count, e.g. 206 (optional, press Enter to skip):
"C:\Users\knuda\.jdks\ms-21.0.10\bin\java" -cp "target\classes;%CPFILE%" dk.puzzle.tools.HoleSolver "@%LINKFILE%" %TRIALS% %BASELABEL%
