@echo off
cd /d "%~dp0"
for /f "usebackq delims=" %%A in ("cp.txt") do set CPFILE=%%A
"C:\Users\knuda\.jdks\ms-21.0.10\bin\java" -cp "target\classes;%CPFILE%" dk.puzzle.blackwood.BlackwoodGpuRunner
