@echo off
cd /d "%~dp0"
set /p CPFILE=<cp.txt
"C:\Users\knuda\.jdks\ms-21.0.10\bin\java" -cp "target\classes;%CPFILE%" dk.puzzle.blackwood.BlackwoodSolver
