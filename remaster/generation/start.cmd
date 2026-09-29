@echo off
rem Double-click to set up and generate the ConceptLens data. Progress is also saved to log.txt.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
echo.
pause
