@echo off
rem Double-click me.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch\bootstrap.ps1" %*
if errorlevel 1 pause
