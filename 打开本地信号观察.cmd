@echo off
setlocal
if not exist "D:\anaconda\pythonw.exe" (
  echo Python runtime not found: D:\anaconda\pythonw.exe
  pause
  exit /b 1
)
start "" "D:\anaconda\pythonw.exe" "%~dp0signal_observer\local_observer.py"
