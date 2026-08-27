@echo off
REM Email the outstanding action queue (morning digest). Nothing is executed.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" tools\send_queue_digest.py %*
endlocal
