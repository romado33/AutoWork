@echo off
REM Review queued action items. Nothing is executed without your approval.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" tools\review.py %*
endlocal
