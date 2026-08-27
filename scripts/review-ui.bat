@echo off
REM Local browser UI for the review queue. Nothing is executed.
REM Mark done to drop an item from the morning to-do email.
setlocal
set PYTHONUNBUFFERED=1
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" tools\review_ui.py %*
endlocal
