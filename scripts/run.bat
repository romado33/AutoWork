@echo off
REM Run the AutoWork pipeline now. Plug in the recorder first.
REM Uses the project's own .venv if setup.bat created one, else system python.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
if exist "ffmpeg.exe" set "PATH=%CD%;%PATH%"
"%PY%" tools\run_pipeline.py %*
endlocal
