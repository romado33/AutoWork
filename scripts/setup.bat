@echo off
REM ============================================================================
REM AutoWork setup. Run once after unzipping.
REM
REM Creates a local virtual environment inside the project folder and installs
REM the Python dependencies into it. The venv lives in the project directory so
REM the whole thing stays self-contained and can be moved or deleted as a unit.
REM
REM PREREQUISITES this cannot install for you:
REM   * Python 3.11+   https://www.python.org/downloads/  (tick "Add to PATH")
REM   * ffmpeg         https://www.gyan.dev/ffmpeg/builds/ (the "essentials" build)
REM                    ffmpeg.exe must be on PATH, or dropped into this folder.
REM
REM ffmpeg is not optional: the audio quality gate and the slicing/compression
REM step are both ffmpeg. Without it nothing downstream runs.
REM ============================================================================
setlocal
cd /d "%~dp0.."

echo.
echo === AutoWork setup ===
echo.

REM ---- Python -----------------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python is not on PATH.
    echo        Install Python 3.11+ and tick "Add python.exe to PATH".
    exit /b 2
)
for /f "delims=" %%v in ('python -c "import sys;print(sys.version.split()[0])"') do set PYVER=%%v
echo Python %PYVER%

python -c "import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.11 or newer is required, found %PYVER%.
    exit /b 2
)

REM ---- ffmpeg -----------------------------------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    if exist "%CD%\ffmpeg.exe" (
        echo ffmpeg: found in project folder
        set "PATH=%CD%;%PATH%"
    ) else (
        echo ERROR: ffmpeg is not on PATH and not in this folder.
        echo        Download the "essentials" build from
        echo          https://www.gyan.dev/ffmpeg/builds/
        echo        and either add it to PATH or copy ffmpeg.exe next to this script's parent.
        exit /b 2
    )
) else (
    echo ffmpeg: on PATH
)

REM ---- virtual environment ----------------------------------------------------
if not exist ".venv" (
    echo Creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment.
        exit /b 1
    )
) else (
    echo .venv already exists, reusing it
)

echo Installing dependencies ...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency install failed.
    exit /b 1
)

REM ---- .env -------------------------------------------------------------------
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo Created .env from the template -- EDIT IT before running.
    )
) else (
    echo .env already present, leaving it alone
)

REM ---- verify -----------------------------------------------------------------
echo.
echo Verifying ...
".venv\Scripts\python.exe" -m pytest tests -q
if errorlevel 1 (
    echo WARNING: tests did not all pass. The install may still be usable.
) else (
    echo Tests pass.
)

echo.
echo === Setup complete ===
echo.
echo Next:
echo   1. Edit .env and put your OPENAI_API_KEY in it.
echo   2. For summary emails, add SMTP_ADDRESS, SMTP_APP_PASSWORD and SUMMARY_TO.
echo      SMTP_APP_PASSWORD must be a Google App Password, not your Gmail password:
echo        https://myaccount.google.com/apppasswords
echo   3. Plug in the recorder and run:  scripts\run.bat
echo   4. To make it automatic on login:
echo        powershell -ExecutionPolicy Bypass -File scripts\Install-Watcher.ps1
echo.
endlocal
