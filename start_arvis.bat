@echo off
REM ============================================================
REM   arvis - one-click launcher (Windows)
REM   Double-click this file to start the voice assistant.
REM ============================================================

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title arvis - Voice Assistant

echo.
echo ============================================
echo   Starting arvis...
echo ============================================
echo.

REM ---- 1. Locate Python ------------------------------------------------
set "PYTHON_CMD="
where python >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=python"
  goto :have_python
)
where py >nul 2>nul
if %errorlevel%==0 (
  set "PYCMD=py -3"
  set "PYTHON_CMD=py -3"
  goto :have_python
)

echo [ERROR] Python was not found on your PATH.
echo Please install Python 3.10+ from https://www.python.org/downloads/
pause
exit /b 1

:have_python
echo [OK] Using Python: %PYTHON_CMD%
%PYTHON_CMD% --version

REM ---- 2. Create virtualenv on first run ------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [INFO] First run - creating virtual environment in .venv ...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

set "VENV_PY=.venv\Scripts\python.exe"

REM ---- 3. Install / update requirements --------------------------------
echo.
echo [INFO] Checking dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [WARN] Some packages failed to install. PyAudio in particular
  echo        sometimes needs Microsoft C++ Build Tools. Continuing anyway...
)

REM ---- 4. Make sure Ollama is running ----------------------------------
where ollama >nul 2>nul
if %errorlevel%==0 (
  echo.
  echo [INFO] Checking Ollama...
  ollama list >nul 2>nul
  if errorlevel 1 (
    echo [INFO] Starting Ollama server...
    start "" ollama serve
    timeout /t 3 /nobreak >nul
  ) else (
    echo [OK] Ollama is already running.
  )
) else (
  echo.
  echo [WARN] Ollama was not found on PATH. If arvis cannot reach the
  echo        model, install it from https://ollama.com and run
  echo        "ollama pull minimax-m3:cloud".
)

REM ---- 5. Launch arvis ------------------------------------------------
echo.
echo ============================================
echo   arvis is starting. Say "arvis" to wake it.
echo   Press Ctrl+C in this window to stop it.
echo ============================================
echo.

".venv\Scripts\python.exe" app.py

echo.
echo ============================================
echo   arvis has stopped.
echo ============================================
echo.
pause
endlocal
