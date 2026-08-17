@echo off
REM ============================================================
REM   arvis - terminal launcher (used by Windows autostart)
REM
REM   This script opens a visible console window, cd's into the
REM   project folder, then runs ``app.py`` under the project's
REM   virtualenv (or the system ``python`` if the venv is missing).
REM
REM   The previous behaviour was a silent ``pythonw.exe`` launch,
REM   which made debugging Windows-startup failures impossible.
REM   With this launcher the user can see ``cd`` and the actual
REM   python error if anything goes wrong.
REM ============================================================

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title arvis - Voice Assistant (autostart)

REM Pick the best Python interpreter we can find.
set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" (
  set "PYTHON_CMD=.venv\Scripts\python.exe"
  goto :have_python
)
where python >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=python"
  goto :have_python
)
where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3"
  goto :have_python
)

echo [ERROR] Python was not found on your PATH.
echo Please install Python 3.10+ from https://www.python.org/downloads/
echo Press any key to close this window.
pause >nul
exit /b 1

:have_python
echo.
echo ============================================
echo   arvis (Windows autostart)
echo   Project folder: %cd%
echo   Python:         %PYTHON_CMD%
echo ============================================
echo.

REM Pass ``--startup`` so arvis can detect that it was launched by the
REM Run key and run the configured startup URLs/apps.
"%PYTHON_CMD%" app.py --startup

set "EXITCODE=%errorlevel%"
echo.
echo ============================================
echo   arvis exited with code %EXITCODE%.
echo   Press any key to close this window.
echo ============================================
pause >nul
endlocal
