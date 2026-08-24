@echo off
REM install_python.cmd
REM Bootstraps Python 3.12+ on Windows for the emullm project:
REM   1. Ensures winget is available (installs App Installer if missing).
REM   2. Installs Python 3.12 via winget if not already present.
REM   3. Creates the project's .venv virtual environment.
REM   4. Installs the project (with test extras) into that venv.
REM
REM Run this from the project root, e.g. by double-clicking it or:
REM   install_python.cmd

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [install_python] Ensuring winget is available ...
set "WINGET_EXE="
for /f "usebackq delims=" %%W in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ensure_winget.ps1"`) do set "WINGET_EXE=%%W"
if not defined WINGET_EXE (
    echo [install_python] winget could not be installed or found.
    echo [install_python] Install "App Installer" from the Microsoft Store, then re-run this script.
    exit /b 1
)
echo [install_python] winget is available: %WINGET_EXE%

echo [install_python] Checking for the Python Launcher (py) ...
where py >nul 2>nul
if errorlevel 1 (
    echo [install_python] py not found. Installing Python 3.12 via winget ...
    "%WINGET_EXE%" install --exact --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [install_python] winget install failed. Install Python manually from
        echo   https://www.python.org/downloads/windows/
        exit /b 1
    )
    echo [install_python] Python installed. Close and reopen this terminal, then re-run this script
    echo   to continue with virtual environment setup.
    exit /b 0
)

echo [install_python] Verifying Python version ...
py --version
py -0p

echo [install_python] Creating virtual environment in .venv ...
if exist ".venv\Scripts\python.exe" (
    echo [install_python] .venv already exists, skipping creation.
) else (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo [install_python] Failed to create .venv with py -3.12; trying default py ...
        py -m venv .venv
        if errorlevel 1 (
            echo [install_python] Failed to create virtual environment.
            exit /b 1
        )
    )
)

echo [install_python] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [install_python] Installing project with test extras ...
".venv\Scripts\python.exe" -m pip install -e ".[test]"
if errorlevel 1 (
    echo [install_python] pip install failed.
    exit /b 1
)

echo.
echo [install_python] Done. Activate the environment with:
echo   .venv\Scripts\Activate.ps1   (PowerShell)
echo   .venv\Scripts\activate.bat   (cmd.exe)
exit /b 0
