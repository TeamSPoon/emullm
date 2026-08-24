@echo off
REM Installs or updates the Windows tools used by emullm.
REM Run from PowerShell or Command Prompt:
REM   .\update_windows_tools.cmd

setlocal
cd /d "%~dp0"


set "WINGET_EXE=winget"
where winget >nul 2>nul
if errorlevel 1 (
    echo [update] winget was not found. Installing Microsoft App Installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; " ^
        "$bundle=Join-Path $env:TEMP 'Microsoft.DesktopAppInstaller.msixbundle'; " ^
        "Invoke-WebRequest -Uri 'https://github.com/microsoft/winget-cli/releases/latest/download/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle' -OutFile $bundle; " ^
        "Add-AppxPackage -Path $bundle"
    if errorlevel 1 (
        echo [update] Automatic App Installer installation failed.
        echo [update] Install or update "App Installer" from the Microsoft Store,
        echo [update] then run this file again.
        REM exit /b 1
    )

    if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\winget.exe" (
        set "WINGET_EXE=%LOCALAPPDATA%\Microsoft\WindowsApps\winget.exe"
    ) else (
        echo [update] App Installer was installed, but winget is not visible yet.
        echo [update] Close and reopen the terminal, then run this file again.
            REM echo SkIPPING exit /b 1

REM #        exit /b 0
    )
)


echo [update] Ensuring winget is available...
set "WINGET_EXE="
for /f "usebackq delims=" %%W in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ensure_winget.ps1"`) do set "WINGET_EXE=%%W"
if not defined WINGET_EXE (
    echo [update] winget could not be installed or found.
    echo [update] Install or update "App Installer" from the Microsoft Store,
    echo [update] then run this file again.
    REM echo SkIPPING exit /b 1
)


if exist ".venv\Scripts\python.exe" (
    echo [update] Updating the virtual environment and project dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 exit /b 1
    ".venv\Scripts\python.exe" -m pip install --upgrade -e ".[test]"
    if errorlevel 1 exit /b 1
) else (
    echo [update] No .venv found; run install_python.cmd to create it.
)

call :update_package Python.Python.3.12 "Python 3.12"
if errorlevel 1 @echo exit /b 1
call :update_package OpenJS.NodeJS.LTS "Node.js LTS"
if errorlevel 1 exit @echo /b 1
call :update_package BurntSushi.ripgrep.MSVC "ripgrep"
if errorlevel 1 exit @echo /b 1



echo.
echo [update] Finished. Close and reopen your terminal so PATH changes take effect.
exit /b 0

:update_package
echo.
echo [update] Updating %~2...
"%WINGET_EXE%" upgrade --exact --id %~1 --accept-package-agreements --accept-source-agreements
if not errorlevel 1 exit /b 0

echo [update] No upgrade applied; ensuring %~2 is installed...
"%WINGET_EXE%" install --exact --id %~1 --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo [update] Failed to install or update %~2.
    exit /b 1
)
exit /b 0
