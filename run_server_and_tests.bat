@echo off
REM Start the emullm relay standalone, wait for it to become ready,
REM run the test suite against it, then stop the server.
REM Usage: run_server_and_tests.bat [extra pytest args]

setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run_server_and_tests] .venv not found. Create it first:
    echo   py -m venv .venv
    echo   .venv\Scripts\python -m pip install -e ".[test]"
    exit /b 1
)

set "HOST=127.0.0.1"
set "PORT=8801"
set "BASE_URL=http://%HOST%:%PORT%"

echo [run_server_and_tests] Starting server on %BASE_URL% ...
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command ^
    "$p = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-m','emullm.standalone','%HOST%','%PORT%' -WindowStyle Hidden -PassThru; $p.Id"`) do set "SERVER_PID=%%P"

if not defined SERVER_PID (
    echo [run_server_and_tests] Failed to start server.
    exit /b 1
)
echo [run_server_and_tests] Server PID %SERVER_PID%

set "READY="
for /l %%I in (1,1,30) do (
    curl -s -o nul -w "%%{http_code}" "%BASE_URL%/v1/models" > "%TEMP%\emullm_status.txt" 2>nul
    set /p STATUS=<"%TEMP%\emullm_status.txt"
    if "!STATUS!"=="200" (
        set "READY=1"
        goto :ready
    )
    timeout /t 1 /nobreak >nul
)

:ready
if not defined READY (
    echo [run_server_and_tests] Server did not become ready in time.
    powershell -NoProfile -Command "Stop-Process -Id %SERVER_PID% -Force -ErrorAction SilentlyContinue"
    exit /b 1
)

echo [run_server_and_tests] Server is ready. Running tests ...
".venv\Scripts\python.exe" -m pytest -q %*
set "TEST_RESULT=%errorlevel%"

echo [run_server_and_tests] Stopping server (PID %SERVER_PID%) ...
powershell -NoProfile -Command "Stop-Process -Id %SERVER_PID% -Force -ErrorAction SilentlyContinue"

exit /b %TEST_RESULT%
