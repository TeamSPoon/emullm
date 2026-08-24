@echo off
REM Run the emullm test suite using the project's virtual environment.
REM Usage: run_tests.bat [extra pytest args]

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run_tests] .venv not found. Create it first:
    echo   py -m venv .venv        ^(or "python -m venv .venv" if "py" is not installed^)
    echo   .venv\Scripts\python -m pip install -e ".[test]"
    exit /b 1
)

REM Use a project-scoped pytest temp root instead of the shared
REM %TEMP%\pytest-of-<user> folder: on a machine shared with other
REM projects/users that folder can end up with broken permissions
REM (e.g. left over from an unrelated test run), which makes every
REM test fail at setup with "PermissionError: Access is denied".
if not defined PYTEST_BASETEMP set "PYTEST_BASETEMP=%TEMP%\emullm-pytest-tmp"

".venv\Scripts\python.exe" -m pytest -q --basetemp="%PYTEST_BASETEMP%" %*
exit /b %errorlevel%
