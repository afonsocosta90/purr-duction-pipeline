@echo off
REM Cross-platform demo launcher - Windows entry point.
REM Usage:  demo.cmd [up|down|logs|reset|fetch-model]   (default: up)
REM Works from PowerShell and cmd with no make or POSIX shell needed.
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (
    python demo\launch.py %*
) else (
    py demo\launch.py %*
)
