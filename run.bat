@echo off
chcp 65001 >nul
title Realtime Transcriber
cd /d "%~dp0"
echo ========================================
echo   Realtime Transcriber
echo   Ctrl+C для остановки
echo ========================================
echo.
python main.py %*
pause
