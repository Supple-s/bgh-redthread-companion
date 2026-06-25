@echo off
REM Double-click launcher for start-voice-tunnels.ps1
REM Foundry(30000) + STT(5000) Quick Tunnels in one window.
REM Reads the no-BOM UTF-8 script as UTF-8 and sets the console to UTF-8 so
REM Korean output is not mojibaked on Windows PowerShell 5.1 (which assumes ANSI for no-BOM files).
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; & ([scriptblock]::Create((Get-Content -Raw -Encoding UTF8 '%~dp0start-voice-tunnels.ps1')))"
pause
