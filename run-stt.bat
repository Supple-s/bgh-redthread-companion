@echo off
REM run-stt.bat - one-click STT backend launcher for beta testers.
REM First double-click: creates the Python venv, installs STT deps, writes .env (STT on).
REM Every double-click after that: just starts the server. No PowerShell knowledge needed.
REM (All messages are English on purpose: .bat text under the Windows console codepage
REM  garbles non-ASCII. Korean stays in README/comments elsewhere.)
setlocal
cd /d "%~dp0"

echo ===============================================
echo   BGH Red Thread - STT Backend (one-click)
echo ===============================================
echo.

REM --- 1. Python present? ---
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found on this PC.
  echo.
  echo   Install Python 3.10 - 3.12 from:
  echo     https://www.python.org/downloads/
  echo   IMPORTANT: on the first install screen, tick
  echo     "Add python.exe to PATH"
  echo.
  echo   Then double-click this file again.
  echo.
  pause
  exit /b 1
)

REM --- 2. First run: create venv + install deps (basic + STT) ---
if not exist ".venv\Scripts\python.exe" (
  echo [setup] First run detected.
  echo [setup] Creating the environment and installing STT dependencies.
  echo [setup] This downloads packages - it can take several minutes. Please wait.
  echo.
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -WithStt
  if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed. Read the messages above, then try again.
    pause
    exit /b 1
  )
)

REM --- 3. Ensure .env exists with STT enabled ---
if not exist ".env" (
  echo [setup] Creating .env from the template and enabling STT...
  copy /y ".env.example" ".env" >nul
  python -c "import io,re;p='.env';s=io.open(p,encoding='utf-8').read();s=re.sub(r'(?m)^STT_ENABLED=.*','STT_ENABLED=true',s) if re.search(r'(?m)^STT_ENABLED=',s) else s+'\nSTT_ENABLED=true\n';io.open(p,'w',encoding='utf-8',newline='\n').write(s)"
)

REM --- 4. Start the server ---
echo.
echo [start] Starting the STT server...
echo [start] Health check:  http://127.0.0.1:5000/health
echo [start] Keep this window open. Close it (or press Ctrl+C) to stop.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-stt.ps1"

echo.
echo [stopped] The server has stopped.
pause
