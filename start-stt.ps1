# start-stt.ps1
# Companion server STT 모드 시작 스크립트
#
# 사용법:
#   .\start-stt.ps1     # STT 활성화 모드로 즉시 시작
#
# 내부적으로 start.ps1 -WithStt와 동일하게 동작합니다.
# 바탕화면 바로가기에서 더블클릭으로 바로 실행할 수 있게 별도 파일로 분리한 버전입니다.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[start-stt] Working dir: $scriptDir" -ForegroundColor Cyan

# 1. venv 존재 확인
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "[start-stt] FAILED: .venv가 없습니다. 먼저 .\setup.ps1 -WithStt 를 실행하세요." -ForegroundColor Red
    exit 1
}

# 2. venv 활성화
Write-Host "[start-stt] Activating .venv..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# 3. flask 설치 확인
python -c "import flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[start-stt] FAILED: flask가 설치돼 있지 않습니다. .\setup.ps1 -WithStt 를 실행하세요." -ForegroundColor Red
    exit 1
}

# 4. STT 의존성 확인 (faster_whisper)
python -c "import faster_whisper" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[start-stt] FAILED: faster_whisper가 설치돼 있지 않습니다. .\setup.ps1 -WithStt 를 실행하세요." -ForegroundColor Red
    exit 1
}

Write-Host "[start-stt] STT 모드 활성. .env의 STT_ENABLED=true 인지 확인하세요." -ForegroundColor Yellow

# 5. 서버 시작
Write-Host ""
Write-Host "[start-stt] Starting companion_server.py (STT enabled)..." -ForegroundColor Green
Write-Host "[start-stt] Health check: http://127.0.0.1:5000/health" -ForegroundColor Green
Write-Host "[start-stt] Stop with Ctrl+C" -ForegroundColor Green
Write-Host ""

python companion_server.py
