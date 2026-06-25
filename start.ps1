# start.ps1
# Companion server 시작 스크립트
#
# 사용법:
#   .\start.ps1                 # 일반 시작 (basic 모드)
#   .\start.ps1 -WithStt        # STT 활성화 안내 출력 후 시작
#   .\start.ps1 -WithLocalEmbed # 로컬 임베딩 의존성 확인 후 시작
#
# 매번 새 PowerShell 창에서 이 스크립트만 실행하면 됩니다.
# venv는 setup.ps1에서 미리 만들어 둬야 합니다.

param(
    [switch]$WithStt,
    [switch]$WithLocalEmbed
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[start] Working dir: $scriptDir" -ForegroundColor Cyan

# 1. venv 존재 확인
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "[start] FAILED: .venv가 없습니다 / .venv not found. 먼저 .\setup.ps1 을 실행하세요 / run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# 2. venv 활성화
Write-Host "[start] Activating .venv..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# 3. flask 설치 확인 (basic 의존성 sanity check)
python -c "import flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[start] FAILED: flask가 설치돼 있지 않습니다 / flask is not installed. .\setup.ps1 을 실행하세요 / run .\setup.ps1." -ForegroundColor Red
    exit 1
}

# 4. STT 사용 시 의존성 확인
if ($WithStt) {
    python -c "import faster_whisper" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[start] FAILED: STT 모드인데 faster_whisper가 없습니다 / STT mode but faster_whisper is missing. .\setup.ps1 -WithStt 를 실행하세요 / run .\setup.ps1 -WithStt." -ForegroundColor Red
        exit 1
    }
    Write-Host "[start] STT 모드 활성 / STT mode on. .env의 STT_ENABLED=true 인지 확인하세요 / check STT_ENABLED=true in .env." -ForegroundColor Yellow
}

# 4b. 로컬 임베딩 사용 시 의존성 확인
if ($WithLocalEmbed) {
    python -c "import sentence_transformers" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[start] FAILED: 로컬 임베딩 모드인데 sentence-transformers가 없습니다 / local embedding mode but sentence-transformers is missing. .\setup.ps1 -WithLocalEmbed 를 실행하세요 / run .\setup.ps1 -WithLocalEmbed." -ForegroundColor Red
        exit 1
    }
    Write-Host "[start] 로컬 임베딩 모드 활성 / local embedding mode on. .env의 LOCAL_EMBEDDING_ENABLED=true 인지 확인하세요 / check LOCAL_EMBEDDING_ENABLED=true in .env." -ForegroundColor Yellow
}

# 5. 서버 시작
Write-Host ""
Write-Host "[start] Starting companion_server.py..." -ForegroundColor Green
Write-Host "[start] Health check: http://127.0.0.1:5000/health" -ForegroundColor Green
Write-Host "[start] Stop with Ctrl+C" -ForegroundColor Green
Write-Host ""

python companion_server.py
