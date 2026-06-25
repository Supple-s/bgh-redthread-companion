# setup.ps1
# Companion server venv 최초 생성 및 의존성 설치
#
# 사용법:
#   .\setup.ps1                 # basic 의존성만 설치
#   .\setup.ps1 -WithStt        # basic + STT 의존성 함께 설치
#   .\setup.ps1 -WithLocalEmbed # basic + 로컬 임베딩(torch) 의존성 함께 설치
#
# venv가 이미 있으면 건너뛰고 의존성 설치만 갱신합니다.

param(
    [switch]$WithStt,
    [switch]$WithLocalEmbed
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[setup] Working dir: $scriptDir" -ForegroundColor Cyan

# 1. venv 생성 (없을 때만)
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "[setup] Creating .venv..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] FAILED: 'python -m venv .venv' returned $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[setup] .venv already exists, skipping creation." -ForegroundColor Green
}

# 2. venv 활성화
Write-Host "[setup] Activating .venv..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# 3. pip 업그레이드
Write-Host "[setup] Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] WARNING: pip upgrade returned $LASTEXITCODE (continuing)" -ForegroundColor Yellow
}

# 4. basic 의존성 설치
Write-Host "[setup] Installing requirements-basic.txt..." -ForegroundColor Cyan
pip install -r requirements-basic.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] FAILED: requirements-basic.txt install returned $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# 5. STT 의존성 (옵션)
if ($WithStt) {
    Write-Host "[setup] Installing requirements-stt.txt..." -ForegroundColor Cyan
    pip install -r requirements-stt.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] FAILED: requirements-stt.txt install returned $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    Write-Host "[setup] STT 의존성 설치 완료 / STT deps installed. faster-whisper가 PyAV 동봉 디코더를 쓰므로 시스템 ffmpeg 설치는 필요 없습니다 / faster-whisper uses PyAV's bundled decoder, so no system ffmpeg is needed." -ForegroundColor Yellow
}

# 6. 로컬 임베딩 의존성 (옵션) — Voyage 키 없이 임베딩. CPU torch를 끌어오므로 무겁다.
if ($WithLocalEmbed) {
    Write-Host "[setup] Installing CPU torch (배포용: CUDA 휠은 수 GB라 회피 / for distribution: the CUDA wheel is multi-GB, avoided)..." -ForegroundColor Cyan
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] FAILED: CPU torch install returned $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    Write-Host "[setup] Installing requirements-local-embed.txt..." -ForegroundColor Cyan
    pip install -r requirements-local-embed.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] FAILED: requirements-local-embed.txt install returned $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    Write-Host "[setup] 로컬 임베딩 의존성 설치 완료 / local embedding deps installed. .env에 LOCAL_EMBEDDING_ENABLED=true 로 켜세요 / enable with LOCAL_EMBEDDING_ENABLED=true in .env." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[setup] DONE." -ForegroundColor Green
Write-Host "[setup] 이제 .\start.ps1 또는 .\start.ps1 -WithStt 로 서버를 시작할 수 있습니다 / now start the server with .\start.ps1 or .\start.ps1 -WithStt." -ForegroundColor Green
