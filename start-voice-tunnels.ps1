# start-voice-tunnels.ps1
# BGH 보이스 Quick Tunnel 런처 — Foundry(30000) + STT(5000)를 동시에 HTTPS로 노출.
#
# 더블클릭 한 번으로 Quick Tunnel 2개를 띄우고:
#   [1] 플레이어에게 공유할 Foundry HTTPS 주소
#   [2] 음성 입력 패널(GM 설정)의 백엔드 URL에 넣을 STT 주소(/stt 포함, 클립보드 자동복사)
# 를 정리해 보여준다.
#
# 선행: Foundry(30000)와 start-stt.ps1(STT 5000)이 먼저 실행 중이어야 한다.
# 종료: 이 창에서 Ctrl+C 또는 창 닫기 → 터널 2개 함께 종료.
#
# 참고: Quick Tunnel은 재시작마다 URL이 바뀐다. 매 세션 [1]을 플레이어에게 공유하고
#       [2]를 voiceInputBackendUrl에 갱신하면 된다(클립보드 복사로 붙여넣기만).

$ErrorActionPreference = "Stop"

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "[tunnels] cloudflared 미설치 / not installed. 설치 / install: winget install --id Cloudflare.cloudflared" -ForegroundColor Red
    exit 1
}

function Test-LocalPort([int]$port) {
    try {
        $client = New-Object Net.Sockets.TcpClient
        $async = $client.BeginConnect("127.0.0.1", $port, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne(500)
        $client.Close()
        return $ok
    } catch {
        return $false
    }
}

foreach ($svc in @(@{ Port = 30000; Name = "Foundry" }, @{ Port = 5000; Name = "STT" })) {
    if (-not (Test-LocalPort $svc.Port)) {
        Write-Host "[tunnels] 경고 / Warning: localhost:$($svc.Port) ($($svc.Name)) 응답 없음 / not responding — 먼저 실행했는지 확인하세요 / start it first." -ForegroundColor Yellow
    }
}

$outFoundry = Join-Path $env:TEMP "bgh-tunnel-foundry.out"
$errFoundry = Join-Path $env:TEMP "bgh-tunnel-foundry.err"
$outStt = Join-Path $env:TEMP "bgh-tunnel-stt.out"
$errStt = Join-Path $env:TEMP "bgh-tunnel-stt.err"
Remove-Item $outFoundry, $errFoundry, $outStt, $errStt -ErrorAction SilentlyContinue

Write-Host "[tunnels] Quick Tunnel 2개 시작 중... / Starting 2 Quick Tunnels..." -ForegroundColor Cyan
$procFoundry = Start-Process cloudflared `
    -ArgumentList "tunnel", "--url", "http://localhost:30000" `
    -RedirectStandardOutput $outFoundry -RedirectStandardError $errFoundry `
    -WindowStyle Hidden -PassThru
$procStt = Start-Process cloudflared `
    -ArgumentList "tunnel", "--url", "http://localhost:5000" `
    -RedirectStandardOutput $outStt -RedirectStandardError $errStt `
    -WindowStyle Hidden -PassThru

function Wait-TunnelUrl($outPath, $errPath) {
    for ($i = 0; $i -lt 60; $i++) {
        foreach ($path in @($errPath, $outPath)) {
            if (Test-Path $path) {
                $match = Select-String -Path $path -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($match) {
                    return $match.Matches[0].Value
                }
            }
        }
        Start-Sleep -Milliseconds 500
    }
    return $null
}

try {
    $urlFoundry = Wait-TunnelUrl $outFoundry $errFoundry
    $urlStt = Wait-TunnelUrl $outStt $errStt

    if (-not $urlFoundry -or -not $urlStt) {
        Write-Host "[tunnels] URL 확보 실패 / Failed to get URL. cloudflared 로그를 확인하세요 / check the cloudflared logs:" -ForegroundColor Red
        Write-Host "  $errFoundry" -ForegroundColor DarkGray
        Write-Host "  $errStt" -ForegroundColor DarkGray
        throw "tunnel url not found"
    }

    $sttBackendUrl = "$urlStt/stt"
    try { Set-Clipboard -Value $sttBackendUrl } catch { }

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host " 보이스 터널 준비 완료 / Voice tunnels ready" -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host " [1] 플레이어에게 공유할 Foundry 주소 (디스코드 등) / Foundry address to share with players (Discord, etc.):" -ForegroundColor Cyan
    Write-Host "     $urlFoundry" -ForegroundColor White
    Write-Host ""
    Write-Host " [2] 음성 입력 패널(GM 설정) > 백엔드 URL 에 붙여넣기 / paste into Voice Input panel (GM settings) > Backend URL:" -ForegroundColor Cyan
    Write-Host "     $sttBackendUrl   (클립보드에 복사됨 / copied to clipboard)" -ForegroundColor White
    Write-Host ""
    Write-Host " * Foundry options.json 에 proxySSL=true, proxyPort=443 설정 필요 / required." -ForegroundColor DarkGray
    Write-Host " * 이 창을 닫거나 Ctrl+C 하면 터널 2개가 함께 종료됩니다 / Closing this window or Ctrl+C stops both tunnels." -ForegroundColor DarkGray
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "[tunnels] 실행 중... / Running... (종료 / exit: Ctrl+C)" -ForegroundColor Cyan

    while (-not $procFoundry.HasExited -and -not $procStt.HasExited) {
        Start-Sleep -Seconds 2
    }
    Write-Host "[tunnels] 터널 프로세스가 종료되었습니다 / Tunnel processes have stopped." -ForegroundColor Yellow
}
finally {
    foreach ($proc in @($procFoundry, $procStt)) {
        if ($proc -and -not $proc.HasExited) {
            try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
        }
    }
}
