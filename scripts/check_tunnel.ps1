# Verify AINet is reachable for the Cloudflare tunnel.
# Usage (from anywhere):
#   powershell -File "C:\Users\Hayden Supple\Desktop\AINet\scripts\check_tunnel.ps1" -PublicUrl https://pathroom.org

param(
  [string]$PublicUrl = $env:AINET_PUBLIC_URL,
  [string]$LocalUrl = "http://127.0.0.1:1111"
)

$ErrorActionPreference = "Continue"
$cfExe = "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe"
$tokenPath = "C:\ProgramData\cloudflared\token"

Write-Host "=== Cloudflared service ===" -ForegroundColor Cyan
$svc = Get-Service Cloudflared -ErrorAction SilentlyContinue
if (-not $svc) {
  Write-Host "FAIL: Cloudflared Windows service not installed." -ForegroundColor Red
  exit 1
}
Write-Host ("Service: {0}  StartType={1}" -f $svc.Status, $svc.StartType)
if ($svc.Status -ne "Running") {
  Write-Host "Starting Cloudflared..." -ForegroundColor Yellow
  Start-Service Cloudflared
  Start-Sleep -Seconds 2
  $svc.Refresh()
  Write-Host ("Service now: {0}" -f $svc.Status)
}

Write-Host ""
Write-Host "=== Tunnel agent ready ===" -ForegroundColor Cyan
try {
  $ready = Invoke-RestMethod -Uri "http://127.0.0.1:20241/ready" -TimeoutSec 3
  Write-Host ("readyConnections={0} connectorId={1}" -f $ready.readyConnections, $ready.connectorId)
  if (-not $ready.readyConnections -or $ready.readyConnections -lt 1) {
    Write-Host "WARN: tunnel process up but no edge connections yet." -ForegroundColor Yellow
  }
} catch {
  Write-Host "FAIL: cloudflared ready endpoint not responding on :20241" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Local chat origin ===" -ForegroundColor Cyan
try {
  $status = Invoke-RestMethod -Uri "$LocalUrl/api/status" -TimeoutSec 5
  Write-Host ("OK {0}  model={1} mode={2}" -f $LocalUrl, $status.model, $status.mode)
} catch {
  Write-Host "FAIL: chat not up at $LocalUrl" -ForegroundColor Red
  Write-Host "Start with: python -m ollama web --bind 127.0.0.1 --port 1111" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Token / tunnel id ===" -ForegroundColor Cyan
if (Test-Path $tokenPath) {
  try {
    $raw = (Get-Content -LiteralPath $tokenPath -Raw -ErrorAction Stop).Trim()
    $parts = $raw.Split(".")
    $payloadB64 = if ($parts.Count -ge 2) { $parts[1] } else { $parts[0] }
    while ($payloadB64.Length % 4) { $payloadB64 += "=" }
    $payloadB64 = $payloadB64.Replace("-", "+").Replace("_", "/")
    $json = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($payloadB64))
    $obj = $json | ConvertFrom-Json
    Write-Host ("tunnel id: {0}" -f $obj.t)
    Write-Host "token file: $tokenPath (not in git)"
  } catch {
    Write-Host "token file exists (SYSTEM-only). Re-run in Admin PowerShell to print tunnel id." -ForegroundColor Yellow
  }
} else {
  Write-Host "WARN: missing $tokenPath" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Public URL ===" -ForegroundColor Cyan
if ($PublicUrl) {
  $PublicUrl = $PublicUrl.TrimEnd("/")
  try {
    $pub = Invoke-RestMethod -Uri "$PublicUrl/api/status" -TimeoutSec 20
    Write-Host ("OK {0}  model={1} mode={2}" -f $PublicUrl, $pub.model, $pub.mode)
  } catch {
    Write-Host ("FAIL hitting {0}: {1}" -f $PublicUrl, $_) -ForegroundColor Red
    Write-Host "Cloudflare Zero Trust -> Networks -> Tunnels -> Public Hostname:" -ForegroundColor Yellow
    Write-Host "  hostname = pathroom.org (or subdomain)" -ForegroundColor Yellow
    Write-Host "  service  = http://127.0.0.1:1111" -ForegroundColor Yellow
  }
} else {
  Write-Host "Pass -PublicUrl https://pathroom.org to probe the edge."
}

Write-Host ""
Write-Host "Mac: open that HTTPS URL (same UI/API as localhost:1111)." -ForegroundColor Green
if (Test-Path $cfExe) {
  $ver = & $cfExe --version 2>&1 | Select-Object -First 1
  Write-Host ("cloudflared: {0}" -f $ver)
}
