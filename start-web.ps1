# Start AINet web chat on all interfaces, port 1111.
# Open: http://127.0.0.1:1111/  or  http://<this-pc-ip>:1111/

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$ollamaDir = "D:\AINet-Tools\Ollama"
if (Test-Path $ollamaDir) {
  $env:PATH = "$ollamaDir;$env:PATH"
}
if (-not $env:OLLAMA_MODELS -and (Test-Path "D:\AINet-Tools\ollama-models")) {
  $env:OLLAMA_MODELS = "D:\AINet-Tools\ollama-models"
}
if (-not $env:AINET_OLLAMA_MODEL) {
  $env:AINET_OLLAMA_MODEL = "qwen3:8b"
}
# Always pin AINet chat to qwen3:8b (never inherit a stale llama env).
$env:AINET_OLLAMA_MODEL = "qwen3:8b"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
  $pyPath = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\python.exe"
  if (Test-Path $pyPath) { Set-Alias -Name python -Value $pyPath }
  else { throw "Python not found on PATH" }
}

# Best-effort: wake Ollama if installed but not answering
try {
  $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434" -UseBasicParsing -TimeoutSec 2
} catch {
  $app = Join-Path $ollamaDir "ollama app.exe"
  if (Test-Path $app) {
    Start-Process -FilePath $app | Out-Null
    Start-Sleep -Seconds 3
  }
}

$ip = (
  Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
  Select-Object -First 1 -ExpandProperty IPAddress
)
Write-Host "AINet web starting..."
Write-Host "  Local:  http://127.0.0.1:1111/"
if ($ip) { Write-Host "  LAN:    http://${ip}:1111/" }

python -m ollama web --bind 0.0.0.0 --port 1111 @args
