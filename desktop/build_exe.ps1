# Build AINet.exe
# Run from repo root:
#   powershell -File desktop/build_exe.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "Python not found on PATH" }

Write-Host "Installing desktop packaging deps..."
python -m pip install -r desktop/requirements-desktop.txt

$dist = Join-Path (Get-Location) "dist"
$work = Join-Path (Get-Location) "build\ainet-desktop"
New-Item -ItemType Directory -Force -Path $dist | Out-Null
New-Item -ItemType Directory -Force -Path $work | Out-Null

Write-Host "Building AINet.exe (one-folder)..."
python -m PyInstaller `
  --noconfirm `
  --clean `
  --name AINet `
  --windowed `
  --paths . `
  --add-data "desktop/static;desktop/static" `
  --add-data "ollama/static;ollama/static" `
  --hidden-import desktop `
  --hidden-import desktop.app `
  --hidden-import desktop.shell_server `
  --hidden-import ollama `
  --hidden-import ollama.webserver `
  --hidden-import ollama.soi_test_server `
  --hidden-import ollama.soi_test_app `
  --hidden-import ainet `
  --distpath $dist `
  --workpath $work `
  desktop/app.py

Write-Host ""
Write-Host "Done."
Write-Host "  Executable folder: $dist\AINet\"
Write-Host "  Run:               $dist\AINet\AINet.exe"
Write-Host ""
Write-Host "Note: keep the repo db/ nearby or set AINET_DB to your database path."
