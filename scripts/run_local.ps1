# Run Local Cascade Voice Agent Gateway (Windows 11 PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Starting Aarav Voice Agent (Local Cascade Pipeline)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Ensure Ollama is running
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
    Write-Host "[✓] Ollama server is active on port 11434." -ForegroundColor Green
} catch {
    Write-Host "[!] Starting background Ollama daemon..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

# 2. Start Gateway Server with local_cascade as default worker
$PythonExe = ".\.venv-gpu\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "`nLaunching Gateway with local_cascade worker..." -ForegroundColor Green
Write-Host "  Developer Console: http://127.0.0.1:8000/console" -ForegroundColor Cyan
Write-Host "  Real-time Socket:  ws://127.0.0.1:8000/v1/realtime" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to terminate.`n" -ForegroundColor DarkGray

$env:PYTHONPATH = "."
& $PythonExe -m orchestration.cli run-gateway --worker-type local_cascade --port 8000 $args
