# Setup Local Cascade Voice Agent Environment (Windows 11 PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Aarav Local Cascade Voice Agent - Setup Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check Python Environment
$PythonExe = ".\.venv-gpu\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[1/6] Creating Python virtual environment at .venv-gpu..." -ForegroundColor Yellow
    python -m venv .venv-gpu
} else {
    Write-Host "[1/6] Found existing virtual environment at .venv-gpu." -ForegroundColor Green
}

# 2. Upgrade pip and install requirements
Write-Host "[2/6] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r requirements.txt

# 3. Check and start Ollama
Write-Host "[3/6] Checking Ollama installation..." -ForegroundColor Yellow
try {
    $ollamaCheck = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCheck) {
        Write-Host "Ollama command not found in PATH. Please install Ollama from https://ollama.ai/" -ForegroundColor Red
    } else {
        Write-Host "Ollama detected: $($ollamaCheck.Source)" -ForegroundColor Green
        # Ensure server is running
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
            Write-Host "Ollama service is active." -ForegroundColor Green
        } catch {
            Write-Host "Starting Ollama service..." -ForegroundColor Yellow
            Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 3
        }
    }
} catch {
    Write-Host "Notice checking Ollama: $_" -ForegroundColor DarkGray
}

# 4. Pull LLM weights
Write-Host "[4/6] Pulling Qwen2.5 1.5B Instruct model in Ollama..." -ForegroundColor Yellow
ollama pull qwen2.5:1.5b

# 5. Pre-warm and download Faster-Whisper & Kokoro TTS models
Write-Host "[5/6] Pre-caching Faster-Whisper and Kokoro TTS models..." -ForegroundColor Yellow
& $PythonExe -c "
from faster_whisper import WhisperModel
print('Downloading/verifying faster-whisper base model...')
m = WhisperModel('base', device='cpu', compute_type='int8')
from orchestration.tts.base import KokoroTTSBackend
print('Pre-warming Kokoro TTS engine...')
k = KokoroTTSBackend(sample_rate=24000)
k.warm_up()
print('All models cached successfully.')
"

# 6. Run Environment Doctor
Write-Host "`n[6/6] Running system verification doctor..." -ForegroundColor Yellow
& $PythonExe -m orchestration.cli doctor

Write-Host "`nSetup complete! You can now start the voice agent by running:" -ForegroundColor Green
Write-Host "  .\scripts\run_local.ps1" -ForegroundColor Cyan
