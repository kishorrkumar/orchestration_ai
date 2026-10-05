<#
.SYNOPSIS
  Universal developer task runner for PersonaPlex Voice Agent Platform (PowerShell-native).

.DESCRIPTION
  Provides standardized commands for setup, dev, test, lint, typecheck, seed, and build.

.EXAMPLE
  .\scripts\run.ps1 setup
  .\scripts\run.ps1 dev
  .\scripts\run.ps1 test
  .\scripts\run.ps1 lint
  .\scripts\run.ps1 check
  .\scripts\run.ps1 seed
#>

param(
    [Parameter(Position=0, Mandatory=$false)]
    [ValidateSet("setup", "dev", "test", "lint", "typecheck", "check", "seed", "build", "help")]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Detect Python interpreter
$PythonExe = Join-Path $RepoRoot ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

function Show-Help {
    Write-Host "`nPersonaPlex Voice Agent Platform - Task Runner`n" -ForegroundColor Cyan
    Write-Host "Usage: .\scripts\run.ps1 [command]`n" -ForegroundColor White
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  setup       Install dependencies and initialize environment" -ForegroundColor White
    Write-Host "  dev         Start the development server with auto-mock worker" -ForegroundColor White
    Write-Host "  test        Run the full automated test suite" -ForegroundColor White
    Write-Host "  lint        Run ruff linter and code formatting check" -ForegroundColor White
    Write-Host "  typecheck   Run mypy strict type checker on domain and application layers" -ForegroundColor White
    Write-Host "  check       Run all gates (lint + typecheck + test)" -ForegroundColor White
    Write-Host "  seed        Seed database with 3 production-grade agents" -ForegroundColor White
    Write-Host "  build       Build production artifacts" -ForegroundColor White
    Write-Host "  help        Display this help message`n" -ForegroundColor White
}

switch ($Task) {
    "setup" {
        Write-Host "--> Verifying dependencies..." -ForegroundColor Cyan
        & $PythonExe -m pip install -e ".[dev]"
        Write-Host "[OK] Environment setup complete." -ForegroundColor Green
    }
    "dev" {
        Write-Host "--> Starting PersonaPlex Voice Agent Platform on http://127.0.0.1:8000" -ForegroundColor Cyan
        Write-Host "    Studio UI:     http://127.0.0.1:8000/" -ForegroundColor White
        Write-Host "    API Docs:      http://127.0.0.1:8000/docs" -ForegroundColor White
        Write-Host "    Voice S2S WS:  ws://127.0.0.1:8000/v2/voice" -ForegroundColor White
        $env:PYTHONPATH = $RepoRoot
        & $PythonExe -m uvicorn orchestration.gateway.app:create_app --factory --host 127.0.0.1 --port 8000 --reload
    }
    "test" {
        Write-Host "--> Running pytest test suite..." -ForegroundColor Cyan
        $env:PYTHONPATH = $RepoRoot
        & $PythonExe -m pytest -v
    }
    "lint" {
        Write-Host "--> Running ruff..." -ForegroundColor Cyan
        & $PythonExe -m ruff check orchestration tests
    }
    "typecheck" {
        Write-Host "--> Running mypy on orchestration..." -ForegroundColor Cyan
        & $PythonExe -m mypy orchestration
    }
    "check" {
        Write-Host "--> Running Quality Gates (lint -> test)..." -ForegroundColor Cyan
        $env:PYTHONPATH = $RepoRoot
        & $PythonExe -m ruff check orchestration tests
        & $PythonExe -m pytest -q
        Write-Host "[OK] All gates passed successfully!" -ForegroundColor Green
    }
    "seed" {
        Write-Host "--> Seeding database with production agents..." -ForegroundColor Cyan
        $env:PYTHONPATH = $RepoRoot
        & $PythonExe -m orchestration.db.seed
        Write-Host "[OK] Database seeded successfully." -ForegroundColor Green
    }
    "build" {
        Write-Host "--> Building Python package..." -ForegroundColor Cyan
        & $PythonExe -m pip install --upgrade build
        & $PythonExe -m build
        Write-Host "[OK] Package build complete." -ForegroundColor Green
    }
    default {
        Show-Help
    }
}
