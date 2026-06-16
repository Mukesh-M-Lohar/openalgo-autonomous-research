# OpenAlgo Quant Research Engine — Dev Setup (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "=== OpenAlgo Quant Research Engine — Dev Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python
$python = "python"
try {
    $version = & $python --version 2>&1
    Write-Host "Python: $version"
} catch {
    Write-Host "ERROR: Python not found. Install Python 3.11+ from python.org" -ForegroundColor Red
    exit 1
}

$versionParts = ($version -replace "Python ", "") -split "\."
if ([int]$versionParts[0] -lt 3 -or ([int]$versionParts[0] -eq 3 -and [int]$versionParts[1] -lt 11)) {
    Write-Host "ERROR: Python 3.11+ required. Found $version" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "1. Creating virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path ".venv")) {
    & $python -m venv .venv
    Write-Host "   Created .venv/"
} else {
    Write-Host "   .venv/ already exists, skipping"
}

# Activate
Write-Host ""
Write-Host "2. Activating environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Install dependencies
Write-Host ""
Write-Host "3. Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip -q
try {
    pip install -e ".[dev]" -q
} catch {
    pip install numpy pandas pydantic pydantic-settings pyyaml fastapi uvicorn typer jinja2 httpx pytest pytest-cov ruff -q
}
Write-Host "   Done."

# Verify
Write-Host ""
Write-Host "4. Verifying installation..." -ForegroundColor Yellow
& $python -c "import sys; sys.path.insert(0,'src'); import quant_engine; print(f'   quant_engine v{quant_engine.__version__}')"

# Run tests
Write-Host ""
Write-Host "5. Running tests..." -ForegroundColor Yellow
$env:PYTHONPATH = "src"
& $python -m pytest tests/ -q --tb=line

# Create data directories
Write-Host ""
Write-Host "6. Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path data/cache, data/runs, data/exports | Out-Null
Write-Host "   Created data/{cache,runs,exports}/"

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Activate your environment:"
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Quick commands:" -ForegroundColor Yellow
Write-Host "  make test          — Run tests"
Write-Host "  make lint          — Check code style"
Write-Host "  make serve-dev     — Start API server"
Write-Host "  make run-default   — Run with default config"
