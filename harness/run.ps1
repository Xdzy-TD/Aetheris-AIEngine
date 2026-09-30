<# Aetheris — Windows test harness (PowerShell equivalent of harness/run.sh) #>
$ErrorActionPreference = "Stop"

Write-Host "`n=== Aetheris Test Harness ===" -ForegroundColor Cyan

# 1. Lint check
Write-Host "`n[1/4] Ruff lint..." -ForegroundColor Yellow
python -m ruff check .
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: ruff" -ForegroundColor Red; exit 1 }

# 2. Type check
Write-Host "`n[2/4] Mypy type check..." -ForegroundColor Yellow
python -m mypy controller specialists confidence interfaces
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: mypy" -ForegroundColor Red; exit 1 }

# 3. Tests
Write-Host "`n[3/4] Pytest..." -ForegroundColor Yellow
$env:AETHERIS_NO_LLM = "1"
python -m pytest tests/ -v
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: pytest" -ForegroundColor Red; exit 1 }

# 4. Demo smoke test
Write-Host "`n[4/4] Demo smoke test..." -ForegroundColor Yellow
python run.py demo --no-llm -q "What land cover is visible?"
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: demo" -ForegroundColor Red; exit 1 }

Write-Host "`n=== All checks passed ===" -ForegroundColor Green
