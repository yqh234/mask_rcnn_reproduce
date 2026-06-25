param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "== Mask R-CNN project environment setup =="
Write-Host "Python command: $Python"
Write-Host "Virtual environment: $VenvPath"

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment..."
    & $Python -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Cannot find virtual environment Python: $VenvPython"
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing base dependencies..."
& $VenvPython -m pip install -r requirements-base.txt

Write-Host ""
Write-Host "Setup finished."
Write-Host "Next commands:"
Write-Host "  & `"$VenvPython`" environment_report.py"
Write-Host "  & `"$VenvPython`" verify_project.py"
