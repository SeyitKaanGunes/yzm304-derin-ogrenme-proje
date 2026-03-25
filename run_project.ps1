$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\bootstrap.ps1 first."
}

& ".\.venv\Scripts\python.exe" -m src.run_all
