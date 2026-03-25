$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv")) {
    try {
        py -3 -m venv .venv
    } catch {
        python -m venv .venv
    }
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
