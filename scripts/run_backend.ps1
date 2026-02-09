$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

py -3.10 -m pip install -r .\backend\requirements.txt
py -3.10 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

