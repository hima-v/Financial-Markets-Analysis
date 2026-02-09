$ErrorActionPreference = "Stop"

$repo = (Split-Path -Parent $PSScriptRoot)

Write-Host "Starting backend and frontend..."
Write-Host "Repo: $repo"

$backendCmd = "Set-Location '$repo'; .\scripts\run_backend.ps1"
$frontendCmd = "Set-Location '$repo'; .\scripts\run_frontend.ps1"

Start-Process powershell -WorkingDirectory $repo -ArgumentList "-NoExit", "-Command", $backendCmd | Out-Null
Start-Process powershell -WorkingDirectory $repo -ArgumentList "-NoExit", "-Command", $frontendCmd | Out-Null

Write-Host "Opened two PowerShell windows."
