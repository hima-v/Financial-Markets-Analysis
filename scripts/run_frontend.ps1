$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$py = "py"
$app = ".\frontend\app.py"
$req = ".\frontend\requirements.txt"
$target = ".\frontend\.packages"

try {
  & $py -3.10 -c "import sys; print(sys.executable)" | Out-Null
} catch {
  Write-Host "Python 3.10 is not available via the py launcher."
  Write-Host "Install it (user scope) with:"
  Write-Host "  winget install --id Python.Python.3.10 -e --scope user"
  exit 1
}

& $py -3.10 -m pip install --upgrade pip | Out-Null

$enableUserSite = & $py -3.10 -c "import site; print('1' if site.ENABLE_USER_SITE else '0')"
if ($enableUserSite -eq "1") {
  & $py -3.10 -m pip install --user -r $req
  & $py -3.10 -m streamlit run $app
  exit 0
}

New-Item -ItemType Directory -Force $target | Out-Null
& $py -3.10 -m pip install --target $target -r $req
$env:PYTHONPATH = (Resolve-Path $target).Path
& $py -3.10 -m streamlit run $app

