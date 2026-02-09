$ErrorActionPreference = "Stop"

Write-Host "py launcher versions:"
py -0p

Write-Host ""
Write-Host "Python 3.11 user-site flag:"
py -3.11 -c "import sys,site; print(sys.executable); print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE)"

Write-Host ""
Write-Host "Python 3.10 user-site flag (if installed):"
try {
  py -3.10 -c "import sys,site; print(sys.executable); print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE)"
} catch {
  Write-Host "Python 3.10 not available."
}

