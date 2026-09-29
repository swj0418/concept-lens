# Generate the experiments, extract CLIP features and export the Observable bundle (to ..\bundle\).
#
#   powershell -ExecutionPolicy Bypass -File .\run.ps1             FFHQ 256, early layers, 200 codes,
#                                                                  SeFA + GANSpace + vector arithmetic + SVM, 48 directions each
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 --parity    also check the JavaScript port against the original server
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 --help      every option
#
# An interrupted run picks up where it stopped.
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run setup.ps1 first." -ForegroundColor Red; exit 1 }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"
& $py (Join-Path $PSScriptRoot "generate.py") @args
exit $LASTEXITCODE
