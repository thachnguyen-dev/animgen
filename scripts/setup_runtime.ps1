# Builds the Python runtime and downloads the models into a DATA folder (run once, on the owner's PC, WITH internet).
# After this, everything runs offline.
#
#   powershell -ExecutionPolicy Bypass -File setup_runtime.ps1 -Data D:\AnimGen -Uv C:\path\to\uv.exe
#
# Layout produced (§9.1):
#   DATA\uv.exe      bundled uv (used offline after a move of the folder)
#   DATA\python\     Python 3.12 managed by uv
#   DATA\venv\       virtual environment with PyTorch (CUDA 12.6), Diffusers, ...
#   DATA\models\     model folders (scripts/download_models.py)
param(
    [Parameter(Mandatory = $true)][string]$Data,
    [Parameter(Mandatory = $true)][string]$Uv,
    [switch]$SkipModels
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here

New-Item -ItemType Directory -Force -Path $Data | Out-Null
$Data = (Resolve-Path $Data).Path
Copy-Item $Uv (Join-Path $Data 'uv.exe') -Force
$uvExe = Join-Path $Data 'uv.exe'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Data 'python'
$env:UV_NO_CONFIG = '1'

Write-Host '== Python 3.12'
& $uvExe python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'uv python install failed' }
$py = Get-ChildItem (Join-Path $Data 'python') -Directory -Filter 'cpython-3.12*' | Select-Object -First 1
if (-not $py) { throw 'Python 3.12 not found after install' }
$pyExe = Join-Path $py.FullName 'python.exe'

Write-Host '== venv'
& $uvExe venv (Join-Path $Data 'venv') --python $pyExe --allow-existing
if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
$venvPy = Join-Path $Data 'venv\Scripts\python.exe'

Write-Host '== packages'
& $uvExe pip install --python $venvPy torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'torch install failed' }
& $uvExe pip install --python $venvPy -r (Join-Path $repo 'backend\requirements-gpu.txt') --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'requirements install failed' }

& $venvPy -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

if (-not $SkipModels) {
    Write-Host '== models (about 15 GB)'
    & $venvPy (Join-Path $here 'download_models.py') (Join-Path $Data 'models')
    if ($LASTEXITCODE -ne 0) { throw 'model download failed' }
}
Write-Host 'Done. Next: scripts\publish.ps1 -Data' $Data
