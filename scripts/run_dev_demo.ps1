# Development: run the WPF app from source with the demo engine (no GPU, no models) and the backend from this
# repository. Needs the .NET 9 SDK and any Python 3.12 with backend\requirements-dev.txt installed.
#
#   powershell -ExecutionPolicy Bypass -File run_dev_demo.ps1 -Python C:\Python312\python.exe
param(
    [Parameter(Mandatory = $true)][string]$Python
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
$data = Join-Path $repo 'devdata'
New-Item -ItemType Directory -Force -Path $data | Out-Null

$env:ANIMGEN_DATA = $data
$env:ANIMGEN_PYTHON = $Python
$env:ANIMGEN_BACKEND = Join-Path $repo 'backend'
dotnet run --project (Join-Path $repo 'app\AnimGen\AnimGen.csproj') -- --demo
