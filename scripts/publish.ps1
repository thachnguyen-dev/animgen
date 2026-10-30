# Publishes the app into a DATA folder that already has the runtime (setup_runtime.ps1). Cut C9: a runtime folder
# and this script instead of a real installer.
#
#   powershell -ExecutionPolicy Bypass -File publish.ps1 -Data D:\AnimGen
#
# Result: DATA\app\AnimGen.exe (self-contained, no .NET install needed), DATA\backend\animgen\...
# The whole DATA folder can be copied or moved; the app repairs the venv path offline on its next start.
param(
    [Parameter(Mandatory = $true)][string]$Data
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
$Data = (Resolve-Path $Data).Path

Write-Host '== app'
dotnet publish (Join-Path $repo 'app\AnimGen\AnimGen.csproj') -c Release -r win-x64 --self-contained true `
    -p:PublishSingleFile=false -o (Join-Path $Data 'app')
if ($LASTEXITCODE -ne 0) { throw 'dotnet publish failed' }

Write-Host '== backend'
$dst = Join-Path $Data 'backend\animgen'
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Data 'backend') | Out-Null
Copy-Item (Join-Path $repo 'backend\animgen') $dst -Recurse
Get-ChildItem $dst -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force

foreach ($d in 'weapons', 'temp', 'logs') { New-Item -ItemType Directory -Force -Path (Join-Path $Data $d) | Out-Null }

# Shortcut-friendly launchers
Set-Content -Path (Join-Path $Data 'AnimGen.cmd') -Value "@start """" ""%~dp0app\AnimGen.exe""" -Encoding ASCII
Set-Content -Path (Join-Path $Data 'AnimGen (thu, khong AI).cmd') -Value "@start """" ""%~dp0app\AnimGen.exe"" --demo" -Encoding ASCII
Write-Host 'Done:' (Join-Path $Data 'app\AnimGen.exe')
