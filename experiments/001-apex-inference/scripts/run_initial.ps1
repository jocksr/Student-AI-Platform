$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $experimentRoot)
$python = Join-Path $experimentRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create a Python virtual environment as described in this experiment''s README.md.'
}
function Run-Apex([string[]]$Arguments) {
    & $python -m apex @Arguments
    if ($LASTEXITCODE -ne 0) { throw "APEX command failed: $Arguments" }
}
Push-Location -LiteralPath $experimentRoot
try {
    & $python scripts/run_tests.py
    if ($LASTEXITCODE -ne 0) { throw 'Correctness tests failed' }
    Run-Apex -Arguments @('hardware')
    Run-Apex -Arguments @('benchmark-memory')
    Run-Apex -Arguments @('benchmark-storage')
    Run-Apex -Arguments @('benchmark-gpu')
    Run-Apex -Arguments @('benchmark-compression','--samples','benchmarks/weight-samples')
    Run-Apex -Arguments @('cross-expert')
    Run-Apex -Arguments @('compare')
    Run-Apex -Arguments @('report')
} finally {
    Pop-Location
}
