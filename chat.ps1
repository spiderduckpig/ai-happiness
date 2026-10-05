[CmdletBinding()]
param(
    [string]$Run,
    [ValidateSet('auto', 'cpu', 'cuda', 'mps')][string]$Device = 'auto'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Push-Location -LiteralPath $PSScriptRoot
try {
    $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        throw 'Project environment is missing. Run start.ps1 to install it first.'
    }
    & $taskPython 'setup_support.py' 'preflight' '--stage' 'run'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $taskChatArgs = @('-u', '-m', 'ai_happiness', 'chat', '--offline', '--device', $Device)
    if ($Run) { $taskChatArgs += @('--run', $Run) }
    & $taskPython @taskChatArgs
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("Stopped: $($_.Exception.Message)")
    exit 1
} finally {
    Pop-Location
}
