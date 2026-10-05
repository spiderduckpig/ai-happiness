[CmdletBinding()]
param(
    [switch]$Recorded,
    [switch]$Expressive,
    [ValidateRange(1, 20)][int]$Trials = 2,
    [ValidateRange(32, 256)][int]$MaxNewTokens = 128,
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
    $taskExampleArgs = @('-u', '-m', 'ai_happiness', 'examples', '--device', $Device,
                        '--trials', [string]$Trials, '--max-new-tokens', [string]$MaxNewTokens)
    if ($Recorded) { $taskExampleArgs += '--recorded' }
    if ($Expressive) { $taskExampleArgs += '--expressive' }
    & $taskPython @taskExampleArgs
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("Stopped: $($_.Exception.Message)")
    exit 1
} finally {
    Pop-Location
}
