[CmdletBinding()]
param(
    [ValidateRange(1, 256)][int]$Instances = 32,
    [ValidateRange(0, 1000000)][int]$Rounds = 3,
    [ValidateRange(32, 256)][int]$MaxNewTokens = 96,
    [ValidateSet('auto', 'cpu', 'cuda', 'mps')][string]$Device = 'auto',
    [string]$Resume
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
    $taskGardenArgs = @('-u', '-m', 'ai_happiness', 'garden', '--device', $Device)
    if ($Resume) { $taskGardenArgs += @('--resume', $Resume) }
    if (-not $Resume -or $PSBoundParameters.ContainsKey('Instances')) {
        $taskGardenArgs += @('--instances', [string]$Instances)
    }
    if (-not $Resume -or $PSBoundParameters.ContainsKey('Rounds')) {
        $taskGardenArgs += @('--rounds', [string]$Rounds)
    }
    if (-not $Resume -or $PSBoundParameters.ContainsKey('MaxNewTokens')) {
        $taskGardenArgs += @('--max-new-tokens', [string]$MaxNewTokens)
    }
    & $taskPython @taskGardenArgs
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("Stopped: $($_.Exception.Message)")
    exit 1
} finally {
    Pop-Location
}
