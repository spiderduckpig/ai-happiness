[CmdletBinding()]
param(
    [switch]$FullSweep,
    [switch]$Cpu,
    [switch]$CheckOnly,
    [string]$Revision = 'c1899de289a04d12100db370d81485cdf75e47ca'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$NativeArgs
    )
    & $Executable @NativeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($NativeArgs -join ' ')"
    }
}

Push-Location -LiteralPath $PSScriptRoot
$taskTranscriptStarted = $false
try {
    $taskLogDirectory = Join-Path $PSScriptRoot '.cache\setup-logs'
    $null = New-Item -ItemType Directory -Force -Path $taskLogDirectory
    $taskLog = Join-Path $taskLogDirectory ((Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.txt')
    $null = Start-Transcript -Path $taskLog
    $taskTranscriptStarted = $true
    Write-Host "Setup log: $taskLog"

    Write-Host 'Checking free memory and disk before starting...'
    Invoke-Checked -Executable 'py' -NativeArgs @('-3.12', 'setup_support.py', 'preflight')
    if ($CheckOnly) { return }

    $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        Write-Host 'Creating a Python 3.12 environment for this project...'
        Invoke-Checked -Executable 'py' -NativeArgs @('-3.12', '-m', 'venv', '.venv')
    }

    Write-Host 'Installing dependencies. The first installation downloads several GB.'
    Invoke-Checked -Executable $taskPython -NativeArgs @('-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'pip')
    $taskTorchFlavor = if ($Cpu) { 'cpu' } else { 'cu128' }
    Invoke-Checked -Executable $taskPython -NativeArgs @('setup_support.py', 'torch', '--flavor', $taskTorchFlavor)
    Invoke-Checked -Executable $taskPython -NativeArgs @('-m', 'pip', 'install', '--no-cache-dir', '-e', '.')

    if ($Cpu) {
        $taskDevice = 'cpu'
    } else {
        Write-Host 'Checking GPU access...'
        Invoke-Checked -Executable $taskPython -NativeArgs @('-c', "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable. Check the GPU/PyTorch installation, or rerun start.ps1 with -Cpu.'; print('GPU:', torch.cuda.get_device_name(0))")
        $taskDevice = 'cuda'
    }

    Write-Host 'Running the offline software checks...'
    Invoke-Checked -Executable $taskPython -NativeArgs @('-m', 'unittest', 'discover', '-s', 'tests', '-v')

    Write-Host 'Starting calibration. Qwen3-0.6B downloads automatically on the first run (about 1.52 GB).'
    Invoke-Checked -Executable $taskPython -NativeArgs @('setup_support.py', 'preflight', '--stage', 'run')
    $calibrationArgs = @('-m', 'ai_happiness', 'run', '--device', $taskDevice, '--revision', $Revision)
    if (-not $FullSweep) {
        $calibrationArgs += @('--layers', '13', '--doses', '0.5,1,2', '--trials', '1')
    }
    Invoke-Checked -Executable $taskPython -NativeArgs $calibrationArgs

    Write-Host ''
    Write-Host 'Calibration finished. The output above identifies the report and, if a setting passed, the chat command.'
    Write-Host 'For a broader search, rerun this script with -FullSweep.'
} catch {
    [Console]::Error.WriteLine("Stopped: $($_.Exception.Message)")
    exit 1
} finally {
    if ($taskTranscriptStarted) { $null = Stop-Transcript }
    Pop-Location
}
