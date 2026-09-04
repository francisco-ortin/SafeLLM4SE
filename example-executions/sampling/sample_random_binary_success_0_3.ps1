# Exported from the PyCharm run configuration: sampling random_binary
$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDirectory "..\..")
Set-Location $ProjectRoot

if (-not (Test-Path 'sample.py')) {
    throw "Required script not found: sample.py"
}

$Arguments = @(
    '.\sample.py',
    '--evaluator',
    'src.safellm4se.sampling.myevaluators.random_binary_evaluator',
    '--verbose',
    '--success-probability=0.3'
)

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    & $PythonCommand.Source @Arguments
    exit $LASTEXITCODE
}

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    & $PythonLauncher.Source @Arguments
    exit $LASTEXITCODE
}

$WslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($WslCommand) {
    $WslArguments = $Arguments -replace '^\.\', './'
    & $WslCommand.Source python3 @WslArguments
    exit $LASTEXITCODE
}

throw "Python was not found in PATH, and WSL is not available."
