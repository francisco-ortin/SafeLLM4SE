# Exported from the PyCharm run configuration: sampling groq random 
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
    'src.safellm4se.sampling.myevaluators.groq.random',
    '--temperature=2.0',
    '--n-min=1',
    '--budget-tokens=1',
    '--api-keys',
    'src/safellm4se/sampling/myevaluators/api-keys.json',
    '--verbose'
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
