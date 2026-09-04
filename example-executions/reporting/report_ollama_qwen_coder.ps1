# Exported from the PyCharm run configuration: reporting ollama qwen-coder
$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDirectory "..\..")
Set-Location $ProjectRoot

if (-not (Test-Path 'report.py')) {
    throw "Required script not found: report.py"
}

$Arguments = @(
    '.\report.py',
    '--input',
    'output/measurements.csv',
    '--output',
    'output/report-qwen-coder.csv',
    '--task-id',
    'task-id-54',
    '--task-name',
    'QwenCoder',
    '--boxplot',
    'output/qwen-coder-boxplot.svg',
    '--violin',
    'output/qwen-coder-violin.svg',
    '--ECDF',
    'output/qwen-coder-ecdf.svg',
    '--raincloud',
    'output/qwen-coder-raincloud.svg',
    '--KDE',
    'output/qwen-coder-kde.svg',
    '--log',
    'DEBUG'
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
