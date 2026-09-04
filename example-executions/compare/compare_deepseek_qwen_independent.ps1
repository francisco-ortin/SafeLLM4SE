# Exported from the PyCharm run configuration: compare independent deepseek qwen
$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDirectory "..\..")
Set-Location $ProjectRoot

if (-not (Test-Path 'compare.py')) {
    throw "Required script not found: compare.py"
}

$Arguments = @(
    '.\compare.py',
    '--task-id-1',
    'task-id-54',
    '--task-id-2',
    'task-id-56',
    '--task-name-1',
    'QwenCoder',
    '--task-name-2',
    'DeepseekCoder',
    '--input',
    'output/measurements.csv',
    '--test-type',
    'independent',
    '--output',
    'output/compare-independent-deepseek-qwen.csv',
    '--log',
    'DEBUG',
    '--boxplot',
    'output/compare-deepseek-qwen-boxkplot.svg',
    '--violin',
    'output/compare-deepseek-qwen-violin.svg',
    '--ECDF',
    'output/compare-deepseek-qwen-ecdf.svg',
    '--raincloud',
    'output/compare-deepseek-qwen-raincloud.svg',
    '--kde',
    'output/compare-deepseek-qwen-kde.svg'
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
