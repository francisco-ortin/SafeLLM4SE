#!/usr/bin/env bash
# Exported from the PyCharm run configuration: reporting ollama qwen-coder
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f report.py ]]; then
    echo "Required script not found: report.py" >&2
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_COMMAND="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_COMMAND="python"
else
    echo "Python was not found in PATH." >&2
    exit 1
fi

"$PYTHON_COMMAND" ./report.py \
    --input \
    output/measurements.csv \
    --output \
    output/report-qwen-coder.csv \
    --task-id \
    task-id-54 \
    --task-name \
    QwenCoder \
    --boxplot \
    output/qwen-coder-boxplot.svg \
    --violin \
    output/qwen-coder-violin.svg \
    --ECDF \
    output/qwen-coder-ecdf.svg \
    --raincloud \
    output/qwen-coder-raincloud.svg \
    --KDE \
    output/qwen-coder-kde.svg \
    --log \
    DEBUG
