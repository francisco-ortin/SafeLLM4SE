#!/usr/bin/env bash
# Exported from the PyCharm run configuration: compare paired deepseek qwen
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f compare.py ]]; then
    echo "Required script not found: compare.py" >&2
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

"$PYTHON_COMMAND" ./compare.py \
    --task-id-1 \
    task-id-54 \
    --task-id-2 \
    task-id-56 \
    --task-name-1 \
    QwenCoder \
    --task-name-2 \
    DeepseekCoder \
    --input \
    output/measurements.csv \
    --test-type \
    paired \
    --output \
    output/compare-paired-deepseek-qwen.csv \
    --log \
    DEBUG \
    --boxplot \
    output/compare-deepseek-qwen-boxkplot.svg \
    --violin \
    output/compare-deepseek-qwen-violin.svg \
    --ECDF \
    output/compare-deepseek-qwen-ecdf.svg \
    --raincloud \
    output/compare-deepseek-qwen-raincloud.svg \
    --kde \
    output/compare-deepseek-qwen-kde.svg
