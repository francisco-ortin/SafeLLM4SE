#!/usr/bin/env bash
# Exported from the PyCharm run configuration: sampling ollama random
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f sample.py ]]; then
    echo "Required script not found: sample.py" >&2
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

"$PYTHON_COMMAND" ./sample.py \
    --evaluator \
    src.safellm4se.sampling.myevaluators.ollama.random \
    --temperature=2.0 \
    --verbose
