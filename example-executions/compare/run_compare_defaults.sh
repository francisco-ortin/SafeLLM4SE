#!/usr/bin/env bash
# Exported from the PyCharm run configuration: compare
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

"$PYTHON_COMMAND" ./compare.py
