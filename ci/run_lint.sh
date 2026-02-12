#!/usr/bin/env bash

# Run ruff check on all configured directories
set -e

# Use venv ruff if available, otherwise use system ruff
if [ -f ".venv-ci/bin/ruff" ]; then
    RUFF_CMD=".venv-ci/bin/ruff"
elif command -v ruff >/dev/null 2>&1; then
    RUFF_CMD="ruff"
elif command -v python3 >/dev/null 2>&1 && python3 -m ruff --version >/dev/null 2>&1; then
    RUFF_CMD="python3 -m ruff"
else
    echo "Error: ruff not found. Please install ruff or activate the CI venv."
    exit 1
fi

DIRS=("./demos" "./examples" "./tests" "./src/trtutils")

for dir in "${DIRS[@]}"; do
    echo "Checking $dir..."
    $RUFF_CMD check "$dir" "$@"
done
