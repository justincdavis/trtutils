#!/usr/bin/env bash

# Run ty typecheck on all configured directories
set -e

# Use venv ty if available, otherwise use system ty
if [ -f ".venv-ci/bin/ty" ]; then
    TY_CMD=".venv-ci/bin/ty"
elif command -v ty >/dev/null 2>&1; then
    TY_CMD="ty"
elif command -v python3 >/dev/null 2>&1 && python3 -m ty --version >/dev/null 2>&1; then
    TY_CMD="python3 -m ty"
else
    echo "Error: ty not found. Please install ty or activate the CI venv."
    exit 1
fi

DIRS=("examples" "tests" "src")

for dir in "${DIRS[@]}"; do
    echo "Type checking $dir..."
    $TY_CMD check "$dir" "$@"
done
