#!/usr/bin/env bash

# Run ruff (lint or format) on all configured directories.
set -euo pipefail

usage() {
    echo "Usage: $0 [--format|--lint] [ruff args...]"
    echo "  --format   Run 'ruff format'"
    echo "  --lint     Run 'ruff check' (default)"
}

MODE="lint"
case "${1:-}" in
    --format)
        MODE="format"
        shift
        ;;
    --lint)
        MODE="lint"
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
esac

# Use venv ruff if available, otherwise use system ruff.
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
    if [ "$MODE" = "format" ]; then
        echo "Formatting $dir..."
        $RUFF_CMD format "$dir" "$@"
    else
        echo "Checking $dir..."
        $RUFF_CMD check "$dir" "$@"
    fi
done
