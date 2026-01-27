#!/usr/bin/env bash

# Ensure .venv-ci exists
if [ ! -d ".venv-ci" ]; then
    ./ci/make_venv.sh
fi

# Activate the venv
source .venv-ci/bin/activate

# Parse command-line arguments
DO_FORMAT=false
DO_LINT=false
DO_TYPECHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            DO_FORMAT=true
            shift
            ;;
        --lint)
            DO_LINT=true
            shift
            ;;
        --typecheck)
            DO_TYPECHECK=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--format] [--lint] [--typecheck]"
            exit 1
            ;;
    esac
done

# Check if at least one flag was provided
if [ "$DO_FORMAT" = false ] && [ "$DO_LINT" = false ] && [ "$DO_TYPECHECK" = false ]; then
    echo "Error: At least one flag must be provided (--format, --lint, --typecheck)"
    exit 1
fi

# Track if any check failed
EXIT_CODE=0

# Run format check
if [ "$DO_FORMAT" = true ]; then
    echo "Running ruff format..."
    if .venv-ci/bin/python3 -m ruff format ./demos && \
       .venv-ci/bin/python3 -m ruff format ./examples && \
       .venv-ci/bin/python3 -m ruff format ./tests && \
       .venv-ci/bin/python3 -m ruff format ./src/trtutils; then
        echo "✓ Format check passed"
    else
        echo "✗ Format check failed"
        EXIT_CODE=1
    fi
fi

# Run lint check
if [ "$DO_LINT" = true ]; then
    echo "Running ruff lint..."
    if .venv-ci/bin/python3 -m ruff check ./demos --fix --ignore=INP001,T201 && \
       .venv-ci/bin/python3 -m ruff check ./examples --fix --ignore=INP001,T201,D103 && \
       .venv-ci/bin/python3 -m ruff check ./tests --fix --ignore=S101,D100,D104,PLR2004,T201 && \
       .venv-ci/bin/python3 -m ruff check ./src/trtutils --fix; then
        echo "✓ Lint check passed"
    else
        echo "✗ Lint check failed"
        EXIT_CODE=1
    fi
fi

# Run typecheck
if [ "$DO_TYPECHECK" = true ]; then
    echo "Running ty typecheck..."
    if .venv-ci/bin/ty check examples && \
       .venv-ci/bin/ty check tests && \
       .venv-ci/bin/ty check src; then
        echo "✓ Typecheck passed"
    else
        echo "✗ Typecheck failed"
        EXIT_CODE=1
    fi
fi

exit $EXIT_CODE
