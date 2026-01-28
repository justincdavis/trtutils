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
    if ./ci/run_format.sh --check --diff; then
        echo "✓ Format check passed"
    else
        echo "✗ Format check failed"
        EXIT_CODE=1
    fi
fi

# Run lint check
if [ "$DO_LINT" = true ]; then
    echo "Running ruff lint..."
    if ./ci/run_lint.sh --no-fix; then
        echo "✓ Lint check passed"
    else
        echo "✗ Lint check failed"
        EXIT_CODE=1
    fi
fi

# Run typecheck
if [ "$DO_TYPECHECK" = true ]; then
    echo "Running ty typecheck..."
    if ./ci/run_type_check.sh; then
        echo "✓ Typecheck passed"
    else
        echo "✗ Typecheck failed"
        EXIT_CODE=1
    fi
fi

exit $EXIT_CODE
