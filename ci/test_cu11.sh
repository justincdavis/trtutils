#!/usr/bin/env bash
set -euo pipefail

COMPOSE="docker compose -f docker/docker-compose.test.yml"
SERVICE="test-cu11"

# Build the container (uses cache if already built)
$COMPOSE build $SERVICE

# Parse flags
DO_TEST=false
DO_LINT=false
DO_TYPECHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            DO_TEST=true
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
        --all)
            DO_TEST=true
            DO_LINT=true
            DO_TYPECHECK=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--lint] [--typecheck] [--all]"
            exit 1
            ;;
    esac
done

# Default to --all if no flags given
if [ "$DO_TEST" = false ] && [ "$DO_LINT" = false ] && [ "$DO_TYPECHECK" = false ]; then
    DO_TEST=true
    DO_LINT=true
    DO_TYPECHECK=true
fi

# Track if any check failed
EXIT_CODE=0

if [ "$DO_LINT" = true ]; then
    echo "=== Running lint (cu11) ==="
    if $COMPOSE run --rm $SERVICE ./ci/run_lint.sh --no-fix; then
        echo "Lint passed"
    else
        echo "Lint failed"
        EXIT_CODE=1
    fi
fi

if [ "$DO_TYPECHECK" = true ]; then
    echo "=== Running typecheck (cu11) ==="
    if $COMPOSE run --rm $SERVICE ./ci/run_type_check.sh; then
        echo "Typecheck passed"
    else
        echo "Typecheck failed"
        EXIT_CODE=1
    fi
fi

if [ "$DO_TEST" = true ]; then
    echo "=== Running tests (cu11) ==="
    if $COMPOSE run --rm $SERVICE python3 -m pytest -rP -v tests/; then
        echo "Tests passed"
    else
        echo "Tests failed"
        EXIT_CODE=1
    fi
fi

exit $EXIT_CODE
