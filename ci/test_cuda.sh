#!/usr/bin/env bash
set -euo pipefail

CUDA_VERSION="${1:?Usage: $0 <cuda-version> [--test] [--lint] [--typecheck] [--all]}"
shift

die() {
    echo "Error: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "required command not found: $cmd"
}

validate_cmd_healthy() {
    local label="$1"
    shift
    "$@" >/dev/null 2>&1 || die "$label check failed; ensure host GPU/CUDA tooling is correctly installed"
}

if [[ ! "$CUDA_VERSION" =~ ^(11|12|13)$ ]]; then
    die "cuda-version must be one of: 11, 12, 13"
fi

require_cmd docker
validate_cmd_healthy "docker compose" docker compose version
require_cmd nvidia-smi
validate_cmd_healthy "nvidia-smi" nvidia-smi
require_cmd nvcc
validate_cmd_healthy "nvcc" nvcc --version

COMPOSE="docker compose -f docker/docker-compose.test.yml"
SERVICE="test-cu${CUDA_VERSION}"

# Build the container (uses cache if already built)
echo "=== Building ${SERVICE} image ==="
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
            echo "Usage: $0 <cuda-version> [--test] [--lint] [--typecheck] [--all]"
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
    echo "=== Running lint (cu${CUDA_VERSION}) ==="
    if $COMPOSE run --rm $SERVICE ./ci/run_ruff.sh --lint --no-fix; then
        echo "Lint passed"
    else
        echo "Lint failed"
        EXIT_CODE=1
    fi
fi

if [ "$DO_TYPECHECK" = true ]; then
    echo "=== Running typecheck (cu${CUDA_VERSION}) ==="
    if $COMPOSE run --rm $SERVICE ./ci/run_type_check.sh; then
        echo "Typecheck passed"
    else
        echo "Typecheck failed"
        EXIT_CODE=1
    fi
fi

if [ "$DO_TEST" = true ]; then
    echo "=== Running tests (cu${CUDA_VERSION}) ==="
    if $COMPOSE run --rm $SERVICE python3 -m pytest -rP -v tests/; then
        echo "Tests passed"
    else
        echo "Tests failed"
        EXIT_CODE=1
    fi
fi

exit $EXIT_CODE
