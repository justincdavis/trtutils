#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

set -euo pipefail

CUDA_VERSION="${1:?Usage: $0 <cuda-version> [--test] [--lint] [--typecheck] [--coverage] [--all]}"
shift

die() {
    echo "Error: $*" >&2
    exit 1
}

if [[ ! "$CUDA_VERSION" =~ ^(11|12|13)$ ]]; then
    die "cuda-version must be one of: 11, 12, 13"
fi

command -v docker >/dev/null 2>&1 || die "required command not found: docker"
docker compose version >/dev/null 2>&1 || die "docker compose check failed; ensure docker compose v2 is installed"
nvidia-smi >/dev/null 2>&1 || die "nvidia-smi check failed; ensure NVIDIA driver is installed"

COMPOSE="docker compose -f docker/docker-compose.test.yml"
SERVICE="test-cu${CUDA_VERSION}"

# WSL2: the NVIDIA Container Toolkit doesn't mount all driver libraries.
# Mount /usr/lib/wsl so containers can access libnvdxgdmal.so.1 et al.
WSL_FLAGS=""
if grep -qi microsoft /proc/version 2>/dev/null; then
    if [ -d /usr/lib/wsl ]; then
        WSL_FLAGS="-v /usr/lib/wsl:/usr/lib/wsl:ro"
        echo "WSL2 detected — mounting /usr/lib/wsl into containers"
    fi
fi

# Ensure .coverage.json exists as a file before Docker mounts it
# (otherwise Docker creates it as a directory)
touch "$(dirname "$0")/../.coverage.json"

# Build the container (uses cache if already built)
echo "=== Building ${SERVICE} image ==="
$COMPOSE build $SERVICE

# Parse flags
DO_TEST=false
DO_LINT=false
DO_TYPECHECK=false
DO_COVERAGE=false

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
        --coverage)
            DO_COVERAGE=true
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
            echo "Usage: $0 <cuda-version> [--test] [--lint] [--typecheck] [--coverage] [--all]"
            exit 1
            ;;
    esac
done

# Default to --all if no flags given
if [ "$DO_TEST" = false ] && [ "$DO_LINT" = false ] && [ "$DO_TYPECHECK" = false ] && [ "$DO_COVERAGE" = false ]; then
    DO_TEST=true
    DO_LINT=true
    DO_TYPECHECK=true
fi

# Track if any check failed
EXIT_CODE=0

if [ "$DO_LINT" = true ]; then
    echo "=== Running lint (cu${CUDA_VERSION}) ==="
    if $COMPOSE run --rm $WSL_FLAGS $SERVICE ./ci/run_ruff.sh --lint --no-fix; then
        echo "Lint passed"
    else
        echo "Lint failed"
        EXIT_CODE=1
    fi
fi

if [ "$DO_TYPECHECK" = true ]; then
    echo "=== Running typecheck (cu${CUDA_VERSION}) ==="
    if $COMPOSE run --rm $WSL_FLAGS $SERVICE ./ci/run_type_check.sh; then
        echo "Typecheck passed"
    else
        echo "Typecheck failed"
        EXIT_CODE=1
    fi
fi

if [ "$DO_TEST" = true ]; then
    echo "=== Running tests (cu${CUDA_VERSION}) ==="
    if $COMPOSE run --rm $WSL_FLAGS $SERVICE ./ci/run_tests.sh; then
        echo "Tests passed"
    else
        echo "Tests failed"
        EXIT_CODE=1
    fi
fi

if [ "$DO_COVERAGE" = true ]; then
    echo "=== Running coverage (cu${CUDA_VERSION}) ==="
    if $COMPOSE run --rm $WSL_FLAGS $SERVICE ./ci/run_coverage.sh; then
        echo "Coverage passed"
    else
        echo "Coverage failed"
        EXIT_CODE=1
    fi
fi

exit $EXIT_CODE
