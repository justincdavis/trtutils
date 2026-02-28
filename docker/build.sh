#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
COMPOSE="docker compose -f $SCRIPT_DIR/docker-compose.test.yml"

die() {
    echo "Error: $*" >&2
    exit 1
}

usage() {
    echo "Usage: $0 [--all] [--act] [--cu11] [--cu12] [--cu13] [--log]"
    echo ""
    echo "Build Docker images for trtutils. Builds run serially."
    echo ""
    echo "Options:"
    echo "  --all   Build all images (default if no targets given)"
    echo "  --act   Build the GitHub Actions runner image"
    echo "  --cu11  Build the CUDA 11 test image"
    echo "  --cu12  Build the CUDA 12 test image"
    echo "  --cu13  Build the CUDA 13 test image"
    echo "  --log   Write per-image logs to docker/logs/"
    echo "  --help  Show this help message"
}

# Preflight
command -v docker >/dev/null 2>&1 || die "required command not found: docker"
docker compose version >/dev/null 2>&1 || die "docker compose check failed; ensure docker compose v2 is installed"

# Parse flags
BUILD_ACT=false
BUILD_CU11=false
BUILD_CU12=false
BUILD_CU13=false
LOG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            BUILD_ACT=true; BUILD_CU11=true; BUILD_CU12=true; BUILD_CU13=true
            shift ;;
        --act)       BUILD_ACT=true;  shift ;;
        --cu11)      BUILD_CU11=true; shift ;;
        --cu12)      BUILD_CU12=true; shift ;;
        --cu13)      BUILD_CU13=true; shift ;;
        --log)       LOG=true;        shift ;;
        --help)      usage; exit 0 ;;
        *)           echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Default to --all if no targets given
if [ "$BUILD_ACT" = false ] && [ "$BUILD_CU11" = false ] && \
   [ "$BUILD_CU12" = false ] && [ "$BUILD_CU13" = false ]; then
    BUILD_ACT=true; BUILD_CU11=true; BUILD_CU12=true; BUILD_CU13=true
fi

if [ "$LOG" = true ]; then
    mkdir -p "$LOG_DIR"
fi

run_build() {
    local label="$1"
    shift
    echo "=== Building ${label} ==="
    if [ "$LOG" = true ]; then
        local logfile="$LOG_DIR/${label}.log"
        "$@" 2>&1 | tee "$logfile"
        echo "Log saved to ${logfile}"
    else
        "$@"
    fi
}

cd "$REPO_ROOT"

if [ "$BUILD_ACT" = true ]; then
    run_build "act" docker build -f docker/Dockerfile.act -t trtutils-act:latest .
fi

if [ "$BUILD_CU11" = true ]; then
    run_build "cu11" $COMPOSE build test-cu11
fi

if [ "$BUILD_CU12" = true ]; then
    run_build "cu12" $COMPOSE build test-cu12
fi

if [ "$BUILD_CU13" = true ]; then
    run_build "cu13" $COMPOSE build test-cu13
fi

echo "=== Done ==="
