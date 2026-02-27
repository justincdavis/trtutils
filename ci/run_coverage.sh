#!/usr/bin/env bash
# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
#
# CI coverage script: runs pytest with branch coverage then validates
# per-module thresholds via the ratcheting script.
#
# Usage:
#   ci/run_coverage.sh              # Run tests + validate thresholds
#   ci/run_coverage.sh --update     # Run tests + ratchet thresholds up
#   ci/run_coverage.sh -k "test_x"  # Pass extra args to pytest
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Separate our flags from pytest flags
UPDATE_FLAG=""
PYTEST_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --update)
            UPDATE_FLAG="--update"
            ;;
        *)
            PYTEST_ARGS+=("$arg")
            ;;
    esac
done

echo "=== Step 1: Running tests with branch coverage ==="
python3 -m pytest \
    tests/ \
    --cov=src/trtutils \
    --cov-branch \
    --cov-report=term-missing \
    --cov-report=json:"${PROJECT_ROOT}/.coverage.json" \
    -v \
    "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"

echo ""
echo "=== Step 2: Validating coverage thresholds ==="
python3 "${SCRIPT_DIR}/check_coverage.py" $UPDATE_FLAG
