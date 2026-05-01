#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

set -e

PYTHON_VERSION="${1:-3.14}"

echo "Creating .venv-ci environment with Python ${PYTHON_VERSION}..."
uv venv .venv-ci --clear --python "$PYTHON_VERSION"
. .venv-ci/bin/activate && \
uv pip install -e ".[ci]" ".[test]"
