#!/usr/bin/env bash

set -e

echo "Creating .venv-ci environment..."
uv venv .venv-ci --clear --python 3.8
. .venv-ci/bin/activate && \
uv pip install ".[all]" ".[ci]" ".[test]"
