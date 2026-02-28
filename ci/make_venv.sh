#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

set -e

echo "Creating .venv-ci environment..."
uv venv .venv-ci --clear --python 3.8
. .venv-ci/bin/activate && \
uv pip install -e ".[all]" ".[ci]" ".[test]"
