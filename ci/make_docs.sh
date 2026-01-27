#!/usr/bin/env bash

set -e

# Ensure .venv-docs exists
if [ ! -d ".venv-docs" ]; then
    echo "Creating .venv-docs environment..."
    uv venv .venv-docs --clear --python 3.8
    . .venv-docs/bin/activate && \
    uv pip install -r docs/requirements.txt
fi

# Activate the venv
source .venv-docs/bin/activate

echo "Building documentation..."
rm -rf docs/_build/*
.venv-docs/bin/python3 ci/build_benchmark_docs.py
.venv-docs/bin/python3 ci/build_example_docs.py
.venv-docs/bin/sphinx-apidoc -o docs/source/ src/trtutils/ --separate --force
cd docs && SPHINXBUILD=../.venv-docs/bin/sphinx-build make html
