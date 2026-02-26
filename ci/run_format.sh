#!/usr/bin/env bash

# Backward-compatible wrapper.
exec ./ci/run_ruff.sh --format "$@"
