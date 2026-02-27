#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

# Backward-compatible wrapper.
exec ./ci/run_ruff.sh --format "$@"
