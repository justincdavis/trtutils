#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
TRTUTILS_IGNORE_MISSING_CUDA=1 python3 -m pytest -m cpu tests/benchmark/ tests/core/test_cache.py -rP -v
