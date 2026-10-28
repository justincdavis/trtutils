#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
TRTUTILS_IGNORE_MISSING_CUDA=1 python3 -m pytest -m cpu tests/benchmark/ tests/core/test_cache.py tests/core/test_bindings.py tests/core/test_buffer.py tests/parallel/test_parallel_detector.py tests/parallel/test_parallel_engines.py tests/core/test_engine_inputs.py tests/core/test_kernels.py tests/test_compat_libs.py tests/test_register.py tests/image/test_ops.py tests/image/test_nms_indices.py tests/image/test_geometry.py tests/image/test_warp.py tests/image/test_ocr_processing.py -rP -v
