# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Kernel test fixtures -- a CUDA stream and a launch helper for the image kernels."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils.core import (
    Kernel,
    create_binding,
    create_stream,
    destroy_stream,
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
    stream_synchronize,
)

THREADS = (32, 32, 1)


@pytest.fixture
def cuda_stream():
    """Create a CUDA stream for the test, destroy after."""
    stream = create_stream()
    yield stream
    destroy_stream(stream)


def run_kernel(stream, spec, args, output, blocks) -> np.ndarray:
    """
    Compile and launch an image kernel, returning the host copy of output.

    ndarray args are uploaded and passed as device pointers; output is passed as its
    device pointer wherever it appears in args; other args are passed as scalars.
    """
    kernel = Kernel(*spec)
    out_binding = create_binding(output, pagelocked_mem=True)
    bindings = []
    ptrs = []
    for arg in args:
        if arg is output:
            ptrs.append(out_binding.allocation)
        elif isinstance(arg, np.ndarray):
            binding = create_binding(arg, is_input=True)
            memcpy_host_to_device_async(binding.allocation, arg, stream)
            bindings.append(binding)
            ptrs.append(binding.allocation)
        else:
            ptrs.append(arg)

    kernel.call(blocks, THREADS, stream, kernel.create_args(*ptrs))
    memcpy_device_to_host_async(out_binding.host_allocation, out_binding.allocation, stream)
    stream_synchronize(stream)
    result = out_binding.host_allocation.copy()

    for binding in bindings:
        binding.free()
    out_binding.free()
    kernel.free()
    return result
