# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
CUDA backend for TRTEngine.

This module provides the CUDA backend for the TRTEngine class. It provides
utilities for managing device memory, copying data between host and device,
and running inference on the engine.

Classes
-------
Binding
    A class for managing a CUDA allocation.
TRTEngineInterface
    An interface for the TRTEngine class.

Functions
---------
allocate_bindings
    Allocate the bindings for a TensorRT engine.
allocate_pinned_memory
    Allocate pagelocked memory using CUDA.
create_binding
    Create a Binding from a np.ndarray.
create_context
    Create a CUDA context.
create_engine
    Create a TensorRT engine from a serialized engine file.
create_stream
    Create a CUDA stream.
cuda_call
    A function for checking the return status of a CUDA call.
cuda_malloc
    Allocate memory on the CUDA device using CUDA runtime.
destroy_context
    Destroy a CUDA context.
destroy_stream
    Destroy a CUDA stream.
memcpy_device_to_host
    Copy data from device to host.
memcpy_host_to_device
    Copy data from host to device.
memcpy_device_to_host_async
    Copy data from device to host async.
memcpy_host_to_device_async
    Copy data from host to device async.
stream_synchronize
    Synchronize the cuda stream.
nvrtc_call
    A function for checking the return status of a NVRTC call.
compile_kernel
    Compile a kernel using NVRTC.
load_kernel
    Load a CUDA module and kernel from PTX from NVRTC.
compile_and_load_kernel
    Compile and load a kernel using NVRTC.

"""

from __future__ import annotations

from ._bindings import Binding, allocate_bindings, create_binding
from ._context import create_context, destroy_context
from ._cuda import cuda_call
from ._engine import create_engine
from ._interface import TRTEngineInterface
from ._memory import (
    allocate_pinned_memory,
    cuda_malloc,
    memcpy_device_to_host,
    memcpy_device_to_host_async,
    memcpy_host_to_device,
    memcpy_host_to_device_async,
)
from ._nvrtc import compile_and_load_kernel, compile_kernel, load_kernel, nvrtc_call
from ._stream import create_stream, destroy_stream, stream_synchronize

__all__ = [
    "Binding",
    "TRTEngineInterface",
    "allocate_bindings",
    "allocate_pinned_memory",
    "compile_and_load_kernel",
    "compile_kernel",
    "create_binding",
    "create_context",
    "create_engine",
    "create_stream",
    "cuda_call",
    "cuda_malloc",
    "destroy_context",
    "destroy_stream",
    "load_kernel",
    "memcpy_device_to_host",
    "memcpy_device_to_host_async",
    "memcpy_host_to_device",
    "memcpy_host_to_device_async",
    "nvrtc_call",
    "stream_synchronize",
]
