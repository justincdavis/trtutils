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
create_context
    Create a CUDA context.
create_engine
    Create a TensorRT engine from a serialized engine file.
cuda_call
    A function for checking the return status of a CUDA call.
destroy_context
    Destroy a CUDA context.
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

"""

from __future__ import annotations

from ._bindings import Binding, allocate_bindings
from ._context import create_context, destroy_context
from ._cuda import cuda_call
from ._engine import create_engine
from ._interface import TRTEngineInterface
from ._memory import (
    memcpy_device_to_host,
    memcpy_device_to_host_async,
    memcpy_host_to_device,
    memcpy_host_to_device_async,
)
from ._stream import stream_synchronize

__all__ = [
    "Binding",
    "TRTEngineInterface",
    "allocate_bindings",
    "create_context",
    "create_engine",
    "cuda_call",
    "destroy_context",
    "memcpy_device_to_host",
    "memcpy_device_to_host_async",
    "memcpy_host_to_device",
    "memcpy_host_to_device_async",
    "stream_synchronize",
]
