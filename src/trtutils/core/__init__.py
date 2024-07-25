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
create_engine
    Create a TensorRT engine from a serialized engine file.
cuda_call
    A function for checking the return status of a CUDA call.
memcpy_device_to_host
    Copy data from device to host.
memcpy_host_to_device
    Copy data from host to device.

"""

from __future__ import annotations

from ._bindings import Binding, allocate_bindings
from ._cuda import cuda_call
from ._engine import create_engine
from ._interface import TRTEngineInterface
from ._memory import memcpy_device_to_host, memcpy_host_to_device

__all__ = [
    "Binding",
    "TRTEngineInterface",
    "allocate_bindings",
    "create_engine",
    "cuda_call",
    "memcpy_device_to_host",
    "memcpy_host_to_device",
]
