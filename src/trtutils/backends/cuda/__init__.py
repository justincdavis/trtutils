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
TRTEngine
    A class for running inference on a TensorRT engine.

Functions
---------
allocate_bindings
    Allocate the bindings for a TensorRT engine.
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
from ._engine import TRTEngine
from ._memory import memcpy_device_to_host, memcpy_host_to_device

__all__ = [
    "Binding",
    "TRTEngine",
    "allocate_bindings",
    "cuda_call",
    "memcpy_device_to_host",
    "memcpy_host_to_device",
]
