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
:class:`Binding`
    A class for managing a CUDA allocation.
:class:`TRTEngineInterface`
    An interface for the TRTEngine class.
:class:`Kernel`
    Wrapper around CUDA kernels.

Functions
---------
:func:`allocate_bindings`
    Allocate the bindings for a TensorRT engine.
:func:`allocate_pinned_memory`
    Allocate pagelocked memory using CUDA.
:func:`create_binding`
    Create a Binding from a np.ndarray.
:func:`create_context`
    Create a CUDA context.
:func:`create_engine`
    Create a TensorRT engine from a serialized engine file.
:func:`create_stream`
    Create a CUDA stream.
:func:`cuda_call`
    A function for checking the return status of a CUDA call.
:func:`cuda_malloc`
    Allocate memory on the CUDA device using CUDA runtime.
:func:`destroy_context`
    Destroy a CUDA context.
:func:`destroy_stream`
    Destroy a CUDA stream.
:func:`memcpy_device_to_host`
    Copy data from device to host.
:func:`memcpy_host_to_device`
    Copy data from host to device.
:func:`memcpy_device_to_host_async`
    Copy data from device to host async.
:func:`memcpy_host_to_device_async`
    Copy data from host to device async.
:func:`stream_synchronize`
    Synchronize the cuda stream.
:func:`nvrtc_call`
    A function for checking the return status of a NVRTC call.
:func:`compile_kernel`
    Compile a kernel using NVRTC.
:func:`load_kernel`
    Load a CUDA module and kernel from PTX from NVRTC.
:func:`compile_and_load_kernel`
    Compile and load a kernel using NVRTC.
:func:`launch_kernel`
    Launch a CUDA kernel.
:func:`create_kernel_args`
    Create the argument array for a kernel call.

"""

from __future__ import annotations

from ._bindings import Binding, allocate_bindings, create_binding
from ._context import create_context, destroy_context
from ._cuda import cuda_call
from ._engine import create_engine
from ._interface import TRTEngineInterface
from ._kernels import Kernel, create_kernel_args, launch_kernel
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
    "Kernel",
    "TRTEngineInterface",
    "allocate_bindings",
    "allocate_pinned_memory",
    "compile_and_load_kernel",
    "compile_kernel",
    "create_binding",
    "create_context",
    "create_engine",
    "create_kernel_args",
    "create_stream",
    "cuda_call",
    "cuda_malloc",
    "destroy_context",
    "destroy_stream",
    "launch_kernel",
    "load_kernel",
    "memcpy_device_to_host",
    "memcpy_device_to_host_async",
    "memcpy_host_to_device",
    "memcpy_host_to_device_async",
    "nvrtc_call",
    "stream_synchronize",
]
