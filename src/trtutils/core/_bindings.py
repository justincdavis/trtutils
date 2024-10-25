# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

# suppress pycuda import error for docs build
with contextlib.suppress(Exception):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]
    from cuda import cudart  # type: ignore[import-untyped, import-not-found]

from ._cuda import cuda_call
from ._memory import allocate_pinned_memory

if TYPE_CHECKING:
    from typing_extensions import Self

_ALLOCATION_LOCK = Lock()


@dataclass
class Binding:
    index: int
    name: str
    dtype: np.dtype
    shape: list[int]
    is_input: bool
    allocation: int
    host_allocation: np.ndarray

    def free(self: Self) -> None:
        """Free the memory of the binding."""
        cuda_call(cudart.cudaFree(self.allocation))

    def __del__(self: Self) -> None:
        # potentially already had free called on it previously
        with contextlib.suppress(RuntimeError):
            self.free()


def allocate_bindings(
    engine: trt.IEngine,
    context: trt.IExecutionContext,
    logger: trt.ILogger,
) -> tuple[list[Binding], list[Binding], list[int], int]:
    """
    Allocate memory for the input and output tensors of a TensorRT engine.

    Parameters
    ----------
    engine : trt.IEngine
        The TensorRT engine to allocate memory for.
    context : trt.IExecutionContext
        The execution context to use.
    logger : trt.ILogger
        The logger to use.

    Returns
    -------
    tuple[list[Binding], list[Binding], list[int], int]
        A tuple containing the input bindings, output bindings, allocations, and batch size.

    Raises
    ------
    RuntimeError
        If no optimization profiles are found.
        If the profile shape is not correct.
    ValueError
        If the batch size is 0.
        If no input tensors are found.
        If no output tensors are found.
        If no memory allocations are found

    """
    # lists for allocations
    inputs = []
    outputs = []
    allocations = []
    batch_size = 0

    # magic numbers
    correct_profile_shape = 3

    # version information to compare againist
    # >= 8.5 must use tensor API, otherwise binding
    # simplify by just checking hasattr
    new_trt_api = hasattr(engine, "num_io_tensors")
    num_tensors = (
        range(engine.num_io_tensors) if new_trt_api else range(engine.num_bindings)
    )

    # based on the version of tensorrt, num_io_tensors is not available in IEngine
    # first case: version 9 or higher OR version 8.5 and higher
    for i in num_tensors:
        if new_trt_api:
            name = engine.get_tensor_name(i)
            is_input = False
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            shape = context.get_tensor_shape(name)
            if is_input and shape[0] < 0:
                if not engine.num_optimization_profiles > 0:
                    err_msg = "No optimization profiles found. Ensure that the engine has at least one optimization profile."
                    raise RuntimeError(err_msg)
                profile_shape = engine.get_tensor_profile_shape(name, 0)
                # ensure that profile shape is min,opt,max
                if len(profile_shape) != correct_profile_shape:
                    err_msg = f"Profile shape for tensor '{name}' has {len(profile_shape)} elements, expected {correct_profile_shape}"
                    raise RuntimeError(err_msg)
                # Set the *max* profile as binding shape
                context.set_input_shape(name, profile_shape[2])
                shape = context.get_tensor_shape(name)
        else:
            is_input = False
            if engine.binding_is_input(i):
                is_input = True
            name = engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
            shape = context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                if not engine.num_optimization_profiles > 0:
                    err_msg = "No optimization profiles found. Ensure that the engine has at least one optimization profile."
                    raise RuntimeError(err_msg)
                profile_shape = engine.get_profile_shape(0, name)
                # ensure that profile shape is min,opt,max
                if len(profile_shape) != correct_profile_shape:
                    err_msg = f"Profile shape for tensor '{name}' has {len(profile_shape)} elements, expected {correct_profile_shape}"
                    raise RuntimeError(err_msg)
                # Set the *max* profile as binding shape
                context.set_binding_shape(i, profile_shape[2])
                shape = context.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
        size = dtype.itemsize
        for s in shape:
            size *= s
        with _ALLOCATION_LOCK:
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = allocate_pinned_memory(size, dtype)
        # host_allocation = allocate_pinned_memory(size, dtype)
        binding = Binding(
            index=i,
            name=name,
            dtype=dtype,
            shape=list(shape),
            is_input=is_input,
            allocation=allocation,
            host_allocation=host_allocation,
        )
        allocations.append(allocation)
        if is_input:
            inputs.append(binding)
        else:
            outputs.append(binding)
        input_str = "Input" if is_input else "Output"
        log_msg = f"{input_str} '{binding.name}' with shape {binding.shape} and dtype {binding.dtype}"
        logger.log(trt.Logger.INFO, log_msg)

    if batch_size == 0:
        err_msg = "Batch size is 0. Ensure that the engine has an input tensor with a valid batch size."
        raise ValueError(err_msg)
    if len(inputs) == 0:
        err_msg = "No input tensors found. Ensure that the engine has at least one input tensor."
        raise ValueError(err_msg)
    if len(outputs) == 0:
        err_msg = "No output tensors found. Ensure that the engine has at least one output tensor."
        raise ValueError(err_msg)
    if len(allocations) == 0:
        err_msg = "No memory allocations found. Ensure that the engine has at least one input and output tensor."
        raise ValueError(err_msg)

    return inputs, outputs, allocations, batch_size
