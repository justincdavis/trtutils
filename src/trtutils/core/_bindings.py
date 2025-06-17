# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

with contextlib.suppress(Exception):
    import tensorrt as trt

    try:
        import cuda.bindings.runtime as cudart
    except (ImportError, ModuleNotFoundError):
        from cuda import cudart

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._cuda import cuda_call
from ._memory import allocate_pinned_memory, cuda_malloc, get_ptr_pair

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass
class Binding:
    """Small wrapper for a host/device allocation pair."""

    index: int
    name: str
    dtype: np.dtype
    shape: list[int]
    is_input: bool
    allocation: int
    host_allocation: np.ndarray
    tensor_format: trt.TensorFormat
    pagelocked_mem: bool
    unified_mem: bool

    def free(self: Self) -> None:
        """Free the memory of the binding."""
        if self.pagelocked_mem:
            cuda_call(cudart.cudaFreeHost(self.host_allocation))
        # else:
        cuda_call(cudart.cudaFree(self.allocation))

    def __del__(self: Self) -> None:
        # potentially already had free called on it previously
        with contextlib.suppress(RuntimeError):
            self.free()


def create_binding(
    array: np.ndarray,
    bind_id: int = 0,
    name: str = "binding",
    tensor_format: trt.TensorFormat = trt.TensorFormat.LINEAR,
    *,
    is_input: bool | None = None,
    pagelocked_mem: bool | None = None,
    unified_mem: bool | None = None,
) -> Binding:
    """
    Create a binding for a TensorRT engine.

    Parameters
    ----------
    array : np.ndarray
        The array to use for the binding.
    bind_id : int, optional
        The index of the binding.
    name : str, optional
        The name of the binding.
    tensor_format : trt.TensorFormat, optional
        The format of the tensor.
    is_input : bool, optional
        Whether the binding is an input or output.
    pagelocked_mem : bool, optional
        Whether or not to use pagelocked memory for host allocations.
        By default None, which means pagelocked memory will be used.
    unified_mem : bool, optional
        Whether or not the system has unified memory.
        If True, use cudaHostAllocMapped to take advantage of unified memory.

    Returns
    -------
    Binding
        The binding for the host/device memory.

    """
    pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True

    # info from the np.ndarray
    shape = array.shape
    dtype = array.dtype
    size = array.itemsize
    for s in shape:
        size *= s

    # allocate host and device memory
    if pagelocked_mem and unified_mem:
        host_allocation = allocate_pinned_memory(
            size, dtype, tuple(shape), unified_mem=unified_mem
        )
        _, device_allocation = get_ptr_pair(host_allocation)
    else:
        device_allocation = cuda_malloc(size)
        if pagelocked_mem:
            host_allocation = allocate_pinned_memory(size, dtype, tuple(shape))
        else:
            host_allocation = np.zeros(tuple(shape), dtype=dtype)

    # make the binding
    return Binding(
        bind_id,
        name,
        dtype,
        list(shape),
        bool(is_input),
        device_allocation,
        host_allocation,
        tensor_format,
        pagelocked_mem,
        bool(unified_mem),
    )


def allocate_bindings(
    engine: trt.IEngine,
    context: trt.IExecutionContext,
    *,
    pagelocked_mem: bool | None = None,
    unified_mem: bool | None = None,
) -> tuple[list[Binding], list[Binding], list[int]]:
    """
    Allocate memory for the input and output tensors of a TensorRT engine.

    Parameters
    ----------
    engine : trt.IEngine
        The TensorRT engine to allocate memory for.
    context : trt.IExecutionContext
        The execution context to use.
    pagelocked_mem : bool, optional
        Whether or not to use pagelocked memory for host allocations.
        By default None, which means pagelocked memory will be used.
    unified_mem : bool, optional
        Whether or not the system has unified memory.
        If True, use cudaHostAllocMapped to take advantage of unified memory.
        By default None, which means the default host allocation will be used.

    Returns
    -------
    tuple[list[Binding], list[Binding], list[int]]
        A tuple containing the input bindings, output bindings, and gpu memory pointers.

    Raises
    ------
    RuntimeError
        If no optimization profiles are found.
        If the profile shape is not correct.
    ValueError
        If no input tensors are found.
        If no output tensors are found.
        If no memory allocations are found

    """
    pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
    unified_mem = unified_mem if unified_mem is not None else False

    # lists for allocations
    inputs: list[Binding] = []
    outputs: list[Binding] = []
    allocations: list[int] = []

    # magic numbers
    correct_profile_shape = 3

    # version information to compare againist
    # >= 8.5 must use tensor API, otherwise binding
    # simplify by just checking hasattr
    num_tensors = (
        range(engine.num_io_tensors) if FLAGS.TRT_10 else range(engine.num_bindings)
    )

    # based on the version of tensorrt, num_io_tensors is not available in IEngine
    # first case: version 9 or higher OR version 8.5 and higher
    for i in num_tensors:
        if FLAGS.TRT_10:
            name = engine.get_tensor_name(i)
            is_input = False
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            shape = context.get_tensor_shape(name)
            data_format = engine.get_tensor_format(name)
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
            data_format = engine.get_binding_format(i)
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

        LOG.debug(f"Allocating for I/O tensor: {name} - is_input: {is_input}")

        # allocate memory and create binding
        binding = create_binding(
            np.zeros(shape, dtype),
            bind_id=i,
            name=name,
            tensor_format=data_format,
            is_input=is_input,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
        )

        allocations.append(binding.allocation)
        if is_input:
            inputs.append(binding)
        else:
            outputs.append(binding)
        input_str = "Input" if is_input else "Output"
        log_msg = f"{input_str}-{i} '{binding.name}' with shape {binding.shape} and dtype {binding.dtype}"
        LOG.debug(log_msg)

    if len(inputs) == 0:
        err_msg = "No input tensors found. Ensure that the engine has at least one input tensor."
        raise ValueError(err_msg)
    if len(outputs) == 0:
        err_msg = "No output tensors found. Ensure that the engine has at least one output tensor."
        raise ValueError(err_msg)
    if len(allocations) == 0:
        err_msg = "No memory allocations found. Ensure that the engine has at least one input and output tensor."
        raise ValueError(err_msg)

    return inputs, outputs, allocations
