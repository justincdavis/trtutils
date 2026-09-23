# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.compat._libs import trt

from ._buffer import Buffer, MemoryLocation

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart


def _prefix(buffer: Buffer, shape: tuple[int, ...]) -> Buffer:
    """View the leading elements of a buffer, reusing the buffer itself at its full shape."""
    return buffer if shape == buffer.shape else buffer.view(shape)


class Binding:
    """
    A TensorRT I/O tensor: its metadata plus a host and a device :class:`Buffer`.

    Both buffers are allocated at the tensor's max (profile) shape. With
    unified memory the device buffer is the device alias of the mapped host
    buffer, so there is a single allocation and no copies are needed.
    """

    def __init__(
        self: Self,
        index: int,
        name: str,
        dtype: np.dtype,
        shape: list[int],
        *,
        is_input: bool,
        host: Buffer,
        device: Buffer,
        tensor_format: trt.TensorFormat = trt.TensorFormat.LINEAR,
    ) -> None:
        """
        Create a binding from a host/device buffer pair.

        Parameters
        ----------
        index : int
            The index of the tensor in the engine.
        name : str
            The name of the tensor.
        dtype : np.dtype
            The datatype of the tensor.
        shape : list[int]
            The allocated (max) shape of the tensor.
        is_input : bool
            Whether the tensor is an engine input.
        host : Buffer
            The host buffer.
        device : Buffer
            The device buffer, or the device alias of a mapped host buffer.
        tensor_format : trt.TensorFormat, optional
            The format of the tensor.

        """
        self.index = index
        self.name = name
        self.dtype = np.dtype(dtype)
        self.shape = list(shape)
        self.is_input = is_input
        self.host = host
        self.device = device
        self.tensor_format = tensor_format

    @property
    def allocation(self: Self) -> int:
        """The device address of the binding."""
        return self.device.ptr

    @property
    def host_allocation(self: Self) -> np.ndarray:
        """The host array of the binding, at the allocated shape."""
        return self.host.array

    @property
    def pagelocked_mem(self: Self) -> bool:
        """Whether the host buffer is pagelocked."""
        return self.host.pinned

    @property
    def unified_mem(self: Self) -> bool:
        """Whether the host buffer is mapped into the device (one shared allocation)."""
        return self.host.mapped

    def stage(self: Self, src: Buffer, stream: cudart.cudaStream_t | None = None) -> Buffer:
        """
        Copy data into the device-visible memory of the binding.

        The data occupies the leading elements of the allocation, which is
        the layout TensorRT expects for a tensor smaller than the max shape.

        Parameters
        ----------
        src : Buffer
            The data to copy, on the host or the device.
        stream : cudart.cudaStream_t, optional
            The stream to enqueue the copy on.

        Returns
        -------
        Buffer
            The device view of the staged data, shaped like ``src``.

        """
        # with unified memory the host buffer *is* the device memory, so a
        # host source is a plain memcpy on the CPU
        target = self.host if self.unified_mem else self.device
        _prefix(target, src.shape).copy_from(src, stream)
        return _prefix(self.device, src.shape)

    def fetch(
        self: Self,
        shape: tuple[int, ...] | list[int] | None = None,
        stream: cudart.cudaStream_t | None = None,
    ) -> Buffer:
        """
        Copy the leading ``shape`` elements of the device memory to the host buffer.

        Parameters
        ----------
        shape : tuple[int, ...] | list[int], optional
            The shape of the data to fetch. Defaults to the allocated shape.
        stream : cudart.cudaStream_t, optional
            The stream to enqueue the copy on. The caller must synchronize
            it before reading the result.

        Returns
        -------
        Buffer
            The host view holding the data.

        """
        shape = tuple(shape) if shape is not None else tuple(self.shape)
        host = _prefix(self.host, shape)
        if not self.unified_mem:
            host.copy_from(_prefix(self.device, shape), stream)
        return host

    def free(self: Self) -> None:
        """Release the binding's buffers."""
        self.host.free()
        self.device.free()

    def __repr__(self: Self) -> str:
        """
        Get a string representation of the binding.

        Returns
        -------
        str
            The representation.

        """
        kind = "input" if self.is_input else "output"
        return (
            f"Binding({self.index}, '{self.name}', {kind}, shape={self.shape}, dtype={self.dtype})"
        )


def create_binding(
    array: np.ndarray,
    bind_id: int = 0,
    name: str = "binding",
    tensor_format: trt.TensorFormat = trt.TensorFormat.LINEAR,
    *,
    use_array_data: bool | None = None,
    is_input: bool | None = None,
    pagelocked_mem: bool | None = None,
    unified_mem: bool | None = None,
) -> Binding:
    """
    Create a binding for a TensorRT engine.

    Parameters
    ----------
    array : np.ndarray
        The array whose shape and dtype define the binding.
    bind_id : int, optional
        The index of the binding.
    name : str, optional
        The name of the binding.
    tensor_format : trt.TensorFormat, optional
        The format of the tensor.
    use_array_data : bool, optional
        Whether to copy the data from the array into the host buffer.
        By default None, which means the data will not be copied.
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
    shape = tuple(array.shape)

    if pagelocked_mem and unified_mem:
        host = Buffer.empty(shape, array.dtype, MemoryLocation.HOST, pinned=True, mapped=True)
        # the device alias holds the host array (and through it the pinned
        # allocation), so it stays valid independently of the host Buffer
        device = Buffer.from_ptr(
            host.device_ptr, shape, array.dtype, MemoryLocation.DEVICE, owner=host.array
        )
    else:
        host = Buffer.empty(shape, array.dtype, MemoryLocation.HOST, pinned=pagelocked_mem)
        device = Buffer.empty(shape, array.dtype, MemoryLocation.DEVICE)

    if use_array_data:
        np.copyto(host.array, array)

    return Binding(
        bind_id,
        name,
        array.dtype,
        list(shape),
        is_input=bool(is_input),
        host=host,
        device=device,
        tensor_format=tensor_format,
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
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::allocate_bindings")
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
    num_tensors = range(engine.num_io_tensors) if FLAGS.TRT_10 else range(engine.num_bindings)

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
            if is_input and any(dim < 0 for dim in shape):
                if not engine.num_optimization_profiles > 0:
                    err_msg = "No optimization profiles found. Ensure that the engine has at least one optimization profile."
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()
                    raise RuntimeError(err_msg)
                profile_shape = engine.get_tensor_profile_shape(name, 0)
                # ensure that profile shape is min,opt,max
                if len(profile_shape) != correct_profile_shape:
                    err_msg = f"Profile shape for tensor '{name}' has {len(profile_shape)} elements, expected {correct_profile_shape}"
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()
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
            if is_input and any(dim < 0 for dim in shape):
                if not engine.num_optimization_profiles > 0:
                    err_msg = "No optimization profiles found. Ensure that the engine has at least one optimization profile."
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()
                    raise RuntimeError(err_msg)
                profile_shape = engine.get_profile_shape(0, name)
                # ensure that profile shape is min,opt,max
                if len(profile_shape) != correct_profile_shape:
                    err_msg = f"Profile shape for tensor '{name}' has {len(profile_shape)} elements, expected {correct_profile_shape}"
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()
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
        log_msg = (
            f"{input_str}-{i} '{binding.name}' with shape {binding.shape} and dtype {binding.dtype}"
        )
        LOG.debug(log_msg)

    if len(inputs) == 0:
        err_msg = "No input tensors found. Ensure that the engine has at least one input tensor."
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        raise ValueError(err_msg)
    if len(outputs) == 0:
        err_msg = "No output tensors found. Ensure that the engine has at least one output tensor."
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        raise ValueError(err_msg)
    if len(allocations) == 0:
        err_msg = "No memory allocations found. Ensure that the engine has at least one input and output tensor."
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        raise ValueError(err_msg)

    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()
    return inputs, outputs, allocations
