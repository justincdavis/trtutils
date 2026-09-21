# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
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


class Binding:
    """
    A host/device Buffer pair with TensorRT engine I/O metadata.

    A Binding pairs a host :class:`Buffer` with a device :class:`Buffer`
    and carries the metadata TensorRT needs to treat the pair as an engine
    I/O tensor (index, name, is_input, tensor_format).

    The legacy attributes ``allocation`` (device pointer) and
    ``host_allocation`` (host numpy array) are preserved as properties, so
    existing call sites continue to work unchanged.
    """

    def __init__(
        self: Self,
        host: Buffer,
        device: Buffer,
        index: int = 0,
        name: str = "binding",
        tensor_format: trt.TensorFormat = trt.TensorFormat.LINEAR,
        *,
        is_input: bool = False,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
    ) -> None:
        """
        Create a Binding from a host Buffer and a device Buffer.

        Parameters
        ----------
        host : Buffer
            The host-side buffer of the pair.
        device : Buffer
            The device-side buffer of the pair. For unified-memory systems
            this may be a non-owning view over the host buffer's device
            alias pointer.
        index : int, optional
            The index of the binding.
        name : str, optional
            The name of the binding.
        tensor_format : trt.TensorFormat, optional
            The format of the tensor.
        is_input : bool
            Whether the binding is an input or output.
        pagelocked_mem : bool, optional
            Whether the host allocation is pagelocked.
            By default None, which means the value is derived from the
            host buffer.
        unified_mem : bool, optional
            Whether the binding was created for a unified memory system.
            By default None, which means the value is derived from the
            host buffer.

        Raises
        ------
        ValueError
            If the buffers are not a (host, device) pair.
            If the buffers differ in size.

        """
        if host.location != MemoryLocation.HOST:
            err_msg = f"Binding host buffer must reside on the host, got {host.location.value}."
            raise ValueError(err_msg)
        if device.location != MemoryLocation.DEVICE:
            err_msg = (
                f"Binding device buffer must reside on the device, got {device.location.value}."
            )
            raise ValueError(err_msg)
        if host.nbytes != device.nbytes:
            err_msg = f"Binding buffer size mismatch: host has {host.nbytes} bytes, device has {device.nbytes} bytes."
            raise ValueError(err_msg)

        self.host = host
        self.device = device
        self.index = index
        self.name = name
        self.tensor_format = tensor_format
        self.is_input = is_input
        self.dtype: np.dtype = host.dtype
        self.shape: list[int] = list(host.shape)
        self.pagelocked_mem = host.pinned if pagelocked_mem is None else pagelocked_mem
        self.unified_mem = host.mapped if unified_mem is None else unified_mem

    @classmethod
    def from_buffers(
        cls: type[Self],
        host: Buffer,
        device: Buffer | None = None,
        index: int = 0,
        name: str = "binding",
        tensor_format: trt.TensorFormat = trt.TensorFormat.LINEAR,
        *,
        is_input: bool = False,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
    ) -> Self:
        """
        Build a Binding from existing Buffers.

        Parameters
        ----------
        host : Buffer
            The host-side buffer of the pair.
        device : Buffer, optional
            The device-side buffer of the pair. If not provided, one is
            derived automatically: a non-owning view of the host buffer's
            device alias pointer when the host buffer is device-mapped,
            otherwise a fresh device allocation of matching size.
        index : int, optional
            The index of the binding.
        name : str, optional
            The name of the binding.
        tensor_format : trt.TensorFormat, optional
            The format of the tensor.
        is_input : bool
            Whether the binding is an input or output.
        pagelocked_mem : bool, optional
            Whether the host allocation is pagelocked.
            By default None, which means the value is derived from the
            host buffer.
        unified_mem : bool, optional
            Whether the binding was created for a unified memory system.
            By default None, which means the value is derived from the
            host buffer.

        Returns
        -------
        Binding
            The binding wrapping the host/device buffer pair.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("core::Binding.from_buffers")
        if device is None:
            if host.mapped:
                device = Buffer.from_ptr(
                    host.device_ptr,
                    host.shape,
                    host.dtype,
                    MemoryLocation.DEVICE,
                )
            else:
                device = Buffer.empty(host.shape, host.dtype, MemoryLocation.DEVICE)
        binding = cls(
            host,
            device,
            index,
            name,
            tensor_format,
            is_input=is_input,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
        )
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return binding

    @property
    def allocation(self: Self) -> int:
        """The device pointer of the binding."""
        return self.device.ptr

    @property
    def host_allocation(self: Self) -> np.ndarray:
        """The host numpy array of the binding."""
        return self.host.array

    def _partial(self: Self, buffer: Buffer, shape: tuple[int, ...] | None) -> Buffer:
        """
        Get the leading-axis view of a buffer matching an active shape.

        Dynamic-shape engines allocate at the max profile shape but may run
        at a smaller submitted shape, in which case only the prefix of the
        allocation participates in the copy. Only the leading (batch) axis
        may shrink; a mismatch in any other dimension is an error rather
        than a silent partial copy of the wrong region.

        Parameters
        ----------
        buffer : Buffer
            The buffer allocated at the max profile shape.
        shape : tuple[int, ...], optional
            The active shape. If None or equal to the allocated shape, the
            buffer is returned unchanged.

        Returns
        -------
        Buffer
            The buffer itself, or a non-owning view over its prefix.

        Raises
        ------
        ValueError
            If the active shape differs outside the leading axis.

        """
        if shape is None:
            return buffer
        active = tuple(int(s) for s in shape)
        allocated = tuple(int(s) for s in buffer.shape)
        if active == allocated:
            return buffer
        if len(active) != len(allocated) or active[1:] != allocated[1:]:
            err_msg = (
                f"Active shape {active} may only differ from the allocated "
                f"shape {allocated} on the leading axis."
            )
            raise ValueError(err_msg)
        return buffer[: active[0]]

    def upload(
        self: Self,
        data: np.ndarray | Buffer,
        stream: cudart.cudaStream_t | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        """
        Copy data into the binding, honoring the memory configuration.

        On unified-memory bindings host data is written directly into the
        mapped host allocation. On pagelocked bindings with a stream, the
        copy is asynchronous. Otherwise, a synchronous copy is performed.
        A device :class:`Buffer` is copied device-to-device in every mode.

        Parameters
        ----------
        data : np.ndarray | Buffer
            The data to copy into the binding: a host array, or a Buffer in
            either memory space.
        stream : cudart.cudaStream_t, optional
            The stream to utilize for asynchronous copies.
        shape : tuple[int, ...], optional
            The active shape of the binding, when a dynamic-shape engine is
            running below its allocated max profile shape. Only the matching
            prefix of the allocation is written.

        """
        on_device = isinstance(data, Buffer) and data.location == MemoryLocation.DEVICE
        if self.pagelocked_mem and self.unified_mem and not on_device:
            # host Buffers land here too, through Buffer.__array__
            np.copyto(self._partial(self.host, shape).array, data)
            return
        device = self._partial(self.device, shape)
        if stream is not None and self.pagelocked_mem:
            device.copy_from(data, stream)
            return
        device.copy_from(data)

    def download(
        self: Self,
        stream: cudart.cudaStream_t | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """
        Copy the binding's device data back to its host allocation.

        On unified-memory bindings this is a no-op and the mapped host
        allocation is returned directly. On pagelocked bindings with a
        stream, the copy is asynchronous and the caller must synchronize
        the stream before reading. Otherwise, a synchronous copy is
        performed.

        Parameters
        ----------
        stream : cudart.cudaStream_t, optional
            The stream to utilize for asynchronous copies.
        shape : tuple[int, ...], optional
            The active shape of the binding, when a dynamic-shape engine is
            running below its allocated max profile shape. Only the matching
            prefix is copied back, and the returned array is that prefix.

        Returns
        -------
        np.ndarray
            The host allocation of the binding, or the valid prefix of it
            when an active shape is given.

        """
        host = self._partial(self.host, shape)
        if self.pagelocked_mem and self.unified_mem:
            return host.array
        device = self._partial(self.device, shape)
        if stream is not None and self.pagelocked_mem:
            device.copy_to(host, stream)
        else:
            device.copy_to(host)
        return host.array

    def free(self: Self) -> None:
        """Free the memory of the binding."""
        self.device.free()
        self.host.free()

    def __del__(self: Self) -> None:
        # potentially already had free called on it previously
        with contextlib.suppress(RuntimeError, AttributeError):
            self.free()


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
        The array to use for the binding.
    bind_id : int, optional
        The index of the binding.
    name : str, optional
        The name of the binding.
    tensor_format : trt.TensorFormat, optional
        The format of the tensor.
    use_array_data : bool, optional
        Whether to use the data from the array for the binding.
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
    unified_mem = bool(unified_mem)

    # allocate the host buffer
    host = Buffer.empty(
        array.shape,
        array.dtype,
        MemoryLocation.HOST,
        pinned=pagelocked_mem,
        mapped=pagelocked_mem and unified_mem,
    )

    # copy the data from the host array to the host allocation
    if use_array_data:
        np.copyto(host.array, array)

    # make the binding, deriving the device buffer from the host buffer
    binding = Binding.from_buffers(
        host,
        None,
        bind_id,
        name,
        tensor_format,
        is_input=bool(is_input),
        pagelocked_mem=pagelocked_mem,
        unified_mem=unified_mem,
    )
    LOG.debug(
        f"Created binding: {binding.name}, shape: {binding.shape}, dtype: {binding.dtype}",
    )
    return binding


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
            if is_input and shape[0] < 0:
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
            if is_input and shape[0] < 0:
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
