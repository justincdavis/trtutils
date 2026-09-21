# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import ctypes

import numpy as np
import nvtx

with contextlib.suppress(Exception):
    from trtutils.compat._libs import cudart

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._cuda import cuda_call


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray) -> None:
    """
    Copy a numpy array to a device pointer with error checking.

    Parameters
    ----------
    device_ptr : int
        The device pointer to copy to.
    host_arr : np.ndarray
        The numpy array to copy.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_host_to_device")
    nbytes = host_arr.size * host_arr.itemsize
    # LOG.debug(f"MemcpyHtoD: {device_ptr} with size: {nbytes}")
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr,
            host_arr.ctypes.data,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int) -> None:
    """
    Copy a device pointer to a numpy array with error checking.

    Parameters
    ----------
    host_arr : np.ndarray
        The numpy array to copy to.
    device_ptr : int
        The device pointer to copy.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_device_to_host")
    nbytes = host_arr.size * host_arr.itemsize
    # LOG.debug(f"MemcpyDtoH: {device_ptr} with size: {nbytes}")
    cuda_call(
        cudart.cudaMemcpy(
            host_arr.ctypes.data,
            device_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_host_to_device_async(
    device_ptr: int,
    host_arr: np.ndarray,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Copy a numpy array to a device pointer with error checking.

    Parameters
    ----------
    device_ptr : int
        The device pointer to copy to.
    host_arr : np.ndarray
        The numpy array to copy.
    stream : cudart.cudaStream_t
        The stream to utilize.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_host_to_device_async")
    nbytes = host_arr.size * host_arr.itemsize
    # LOG.debug(f"MemcpyHtoD_Async: {device_ptr} with size: {nbytes}")
    cuda_call(
        cudart.cudaMemcpyAsync(
            device_ptr,
            host_arr.ctypes.data,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_device_to_host_async(
    host_arr: np.ndarray,
    device_ptr: int,
    stream: cudart.cudaStream_t,
    nbytes: int | None = None,
) -> None:
    """
    Copy a device pointer to a numpy array with error checking.

    Parameters
    ----------
    host_arr : np.ndarray
        The numpy array to copy to.
    device_ptr : int
        The device pointer to copy.
    stream : cudart.cudaStream_t
        The stream to utilize.
    nbytes : int, optional
        The number of bytes to copy. By default None, which copies
        the full size of the host array. Used to copy only the valid
        prefix of an allocation (e.g. partial batches on engines built
        with a dynamic batch dimension).

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_device_to_host_async")
    if nbytes is None:
        nbytes = host_arr.size * host_arr.itemsize
    # LOG.debug(f"MemcpyDtoH_Async: {device_ptr} with size: {nbytes}")
    cuda_call(
        cudart.cudaMemcpyAsync(
            host_arr.ctypes.data,
            device_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_device_to_device(
    dst_ptr: int,
    src_ptr: int,
    nbytes: int,
) -> None:
    """
    Copy from one device pointer to another with error checking.

    Parameters
    ----------
    dst_ptr : int
        The destination device pointer.
    src_ptr : int
        The source device pointer.
    nbytes : int
        The number of bytes to copy.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_device_to_device")
    cuda_call(
        cudart.cudaMemcpy(
            dst_ptr,
            src_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_device_to_device_async(
    dst_ptr: int,
    src_ptr: int,
    nbytes: int,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Copy from one device pointer to another asynchronously.

    Parameters
    ----------
    dst_ptr : int
        The destination device pointer.
    src_ptr : int
        The source device pointer.
    nbytes : int
        The number of bytes to copy.
    stream : cudart.cudaStream_t
        The stream to utilize.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_device_to_device_async")
    cuda_call(
        cudart.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            stream,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_host_to_device_offset(
    device_ptr: int,
    host_arr: np.ndarray,
    offset_bytes: int,
) -> None:
    """
    Copy a numpy array to a device pointer at a specific offset.

    Parameters
    ----------
    device_ptr : int
        The base device pointer.
    host_arr : np.ndarray
        The numpy array to copy.
    offset_bytes : int
        The byte offset into the device buffer.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_host_to_device_offset")
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr + offset_bytes,
            host_arr.ctypes.data,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_host_to_device_offset_async(
    device_ptr: int,
    host_arr: np.ndarray,
    offset_bytes: int,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Copy a numpy array to a device pointer at a specific offset asynchronously.

    Parameters
    ----------
    device_ptr : int
        The base device pointer.
    host_arr : np.ndarray
        The numpy array to copy.
    offset_bytes : int
        The byte offset into the device buffer.
    stream : cudart.cudaStream_t
        The stream to utilize.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_host_to_device_offset_async")
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpyAsync(
            device_ptr + offset_bytes,
            host_arr.ctypes.data,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def cuda_malloc(
    nbytes: int,
) -> int:
    """
    Perform a memory allocation using cudart.cudaMalloc.

    Parameters
    ----------
    nbytes : int
        The number of bytes to allocate.

    Returns
    -------
    int
        The pointer to the allocated memory.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::cuda_malloc")
    device_ptr: int = cuda_call(cudart.cudaMalloc(nbytes))
    LOG.debug(f"Allocated, device_ptr: {device_ptr}, size: {nbytes}")
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()
    return device_ptr


def allocate_pinned_memory(
    nbytes: int,
    dtype: np.dtype,
    shape: tuple[int, ...] | None = None,
    *,
    unified_mem: bool | None = None,
) -> np.ndarray:
    """
    Allocate pinned (page-locked) memory on the host, required for asynchronous memory transfers.

    The shape of the pagelocked memory is a 1D numpy array, so CPU side reshaping
    is required for some applications. If shape is passed, then the shape will not
    be 1D, but memory transfer may have complications.

    Parameters
    ----------
    nbytes : int
        The number of bytes to allocate.
    dtype : np.dtype
        The data type for the allocated memory.
    shape : tuple[int, ...], optional
        An optional shape for the pagelocked memory array.
        If not provided, the array will be 1D.
    unified_mem : bool, optional
        If True, use cudaHostAllocMapped to take advantage of unified memory.

    Returns
    -------
    np.ndarray
        A numpy array backed by pinned memory.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::allocate_pinned_memory")
    flags = cudart.cudaHostAllocMapped if unified_mem else cudart.cudaHostAllocDefault
    # allocate pinned memory and get a pointer to it directly
    host_ptr = cuda_call(cudart.cudaHostAlloc(nbytes, flags))

    # create the numpy array
    array_type = ctypes.c_byte * nbytes
    array: np.ndarray = np.ctypeslib.as_array(array_type.from_address(host_ptr))

    # set datatype and shape
    array = array.view(dtype)
    shape = (nbytes // dtype.itemsize,) if shape is None else shape

    LOG.debug(
        f"Allocated-pagelocked, host_ptr: {host_ptr}, size: {nbytes}, shape: {shape}",
    )

    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()
    return array.reshape(shape)


def get_ptr_pair(host_array: np.ndarray) -> tuple[int, int]:
    """
    Get the pointer pairs (host/device) of a pagelocked allocation.

    Parameters
    ----------
    host_array : np.ndarray
        A np.ndarray allocated by the allocate_pinned_memory function.

    Returns
    -------
    tuple[int, int]
        The host and device pointer of the allocation.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::get_ptr_pair")
    host_ptr = host_array.ctypes.data
    device_ptr = cuda_call(cudart.cudaHostGetDevicePointer(host_ptr, 0))

    LOG.debug(f"Acquired pointers: (host: {host_ptr}, device: {device_ptr}) from ndarray")

    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()
    return host_ptr, device_ptr


def allocate_managed_memory(
    nbytes: int,
    stream: cudart.cudaStream_t | None = None,
) -> int:
    """
    Allocate managed memory.

    Parameters
    ----------
    nbytes : int
        The number of bytes to allocate.
    stream : cudart.cudaStream_t, optional
        The stream to utilize.

    Returns
    -------
    int
        The pointer to the allocated memory.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::allocate_managed_memory")
    device_ptr: int = cuda_call(cudart.cudaMallocManaged(nbytes, cudart.cudaMemAttachGlobal))

    # if a stream is provided, we should attach the memory
    if stream is not None:
        cuda_call(cudart.cudaStreamAttachMemAsync(stream, device_ptr, 0, cudart.cudaMemAttachGlobal))

    LOG.debug(f"Allocated-managed, device_ptr: {device_ptr}, size: {nbytes}")
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()
    return device_ptr


def cuda_free(device_ptr: int) -> None:
    """
    Free a CUDA device pointer.

    Parameters
    ----------
    device_ptr : int
        The device pointer to free.

    """
    cuda_call(cudart.cudaFree(device_ptr))


def cuda_host_free(host_ptr: int | np.ndarray) -> None:
    """
    Free a CUDA host pointer.

    Parameters
    ----------
    host_ptr : int
        The host pointer to free.

    """
    if isinstance(host_ptr, np.ndarray):
        host_ptr = host_ptr.ctypes.data
    cuda_call(cudart.cudaFreeHost(host_ptr))


def free_device_ptrs(ptrs: list[int]) -> None:
    """
    Free a list of CUDA device pointers.

    Parameters
    ----------
    ptrs : list[int]
        The device pointers to free.

    """
    for p in ptrs:
        cuda_free(p)


def allocate_to_device(
    data: list[np.ndarray],
) -> list[int]:
    """
    Allocate device memory for each numpy array and copy the data over.

    Parameters
    ----------
    data : list[np.ndarray]
        The numpy arrays to copy.

    Returns
    -------
    list[int]
        The device pointers to the allocated memory.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::allocate_to_device")
    device_ptrs: list[int] = []
    for arr in data:
        ptr = cuda_malloc(arr.nbytes)
        memcpy_host_to_device(ptr, arr)
        device_ptrs.append(ptr)
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()
    return device_ptrs


def memcpy_2d(
    dst_ptr: int,
    dpitch: int,
    src_ptr: int,
    spitch: int,
    width_bytes: int,
    height: int,
    kind: cudart.cudaMemcpyKind,
) -> None:
    """
    Copy a pitched 2D region of memory with error checking.

    Copies ``height`` rows of ``width_bytes`` bytes each, where consecutive
    rows are ``spitch`` bytes apart in the source and ``dpitch`` bytes apart
    in the destination.

    Parameters
    ----------
    dst_ptr : int
        The destination pointer.
    dpitch : int
        The pitch (in bytes) between destination rows.
    src_ptr : int
        The source pointer.
    spitch : int
        The pitch (in bytes) between source rows.
    width_bytes : int
        The number of bytes to copy per row.
    height : int
        The number of rows to copy.
    kind : cudart.cudaMemcpyKind
        The direction of the copy.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_2d")
    cuda_call(
        cudart.cudaMemcpy2D(
            dst_ptr,
            dpitch,
            src_ptr,
            spitch,
            width_bytes,
            height,
            kind,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_2d_async(
    dst_ptr: int,
    dpitch: int,
    src_ptr: int,
    spitch: int,
    width_bytes: int,
    height: int,
    kind: cudart.cudaMemcpyKind,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Copy a pitched 2D region of memory asynchronously with error checking.

    Copies ``height`` rows of ``width_bytes`` bytes each, where consecutive
    rows are ``spitch`` bytes apart in the source and ``dpitch`` bytes apart
    in the destination.

    Parameters
    ----------
    dst_ptr : int
        The destination pointer.
    dpitch : int
        The pitch (in bytes) between destination rows.
    src_ptr : int
        The source pointer.
    spitch : int
        The pitch (in bytes) between source rows.
    width_bytes : int
        The number of bytes to copy per row.
    height : int
        The number of rows to copy.
    kind : cudart.cudaMemcpyKind
        The direction of the copy.
    stream : cudart.cudaStream_t
        The stream to utilize.

    """
    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::memcpy_2d_async")
    cuda_call(
        cudart.cudaMemcpy2DAsync(
            dst_ptr,
            dpitch,
            src_ptr,
            spitch,
            width_bytes,
            height,
            kind,
            stream,
        ),
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def _contiguous_strides(shape: tuple[int, ...], itemsize: int) -> list[int]:
    """
    Compute the C-contiguous byte strides for a shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape to compute strides for.
    itemsize : int
        The size of each element in bytes.

    Returns
    -------
    list[int]
        The C-contiguous byte strides.

    """
    strides = [itemsize] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return strides


def _memcpy_nd(
    host_arr: np.ndarray,
    device_ptr: int,
    kind: cudart.cudaMemcpyKind,
    stream: cudart.cudaStream_t | None,
) -> None:
    """
    Copy between a strided N-D numpy array and contiguous device memory.

    The transfer is planned by collapsing the contiguous trailing dimensions
    of the host array into rows, mapping the next dimension onto a pitched
    2D copy, and iterating any remaining leading dimensions (supporting 4D,
    5D, and beyond). Contiguous arrays degrade to a single flat copy.

    Parameters
    ----------
    host_arr : np.ndarray
        The host array (possibly a non-contiguous view).
    device_ptr : int
        The pointer to the contiguous device memory.
    kind : cudart.cudaMemcpyKind
        The direction of the copy (host-to-device or device-to-host).
    stream : cudart.cudaStream_t, optional
        If provided, copies are issued asynchronously on the stream.

    """
    h2d = kind == cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    nbytes = host_arr.size * host_arr.itemsize

    # fast path: fully contiguous host arrays are a single flat copy
    if host_arr.flags["C_CONTIGUOUS"]:
        if h2d:
            if stream is not None:
                memcpy_host_to_device_async(device_ptr, host_arr, stream)
            else:
                memcpy_host_to_device(device_ptr, host_arr)
        elif stream is not None:
            memcpy_device_to_host_async(host_arr, device_ptr, stream)
        else:
            memcpy_device_to_host(host_arr, device_ptr)
        return

    if FLAGS.NVTX_ENABLED:
        nvtx.push_range("core::_memcpy_nd")

    itemsize = host_arr.itemsize
    shape = host_arr.shape
    h_strides = host_arr.strides
    d_strides = _contiguous_strides(shape, itemsize)
    host_base = host_arr.ctypes.data

    # collapse the contiguous trailing dimensions into a row of width bytes
    width = itemsize
    pitch_dim = len(shape) - 1
    while pitch_dim >= 0 and h_strides[pitch_dim] == width:
        width *= shape[pitch_dim]
        pitch_dim -= 1

    # the strides can describe a fully contiguous layout even when the
    # C_CONTIGUOUS flag is unset (e.g. some size-1 dimension views)
    if pitch_dim < 0:
        if h2d:
            if stream is not None:
                cuda_call(
                    cudart.cudaMemcpyAsync(device_ptr, host_base, nbytes, kind, stream),
                )
            else:
                cuda_call(cudart.cudaMemcpy(device_ptr, host_base, nbytes, kind))
        elif stream is not None:
            cuda_call(
                cudart.cudaMemcpyAsync(host_base, device_ptr, nbytes, kind, stream),
            )
        else:
            cuda_call(cudart.cudaMemcpy(host_base, device_ptr, nbytes, kind))
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return

    # rows along pitch_dim, all outer dimensions are iterated on the host
    height = shape[pitch_dim]
    spitch = h_strides[pitch_dim]
    outer_shape = shape[:pitch_dim]

    # cudaMemcpy2D requires pitch >= width and positive; otherwise (negative
    # or overlapping strides from reversed/broadcast views) copy row by row
    pitched = spitch >= width

    for outer_idx in np.ndindex(outer_shape):
        host_off = sum(i * st for i, st in zip(outer_idx, h_strides))
        dev_off = sum(i * st for i, st in zip(outer_idx, d_strides))
        host_ptr = host_base + host_off
        dev_ptr = device_ptr + dev_off
        dst_ptr, dpitch, src_ptr = (dev_ptr, width, host_ptr) if h2d else (host_ptr, spitch, dev_ptr)
        src_pitch = spitch if h2d else width
        if pitched:
            if stream is not None:
                memcpy_2d_async(dst_ptr, dpitch, src_ptr, src_pitch, width, height, kind, stream)
            else:
                memcpy_2d(dst_ptr, dpitch, src_ptr, src_pitch, width, height, kind)
        else:
            for row in range(height):
                row_host = host_ptr + row * spitch
                row_dev = dev_ptr + row * width
                row_dst, row_src = (row_dev, row_host) if h2d else (row_host, row_dev)
                if stream is not None:
                    cuda_call(
                        cudart.cudaMemcpyAsync(row_dst, row_src, width, kind, stream),
                    )
                else:
                    cuda_call(cudart.cudaMemcpy(row_dst, row_src, width, kind))

    LOG.debug(
        f"MemcpyND: shape={shape}, width={width}, height={height}, nbytes={nbytes}",
    )
    if FLAGS.NVTX_ENABLED:
        nvtx.pop_range()


def memcpy_nd_host_to_device(
    device_ptr: int,
    host_arr: np.ndarray,
) -> None:
    """
    Copy a possibly non-contiguous N-D numpy array to contiguous device memory.

    Supports arrays of any rank (designed and tested up to 5D), including
    strided views such as transposes, stepped slices, and reversed axes.
    Contiguous arrays degrade to a single flat copy.

    Parameters
    ----------
    device_ptr : int
        The device pointer to copy to.
    host_arr : np.ndarray
        The numpy array (or view) to copy.

    """
    _memcpy_nd(host_arr, device_ptr, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, None)


def memcpy_nd_host_to_device_async(
    device_ptr: int,
    host_arr: np.ndarray,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Copy a possibly non-contiguous N-D numpy array to device memory async.

    Supports arrays of any rank (designed and tested up to 5D), including
    strided views such as transposes, stepped slices, and reversed axes.
    The caller must keep the host array alive and unmodified until the
    stream is synchronized.

    Parameters
    ----------
    device_ptr : int
        The device pointer to copy to.
    host_arr : np.ndarray
        The numpy array (or view) to copy.
    stream : cudart.cudaStream_t
        The stream to utilize.

    """
    _memcpy_nd(host_arr, device_ptr, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)


def memcpy_nd_device_to_host(
    host_arr: np.ndarray,
    device_ptr: int,
) -> None:
    """
    Copy contiguous device memory into a possibly non-contiguous N-D array.

    Supports arrays of any rank (designed and tested up to 5D), including
    strided views such as transposes, stepped slices, and reversed axes.
    Contiguous arrays degrade to a single flat copy.

    Parameters
    ----------
    host_arr : np.ndarray
        The numpy array (or view) to copy into.
    device_ptr : int
        The device pointer to copy from.

    """
    _memcpy_nd(host_arr, device_ptr, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, None)


def memcpy_nd_device_to_host_async(
    host_arr: np.ndarray,
    device_ptr: int,
    stream: cudart.cudaStream_t,
) -> None:
    """
    Copy contiguous device memory into a non-contiguous N-D array async.

    Supports arrays of any rank (designed and tested up to 5D), including
    strided views such as transposes, stepped slices, and reversed axes.
    The caller must keep the host array alive until the stream is
    synchronized.

    Parameters
    ----------
    host_arr : np.ndarray
        The numpy array (or view) to copy into.
    device_ptr : int
        The device pointer to copy from.
    stream : cudart.cudaStream_t
        The stream to utilize.

    """
    _memcpy_nd(host_arr, device_ptr, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
