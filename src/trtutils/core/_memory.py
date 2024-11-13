# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import ctypes
import logging
from threading import Lock

import numpy as np

# suppress pycuda import error for docs build
with contextlib.suppress(Exception):
    from cuda import cudart  # type: ignore[import-untyped, import-not-found]

from ._cuda import cuda_call

_MEM_ALLOC_LOCK = Lock()

_log = logging.getLogger(__name__)


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
    nbytes = host_arr.size * host_arr.itemsize
    # _log.debug(f"MemcpyHtoD: {device_ptr} with size: {nbytes}")
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr,
            host_arr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ),
    )


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
    nbytes = host_arr.size * host_arr.itemsize
    # _log.debug(f"MemcpyDtoH: {device_ptr} with size: {nbytes}")
    cuda_call(
        cudart.cudaMemcpy(
            host_arr,
            device_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        ),
    )


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
    nbytes = host_arr.size * host_arr.itemsize
    # _log.debug(f"MemcpyHtoD_Async: {device_ptr} with size: {nbytes}")
    cuda_call(
        cudart.cudaMemcpyAsync(
            device_ptr,
            host_arr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ),
    )


def memcpy_device_to_host_async(
    host_arr: np.ndarray,
    device_ptr: int,
    stream: cudart.cudaStream_t,
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

    """
    nbytes = host_arr.size * host_arr.itemsize
    # _log.debug(f"MemcpyDtoH_Async: {device_ptr} with size: {nbytes}")
    cuda_call(
        cudart.cudaMemcpyAsync(
            host_arr,
            device_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        ),
    )


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
    with _MEM_ALLOC_LOCK:
        device_ptr: int = cuda_call(cudart.cudaMalloc(nbytes))
    _log.debug(f"Allocated, device_ptr: {device_ptr}, size: {nbytes}")
    return device_ptr


def allocate_pinned_memory(
    nbytes: int,
    dtype: np.dtype,
    shape: tuple[int, ...] | None = None,
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

    Returns
    -------
    np.ndarray
        A numpy array backed by pinned memory.

    """
    # allocate pinned memory and get a pointer to it directly
    with _MEM_ALLOC_LOCK:
        host_ptr = cuda_call(cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault))

    # create the numpy array
    array_type = ctypes.c_byte * nbytes
    array: np.ndarray = np.ctypeslib.as_array(array_type.from_address(host_ptr))

    # set datatype and shape
    array = array.view(dtype)
    shape = (nbytes // dtype.itemsize,) if shape is None else shape

    _log.debug(
        f"Allocated-pagelocked, host_ptr: {host_ptr}, size: {nbytes}, shape: {shape}",
    )

    return array.reshape(shape)
