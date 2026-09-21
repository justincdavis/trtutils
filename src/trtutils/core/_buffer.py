# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import ctypes
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG

with contextlib.suppress(Exception):
    from trtutils.compat._libs import cudart

from ._cuda import cuda_call
from ._memory import (
    allocate_pinned_memory,
    cuda_malloc,
    get_ptr_pair,
    memcpy_device_to_device,
    memcpy_device_to_device_async,
    memcpy_device_to_host,
    memcpy_device_to_host_async,
    memcpy_host_to_device,
    memcpy_host_to_device_async,
    memcpy_nd_device_to_host,
    memcpy_nd_device_to_host_async,
    memcpy_nd_host_to_device,
    memcpy_nd_host_to_device_async,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class MemoryLocation(Enum):
    """The physical location of a :class:`Buffer` allocation."""

    HOST = "host"
    DEVICE = "device"


class Buffer:
    """
    A single typed memory allocation residing on either the host or the device.

    A Buffer is *one* allocation in *one* memory space, paired with the
    dtype/shape metadata required to interpret it. This is in contrast to
    :class:`Binding`, which is a host/device *pair* carrying TensorRT engine
    I/O metadata (index, name, is_input, tensor_format).

    Host buffers wrap a numpy array (optionally pagelocked/pinned, optionally
    mapped for unified memory) and expose ``__array__`` for zero-copy numpy
    interop. Device buffers wrap a raw CUDA allocation and expose
    ``__cuda_array_interface__`` for zero-copy interop with CuPy, Numba,
    and PyTorch.

    Prefer the classmethod constructors :meth:`Buffer.empty`,
    :meth:`Buffer.from_array`, and :meth:`Buffer.from_ptr` over ``__init__``.
    """

    def __init__(
        self: Self,
        location: MemoryLocation,
        dtype: np.dtype,
        shape: tuple[int, ...],
        ptr: int,
        host_array: np.ndarray | None = None,
        *,
        pinned: bool = False,
        mapped: bool = False,
        owns_memory: bool = True,
    ) -> None:
        """
        Create a Buffer from pre-allocated memory.

        Parameters
        ----------
        location : MemoryLocation
            Where the allocation resides.
        dtype : np.dtype
            The datatype of the elements in the buffer.
        shape : tuple[int, ...]
            The logical shape of the buffer.
        ptr : int
            The pointer to the allocation in its memory space.
        host_array : np.ndarray, optional
            For host buffers, the numpy array backing the allocation.
        pinned : bool
            Whether a host allocation is pagelocked. Ignored for device buffers.
        mapped : bool
            Whether a pinned host allocation is mapped into the device address
            space (cudaHostAllocMapped / unified memory). Ignored for device buffers.
        owns_memory : bool
            Whether this Buffer is responsible for freeing the allocation.
            Non-owning buffers (views) never free on :meth:`free` or deletion.

        Raises
        ------
        ValueError
            If a host buffer is created without a backing array.

        """
        if location == MemoryLocation.HOST and host_array is None:
            err_msg = "Host buffers require a backing numpy array."
            raise ValueError(err_msg)

        self._location = location
        self._dtype = np.dtype(dtype)
        self._shape = tuple(int(s) for s in shape)
        self._ptr = ptr
        self._host_array = host_array
        self._pinned = pinned
        self._mapped = mapped
        self._owns_memory = owns_memory
        self._freed = False
        self._device_alias_ptr: int | None = None
        # keepalive reference for non-owning views: the parent Buffer for
        # views derived from another Buffer, or the foreign owner of the
        # allocation for views built via from_ptr
        self._parent: object | None = None

        if location == MemoryLocation.HOST and pinned and mapped:
            # host_array is always provided for host allocations
            _, self._device_alias_ptr = get_ptr_pair(host_array)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(
        cls: type[Self],
        shape: tuple[int, ...],
        dtype: np.dtype,
        location: MemoryLocation = MemoryLocation.HOST,
        *,
        pinned: bool | None = None,
        mapped: bool | None = None,
    ) -> Self:
        """
        Allocate a new, uninitialized Buffer.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the buffer to allocate.
        dtype : np.dtype
            The datatype of the elements in the buffer.
        location : MemoryLocation
            Where to allocate the buffer. Default is host.
        pinned : bool, optional
            For host buffers, whether to allocate pagelocked memory.
            By default None, which means pinned memory will be used.
        mapped : bool, optional
            For pinned host buffers, whether to map the allocation into the
            device address space (unified memory systems).
            By default None, which means the allocation is not mapped.

        Returns
        -------
        Buffer
            The newly allocated buffer.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("core::Buffer.empty")
        dtype = np.dtype(dtype)
        nbytes = dtype.itemsize
        for s in shape:
            nbytes *= int(s)

        pinned = pinned if pinned is not None else True
        mapped = mapped if mapped is not None else False

        if location == MemoryLocation.DEVICE:
            ptr = cuda_malloc(nbytes)
            buffer = cls(location, dtype, tuple(shape), ptr)
        elif pinned:
            host_array = allocate_pinned_memory(
                nbytes,
                dtype,
                tuple(shape),
                unified_mem=mapped,
            )
            buffer = cls(
                location,
                dtype,
                tuple(shape),
                host_array.ctypes.data,
                host_array,
                pinned=True,
                mapped=mapped,
            )
        else:
            host_array = np.zeros(tuple(shape), dtype=dtype)
            buffer = cls(
                location,
                dtype,
                tuple(shape),
                host_array.ctypes.data,
                host_array,
            )

        LOG.debug(
            f"Buffer.empty: location={location.value}, shape={buffer.shape}, dtype={buffer.dtype}, ptr={buffer.ptr}",
        )
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return buffer

    @classmethod
    def from_array(
        cls: type[Self],
        array: np.ndarray,
        location: MemoryLocation = MemoryLocation.HOST,
        *,
        pinned: bool | None = None,
        mapped: bool | None = None,
    ) -> Self:
        """
        Allocate a new Buffer and copy the contents of a numpy array into it.

        Parameters
        ----------
        array : np.ndarray
            The array whose shape, dtype, and data are used.
        location : MemoryLocation
            Where to allocate the buffer. Default is host.
        pinned : bool, optional
            For host buffers, whether to allocate pagelocked memory.
            By default None, which means pinned memory will be used.
        mapped : bool, optional
            For pinned host buffers, whether to map the allocation into the
            device address space (unified memory systems).
            By default None, which means the allocation is not mapped.

        Returns
        -------
        Buffer
            The newly allocated buffer containing a copy of the array data.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("core::Buffer.from_array")
        buffer = cls.empty(
            array.shape,
            array.dtype,
            location,
            pinned=pinned,
            mapped=mapped,
        )
        if location == MemoryLocation.DEVICE:
            memcpy_nd_host_to_device(buffer.ptr, array)
        else:
            np.copyto(buffer.array, array)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return buffer

    @classmethod
    def from_ptr(
        cls: type[Self],
        ptr: int,
        shape: tuple[int, ...],
        dtype: np.dtype,
        location: MemoryLocation = MemoryLocation.DEVICE,
        owner: object | None = None,
    ) -> Self:
        """
        Create a non-owning Buffer view over existing memory.

        The returned Buffer will never free the underlying allocation. This is
        useful for viewing the host/device halves of a :class:`Binding` or
        memory allocated by another library.

        Parameters
        ----------
        ptr : int
            The pointer to the existing allocation.
        shape : tuple[int, ...]
            The logical shape of the memory.
        dtype : np.dtype
            The datatype of the elements in the memory.
        location : MemoryLocation
            Where the allocation resides. Default is device.
        owner : object, optional
            The object that owns the underlying allocation. A reference is
            held for the lifetime of the view, so the owner cannot be
            garbage collected out from under it. Required when wrapping
            memory owned by another library.

        Returns
        -------
        Buffer
            A non-owning view of the memory.

        """
        dtype = np.dtype(dtype)
        host_array: np.ndarray | None = None
        if location == MemoryLocation.HOST:
            nbytes = dtype.itemsize
            for s in shape:
                nbytes *= int(s)
            array_type = ctypes.c_byte * nbytes
            host_array = (
                np.ctypeslib.as_array(array_type.from_address(ptr)).view(dtype).reshape(shape)
            )
        buffer = cls(
            location,
            dtype,
            tuple(shape),
            ptr,
            host_array,
            owns_memory=False,
        )
        # keep the owning object alive for the lifetime of the view
        buffer._parent = owner
        return buffer

    @classmethod
    def from_cuda_array(cls: type[Self], obj: object) -> Self:
        """
        Create a non-owning device Buffer view over any CUDA-array-interface object.

        Accepts CuPy / Numba / PyTorch tensors, nvImageCodec images, another
        Buffer, or anything else exposing ``__cuda_array_interface__``. The
        object is kept alive for the lifetime of the view.

        Parameters
        ----------
        obj : object
            An object exposing the CUDA Array Interface.

        Returns
        -------
        Buffer
            A non-owning device view of the object's memory.

        Raises
        ------
        TypeError
            If the object does not expose ``__cuda_array_interface__``.
        ValueError
            If the memory is not C-contiguous.

        """
        interface = getattr(obj, "__cuda_array_interface__", None)
        if interface is None:
            err_msg = f"{type(obj).__name__} does not expose __cuda_array_interface__."
            raise TypeError(err_msg)
        shape = tuple(int(s) for s in interface["shape"])
        dtype = np.dtype(interface["typestr"])
        strides = interface.get("strides")
        if strides is not None and tuple(strides) != tuple(np.empty(shape, dtype=dtype).strides):
            err_msg = (
                f"Buffer views require C-contiguous memory, got strides {strides} for shape {shape}."
            )
            raise ValueError(err_msg)
        ptr, _ = interface["data"]
        return cls.from_ptr(int(ptr), shape, dtype, MemoryLocation.DEVICE, owner=obj)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def location(self: Self) -> MemoryLocation:
        """The memory space this buffer resides in."""
        return self._location

    @property
    def dtype(self: Self) -> np.dtype:
        """The datatype of the elements in the buffer."""
        return self._dtype

    @property
    def shape(self: Self) -> tuple[int, ...]:
        """The logical shape of the buffer."""
        return self._shape

    @property
    def size(self: Self) -> int:
        """The number of elements in the buffer."""
        size = 1
        for s in self._shape:
            size *= s
        return size

    @property
    def nbytes(self: Self) -> int:
        """The size of the buffer in bytes."""
        return self.size * self._dtype.itemsize

    @property
    def ptr(self: Self) -> int:
        """The pointer to the allocation in its native memory space."""
        return self._ptr

    @property
    def device_ptr(self: Self) -> int:
        """
        The device-visible pointer for this buffer.

        For device buffers this is the allocation itself. For mapped (unified)
        pinned host buffers this is the device alias pointer.

        Raises
        ------
        RuntimeError
            If the buffer is a host buffer that is not device-mapped.

        """
        if self._location == MemoryLocation.DEVICE:
            return self._ptr
        if self._device_alias_ptr is not None:
            return self._device_alias_ptr
        err_msg = "Host buffer is not mapped into the device address space."
        raise RuntimeError(err_msg)

    @property
    def array(self: Self) -> np.ndarray:
        """
        The numpy array backing a host buffer (zero-copy).

        Raises
        ------
        RuntimeError
            If the buffer resides on the device.

        """
        if self._host_array is None:
            err_msg = (
                "Device buffers have no host array. Use to_host() or numpy() to copy the data back."
            )
            raise RuntimeError(err_msg)
        return self._host_array

    @property
    def pinned(self: Self) -> bool:
        """Whether a host buffer is pagelocked."""
        return self._pinned

    @property
    def mapped(self: Self) -> bool:
        """Whether a pinned host buffer is mapped into the device address space."""
        return self._mapped

    @property
    def owns_memory(self: Self) -> bool:
        """Whether this buffer is responsible for freeing its allocation."""
        return self._owns_memory

    # ------------------------------------------------------------------
    # numpy / CUDA interop
    # ------------------------------------------------------------------

    def __array__(self: Self, dtype: np.dtype | None = None) -> np.ndarray:
        """
        Support implicit numpy conversion for host buffers.

        Parameters
        ----------
        dtype : np.dtype, optional
            An optional dtype to convert to.

        Returns
        -------
        np.ndarray
            The backing host array.

        """
        arr = self.array
        return arr.astype(dtype) if dtype is not None else arr

    @property
    def __cuda_array_interface__(self: Self) -> dict:
        """
        CUDA Array Interface (v3) for zero-copy device interop.

        Valid for device buffers and mapped pinned host buffers. Allows the
        buffer to be consumed directly by CuPy, Numba, PyTorch, etc.

        Returns
        -------
        dict
            The CUDA array interface description.

        """
        return {
            "shape": self._shape,
            "typestr": self._dtype.str,
            "data": (self.device_ptr, False),
            "version": 3,
            "strides": None,
        }

    # ------------------------------------------------------------------
    # transfers
    # ------------------------------------------------------------------

    def to_device(
        self: Self,
        stream: cudart.cudaStream_t | None = None,
    ) -> Buffer:
        """
        Copy this buffer to the device.

        If the buffer already resides on the device, a new device buffer is
        allocated and the data is copied (device-to-device).

        Parameters
        ----------
        stream : cudart.cudaStream_t, optional
            If provided, the copy is performed asynchronously on the stream.
            The caller is responsible for synchronizing the stream before
            reading the result. Async host-to-device copies should use
            pinned host buffers.

        Returns
        -------
        Buffer
            A new device buffer containing a copy of the data.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("core::Buffer.to_device")
        dst = Buffer.empty(self._shape, self._dtype, MemoryLocation.DEVICE)
        self.copy_to(dst, stream)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return dst

    def to_host(
        self: Self,
        stream: cudart.cudaStream_t | None = None,
        *,
        pinned: bool | None = None,
    ) -> Buffer:
        """
        Copy this buffer to the host.

        If the buffer already resides on the host, a new host buffer is
        allocated and the data is copied.

        Parameters
        ----------
        stream : cudart.cudaStream_t, optional
            If provided, the copy is performed asynchronously on the stream.
            The caller is responsible for synchronizing the stream before
            reading the result.
        pinned : bool, optional
            Whether the new host buffer should be pagelocked.
            By default None, which means pinned memory will be used.

        Returns
        -------
        Buffer
            A new host buffer containing a copy of the data.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("core::Buffer.to_host")
        dst = Buffer.empty(self._shape, self._dtype, MemoryLocation.HOST, pinned=pinned)
        self.copy_to(dst, stream)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return dst

    def copy_to(
        self: Self,
        dst: Buffer | np.ndarray,
        stream: cudart.cudaStream_t | None = None,
    ) -> None:
        """
        Copy the contents of this buffer into another buffer or numpy array.

        For Buffer destinations, the direction of the copy (H2D, D2H, D2D,
        H2H) is dispatched from the locations of the two buffers. Numpy array
        destinations are treated as host memory and may be non-contiguous
        N-D views (designed and tested up to 5D), in which case a strided
        transfer is performed.

        Parameters
        ----------
        dst : Buffer | np.ndarray
            The destination. Must have the same number of bytes.
        stream : cudart.cudaStream_t, optional
            If provided, the copy is performed asynchronously on the stream.
            Host-to-host copies are always synchronous.

        Raises
        ------
        ValueError
            If the source and destination sizes do not match.

        """
        if isinstance(dst, np.ndarray):
            if dst.size * dst.itemsize != self.nbytes:
                err_msg = f"Buffer size mismatch: src has {self.nbytes} bytes, dst has {dst.size * dst.itemsize} bytes."
                raise ValueError(err_msg)
            if self._location == MemoryLocation.HOST:
                np.copyto(dst, self.array.reshape(dst.shape))
            elif stream is not None:
                memcpy_nd_device_to_host_async(dst, self._ptr, stream)
            else:
                memcpy_nd_device_to_host(dst, self._ptr)
            return
        if self.nbytes != dst.nbytes:
            err_msg = (
                f"Buffer size mismatch: src has {self.nbytes} bytes, dst has {dst.nbytes} bytes."
            )
            raise ValueError(err_msg)

        src_loc, dst_loc = self._location, dst.location
        if src_loc == MemoryLocation.HOST and dst_loc == MemoryLocation.HOST:
            np.copyto(dst.array, self.array.reshape(dst.array.shape))
        elif src_loc == MemoryLocation.HOST and dst_loc == MemoryLocation.DEVICE:
            if stream is not None:
                memcpy_host_to_device_async(dst.ptr, self.array, stream)
            else:
                memcpy_host_to_device(dst.ptr, self.array)
        elif src_loc == MemoryLocation.DEVICE and dst_loc == MemoryLocation.HOST:
            if stream is not None:
                memcpy_device_to_host_async(dst.array, self._ptr, stream)
            else:
                memcpy_device_to_host(dst.array, self._ptr)
        elif stream is not None:
            memcpy_device_to_device_async(dst.ptr, self._ptr, self.nbytes, stream)
        else:
            memcpy_device_to_device(dst.ptr, self._ptr, self.nbytes)

    def copy_from(
        self: Self,
        src: Buffer | np.ndarray,
        stream: cudart.cudaStream_t | None = None,
    ) -> None:
        """
        Copy the contents of another buffer or numpy array into this buffer.

        Parameters
        ----------
        src : Buffer | np.ndarray
            The source of the data. Numpy arrays are treated as host memory
            and may be non-contiguous N-D views (designed and tested up to
            5D), in which case a strided transfer is performed.
        stream : cudart.cudaStream_t, optional
            If provided, the copy is performed asynchronously on the stream.
            For asynchronous copies the array must be kept alive and
            unmodified until the stream is synchronized.

        Raises
        ------
        ValueError
            If the source and destination sizes do not match.

        """
        if isinstance(src, np.ndarray):
            if src.size * src.itemsize != self.nbytes:
                err_msg = f"Buffer size mismatch: src has {src.size * src.itemsize} bytes, dst has {self.nbytes} bytes."
                raise ValueError(err_msg)
            if self._location == MemoryLocation.HOST:
                np.copyto(self.array.reshape(src.shape), src)
            elif stream is not None:
                memcpy_nd_host_to_device_async(self._ptr, src, stream)
            else:
                memcpy_nd_host_to_device(self._ptr, src)
            return
        src.copy_to(self, stream)

    def numpy(self: Self) -> np.ndarray:
        """
        Get the buffer contents as a numpy array.

        For host buffers this is zero-copy (the backing array is returned).
        For device buffers a synchronous device-to-host copy is performed
        into a fresh, pageable numpy array.

        Returns
        -------
        np.ndarray
            The contents of the buffer.

        """
        if self._location == MemoryLocation.HOST:
            return self.array
        out = np.empty(self._shape, dtype=self._dtype)
        memcpy_device_to_host(out, self._ptr)
        return out

    def __getitem__(self: Self, key: int | slice) -> Buffer:
        """
        Get a non-owning view of a sub-region along the leading axis.

        Integer keys select one entry of the leading dimension; slice keys
        (step 1 only) select a range. The returned Buffer is a contiguous,
        non-owning view over the same memory, useful for transferring into
        or out of individual batch slots of 4D/5D buffers. The view holds a
        reference to this buffer, keeping the allocation alive.

        Parameters
        ----------
        key : int | slice
            The index or slice along the leading axis.

        Returns
        -------
        Buffer
            A non-owning view over the selected region.

        Raises
        ------
        IndexError
            If the buffer is 0-dimensional or the index is out of range.
        ValueError
            If a slice with a step other than 1 is used.

        """
        if not self._shape:
            err_msg = "Cannot index a 0-dimensional buffer."
            raise IndexError(err_msg)

        inner_shape = self._shape[1:]
        inner_size = self._dtype.itemsize
        for dim in inner_shape:
            inner_size *= dim

        if isinstance(key, slice):
            if key.step not in (None, 1):
                err_msg = "Buffer views only support slices with step 1."
                raise ValueError(err_msg)
            start, stop, _ = key.indices(self._shape[0])
            length = max(0, stop - start)
            new_shape: tuple[int, ...] = (length, *inner_shape)
            offset = start * inner_size
        else:
            index = key + self._shape[0] if key < 0 else key
            if not 0 <= index < self._shape[0]:
                err_msg = f"Index {key} out of range for leading axis of size {self._shape[0]}."
                raise IndexError(err_msg)
            new_shape = inner_shape
            offset = index * inner_size

        host_array: np.ndarray | None = None
        if self._host_array is not None:
            flat = self._host_array.reshape(-1).view(np.uint8)
            nbytes = self._dtype.itemsize
            for dim in new_shape:
                nbytes *= dim
            host_array = flat[offset : offset + nbytes].view(self._dtype).reshape(new_shape)

        view = Buffer(
            self._location,
            self._dtype,
            new_shape,
            self._ptr + offset,
            host_array,
            pinned=self._pinned,
            mapped=self._mapped,
            owns_memory=False,
        )
        # keep the owning buffer alive for the lifetime of the view
        view._parent = self
        return view

    def reshape(self: Self, shape: tuple[int, ...]) -> Buffer:
        """
        Return a non-owning view of this buffer with a new shape.

        The total number of elements must be unchanged. No data is moved.

        Parameters
        ----------
        shape : tuple[int, ...]
            The new shape for the view.

        Returns
        -------
        Buffer
            A non-owning view over the same memory.

        Raises
        ------
        ValueError
            If the new shape has a different number of elements.

        """
        new_size = 1
        for s in shape:
            new_size *= int(s)
        if new_size != self.size:
            err_msg = f"Cannot reshape buffer of {self.size} elements into shape {shape}."
            raise ValueError(err_msg)
        host_array = self._host_array.reshape(shape) if self._host_array is not None else None
        view = Buffer(
            self._location,
            self._dtype,
            tuple(shape),
            self._ptr,
            host_array,
            pinned=self._pinned,
            mapped=self._mapped,
            owns_memory=False,
        )
        # keep the owning buffer alive for the lifetime of the view
        view._parent = self
        return view

    # ------------------------------------------------------------------
    # lifetime
    # ------------------------------------------------------------------

    def free(self: Self) -> None:
        """Free the memory of the buffer. No-op for non-owning views."""
        if self._freed or not self._owns_memory:
            return
        self._freed = True
        if self._location == MemoryLocation.DEVICE:
            cuda_call(cudart.cudaFree(self._ptr))
        elif self._pinned:
            cuda_call(cudart.cudaFreeHost(self._ptr))
        # pageable host memory is freed by the garbage collector

    def __enter__(self: Self) -> Self:
        """
        Enter the context manager.

        Returns
        -------
        Buffer
            The buffer itself.

        """
        return self

    def __exit__(self: Self, *args: object) -> None:
        """
        Exit the context manager, freeing the buffer.

        Parameters
        ----------
        *args : object
            The exception information, unused.

        """
        self.free()

    def __del__(self: Self) -> None:
        # potentially already had free called on it previously
        with contextlib.suppress(RuntimeError, AttributeError):
            self.free()

    def __repr__(self: Self) -> str:
        """
        Get the string representation of the buffer.

        Returns
        -------
        str
            The string representation.

        """
        return (
            f"Buffer(location={self._location.value}, shape={self._shape}, "
            f"dtype={self._dtype}, ptr={self._ptr}, pinned={self._pinned}, "
            f"mapped={self._mapped}, owns_memory={self._owns_memory})"
        )
