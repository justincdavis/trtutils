# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
"""
The :class:`Buffer` type: the single currency for data passed into trtutils.

A Buffer is one contiguous, typed allocation that lives either on the host
(pageable, pinned, or pinned + mapped into the device address space) or on
the device. Every engine input is a Buffer, which means the engine always
knows the shape, dtype, and location of what it is given and never has to
guess from a raw pointer or reverse-engineer a numpy array's strides.
"""

from __future__ import annotations

import contextlib
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import nvtx

from trtutils._flags import FLAGS

with contextlib.suppress(Exception):
    from trtutils.compat._libs import cudart

from ._cuda import cuda_call

if TYPE_CHECKING:
    from typing_extensions import Self

# TensorRT requires I/O tensor addresses aligned to 256 bytes, and
# cudaMalloc / cudaHostAlloc always return at least that alignment
DEVICE_ALIGNMENT = 256


class MemoryLocation(Enum):
    """The memory space a :class:`Buffer` resides in."""

    HOST = "host"
    DEVICE = "device"


class _DeviceAllocation:
    """Owns a cudaMalloc allocation; freed when the last reference is dropped."""

    def __init__(self: Self, nbytes: int) -> None:
        self.ptr: int = cuda_call(cudart.cudaMalloc(nbytes)) if nbytes > 0 else 0

    def __del__(self: Self) -> None:
        # the CUDA context may already be gone during interpreter shutdown
        with contextlib.suppress(Exception):
            if self.ptr:
                cuda_call(cudart.cudaFree(self.ptr))
        self.ptr = 0


class _PinnedAllocation:
    """
    Owns a cudaHostAlloc allocation and exposes it to numpy.

    Numpy arrays created from this object (and every view derived from them)
    hold it as their ``base``, so the pinned memory is only returned to CUDA
    once no Buffer *and* no numpy array references it anymore.
    """

    def __init__(self: Self, nbytes: int, *, mapped: bool) -> None:
        flags = cudart.cudaHostAllocMapped if mapped else cudart.cudaHostAllocDefault
        self.ptr: int = cuda_call(cudart.cudaHostAlloc(nbytes, flags))
        self.device_ptr: int | None = (
            cuda_call(cudart.cudaHostGetDevicePointer(self.ptr, 0)) if mapped else None
        )
        self.__array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (self.ptr, False),
            "version": 3,
        }

    def __del__(self: Self) -> None:
        with contextlib.suppress(Exception):
            if self.ptr:
                cuda_call(cudart.cudaFreeHost(self.ptr))
        self.ptr = 0


def _nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    nbytes = dtype.itemsize
    for dim in shape:
        nbytes *= dim
    return nbytes


def _normalize_shape(
    shape: tuple[int, ...] | list[int] | int,
    size: int | None = None,
) -> tuple[int, ...]:
    """Resolve a shape, inferring a single -1 dimension from ``size`` when given."""
    dims = (shape,) if isinstance(shape, int) else tuple(int(d) for d in shape)
    if dims.count(-1) > 1 or (-1 in dims and size is None):
        err_msg = f"Cannot infer the -1 dimension of shape {dims}."
        raise ValueError(err_msg)
    if -1 in dims and size is not None:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or size % known != 0:
            err_msg = f"Cannot infer shape {dims} for {size} elements."
            raise ValueError(err_msg)
        dims = tuple(size // known if d == -1 else d for d in dims)
    if any(d < 0 for d in dims):
        err_msg = f"Invalid shape {dims}."
        raise ValueError(err_msg)
    return dims


def _is_c_contiguous(shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int) -> bool:
    expected = itemsize
    for dim, stride in zip(reversed(shape), reversed(strides)):
        # a dimension of size 1 never advances, so its stride is irrelevant
        if dim != 1 and stride != expected:
            return False
        expected *= dim
    return True


class Buffer:
    """
    One contiguous, typed allocation on the host or on the device.

    Buffers are the only input type accepted by :class:`trtutils.TRTEngine`.
    They carry everything the engine needs to consume data without guessing:
    the memory location, the dtype, and the exact shape.

    Create Buffers with the classmethods:

    - :meth:`Buffer.wrap` - zero-copy view of a C-contiguous numpy array, of
      any object exposing ``__cuda_array_interface__`` (CuPy, PyTorch,
      Numba, ...), or of another Buffer. This is the per-call path for
      feeding existing data to an engine.
    - :meth:`Buffer.empty` - a new, uninitialized allocation.
    - :meth:`Buffer.from_array` - a new allocation holding a copy of an array
      (any strides).
    - :meth:`Buffer.from_ptr` - a view over memory owned by someone else.

    Memory is reference counted: views (:meth:`view`, :meth:`reshape`,
    indexing) and numpy arrays obtained from :attr:`array` keep the
    underlying allocation alive, so it is never freed while still reachable.
    """

    def __init__(
        self: Self,
        location: MemoryLocation,
        dtype: np.dtype,
        shape: tuple[int, ...],
        ptr: int,
        *,
        array: np.ndarray | None = None,
        owner: object | None = None,
        pinned: bool = False,
        mapped_ptr: int | None = None,
        readonly: bool = False,
    ) -> None:
        """
        Create a Buffer over existing memory. Prefer the classmethod constructors.

        Parameters
        ----------
        location : MemoryLocation
            The memory space the allocation resides in.
        dtype : np.dtype
            The datatype of the elements.
        shape : tuple[int, ...]
            The shape of the buffer. The memory is C-contiguous.
        ptr : int
            The address of the first element in its memory space.
        array : np.ndarray, optional
            For host buffers, the numpy array over the same memory.
            Required for host buffers.
        owner : object, optional
            An object kept alive for as long as this Buffer exists, typically
            the object that owns the memory.
        pinned : bool
            Whether a host allocation is pagelocked.
        mapped_ptr : int, optional
            For mapped pinned host memory, the device address of this buffer.
        readonly : bool
            Whether the memory must not be written through this Buffer.

        Raises
        ------
        ValueError
            If a host buffer is created without a backing numpy array.

        """
        if location == MemoryLocation.HOST and array is None:
            err_msg = "Host buffers require a backing numpy array."
            raise ValueError(err_msg)
        self._location = location
        self._dtype = np.dtype(dtype)
        self._shape = tuple(int(d) for d in shape)
        self._ptr = int(ptr)
        self._array = array
        self._owner = owner
        self._pinned = pinned
        self._mapped_ptr = mapped_ptr
        self._readonly = readonly
        self._released = False

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(
        cls: type[Self],
        shape: tuple[int, ...] | list[int] | int,
        dtype: np.dtype | type,
        location: MemoryLocation = MemoryLocation.HOST,
        *,
        pinned: bool | None = None,
        mapped: bool | None = None,
    ) -> Self:
        """
        Allocate a new, uninitialized Buffer.

        Parameters
        ----------
        shape : tuple[int, ...] | list[int] | int
            The shape of the buffer.
        dtype : np.dtype | type
            The datatype of the elements.
        location : MemoryLocation
            Where to allocate. Default is the host.
        pinned : bool, optional
            For host buffers, whether to allocate pagelocked memory.
            By default True.
        mapped : bool, optional
            For pinned host buffers, whether to map the allocation into the
            device address space (zero-copy / unified memory). By default False.

        Returns
        -------
        Buffer
            The new buffer.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("core::Buffer.empty")
        dtype = np.dtype(dtype)
        dims = _normalize_shape(shape)
        nbytes = _nbytes(dims, dtype)
        pinned = pinned if pinned is not None else True
        mapped = mapped if mapped is not None else False

        if location == MemoryLocation.DEVICE:
            device_alloc = _DeviceAllocation(nbytes)
            buffer = cls(location, dtype, dims, device_alloc.ptr, owner=device_alloc)
        elif pinned and nbytes > 0:
            pinned_alloc = _PinnedAllocation(nbytes, mapped=mapped)
            array = np.asarray(pinned_alloc).view(dtype).reshape(dims)
            buffer = cls(
                location,
                dtype,
                dims,
                pinned_alloc.ptr,
                array=array,
                pinned=True,
                mapped_ptr=pinned_alloc.device_ptr,
            )
        else:
            array = np.empty(dims, dtype=dtype)
            buffer = cls(location, dtype, dims, array.ctypes.data, array=array)

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
        Allocate a new Buffer holding a copy of a numpy array.

        Unlike :meth:`wrap`, the array may have any strides; it is gathered
        into the new C-contiguous allocation.

        Parameters
        ----------
        array : np.ndarray
            The data to copy.
        location : MemoryLocation
            Where to allocate. Default is the host.
        pinned : bool, optional
            For host buffers, whether to allocate pagelocked memory.
            By default True.
        mapped : bool, optional
            For pinned host buffers, whether to map the allocation into the
            device address space. By default False.

        Returns
        -------
        Buffer
            The new buffer.

        """
        buffer = cls.empty(array.shape, array.dtype, location, pinned=pinned, mapped=mapped)
        if location == MemoryLocation.DEVICE:
            buffer.copy_from(cls.wrap(np.ascontiguousarray(array)))
        else:
            np.copyto(buffer.array, array)
        return buffer

    @classmethod
    def wrap(cls: type[Self], obj: object) -> Buffer:
        """
        Create a zero-copy Buffer view of existing data.

        Accepts:

        - a :class:`Buffer`, returned unchanged
        - a C-contiguous ``np.ndarray``, viewed as a pageable host buffer
        - any object exposing ``__cuda_array_interface__`` (CuPy, PyTorch,
          Numba, nvImageCodec, ...), viewed as a device buffer. If the
          interface names a producer stream, that stream is synchronized
          (per the CUDA Array Interface v3 contract) so the data is complete.

        The wrapped object is kept alive for as long as the Buffer exists.

        Parameters
        ----------
        obj : object
            The data to view.

        Returns
        -------
        Buffer
            A view over the object's memory.

        Raises
        ------
        TypeError
            If the object is not a Buffer, numpy array, or CUDA array.
        ValueError
            If the memory is not C-contiguous, or is masked.

        """
        if isinstance(obj, Buffer):
            return obj
        if isinstance(obj, np.ndarray):
            if not obj.flags.c_contiguous:
                err_msg = (
                    f"Buffer.wrap requires a C-contiguous array, got strides {obj.strides} for "
                    f"shape {obj.shape}. Use Buffer.from_array (copies) or np.ascontiguousarray."
                )
                raise ValueError(err_msg)
            return cls(
                MemoryLocation.HOST,
                obj.dtype,
                obj.shape,
                obj.ctypes.data,
                array=obj,
                readonly=not obj.flags.writeable,
            )
        interface = getattr(obj, "__cuda_array_interface__", None)
        if interface is None:
            err_msg = (
                f"Cannot wrap {type(obj).__name__} as a Buffer: expected a Buffer, a numpy "
                "array, or an object exposing __cuda_array_interface__."
            )
            raise TypeError(err_msg)
        return cls._from_cuda_array_interface(obj, interface)

    @classmethod
    def _from_cuda_array_interface(cls: type[Self], obj: object, interface: dict) -> Self:
        shape = tuple(int(d) for d in interface["shape"])
        dtype = np.dtype(interface["typestr"])
        if interface.get("mask") is not None:
            err_msg = "Masked CUDA arrays are not supported."
            raise ValueError(err_msg)
        strides = interface.get("strides")
        if strides is not None and not _is_c_contiguous(shape, tuple(strides), dtype.itemsize):
            err_msg = (
                f"Buffer.wrap requires C-contiguous memory, got strides {strides} for shape {shape}."
            )
            raise ValueError(err_msg)
        # CAI v3: the consumer must order itself after the producer's stream
        stream = interface.get("stream")
        if stream is not None:
            cuda_call(cudart.cudaStreamSynchronize(cudart.cudaStream_t(int(stream))))
        ptr, readonly = interface["data"]
        return cls(
            MemoryLocation.DEVICE,
            dtype,
            shape,
            int(ptr),
            owner=obj,
            readonly=bool(readonly),
        )

    @classmethod
    def from_ptr(
        cls: type[Self],
        ptr: int,
        shape: tuple[int, ...] | list[int],
        dtype: np.dtype | type,
        location: MemoryLocation = MemoryLocation.DEVICE,
        *,
        owner: object | None = None,
    ) -> Self:
        """
        Create a Buffer view over memory owned by someone else.

        The Buffer never frees the memory. Pass the owning object as
        ``owner`` so it is kept alive for as long as the view exists.

        Parameters
        ----------
        ptr : int
            The address of the memory in its memory space.
        shape : tuple[int, ...] | list[int]
            The shape of the memory (C-contiguous).
        dtype : np.dtype | type
            The datatype of the elements.
        location : MemoryLocation
            Where the memory resides. Default is the device.
        owner : object, optional
            The object owning the memory, kept alive by the view.

        Returns
        -------
        Buffer
            A view over the memory.

        """
        dtype = np.dtype(dtype)
        dims = _normalize_shape(tuple(shape))
        array: np.ndarray | None = None
        if location == MemoryLocation.HOST:
            nbytes = _nbytes(dims, dtype)
            interface = {
                "shape": (nbytes,),
                "typestr": "|u1",
                "data": (int(ptr), False),
                "version": 3,
            }
            array = np.asarray(_ArrayInterface(interface, owner)).view(dtype).reshape(dims)
        return cls(location, dtype, dims, ptr, array=array, owner=owner)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def location(self: Self) -> MemoryLocation:
        """The memory space the buffer resides in."""
        return self._location

    @property
    def is_device(self: Self) -> bool:
        """Whether the buffer resides on the device."""
        return self._location == MemoryLocation.DEVICE

    @property
    def is_host(self: Self) -> bool:
        """Whether the buffer resides on the host."""
        return self._location == MemoryLocation.HOST

    @property
    def dtype(self: Self) -> np.dtype:
        """The datatype of the elements."""
        return self._dtype

    @property
    def shape(self: Self) -> tuple[int, ...]:
        """The shape of the buffer."""
        return self._shape

    @property
    def ndim(self: Self) -> int:
        """The number of dimensions."""
        return len(self._shape)

    @property
    def size(self: Self) -> int:
        """The number of elements."""
        size = 1
        for dim in self._shape:
            size *= dim
        return size

    @property
    def nbytes(self: Self) -> int:
        """The size of the buffer in bytes."""
        return self.size * self._dtype.itemsize

    @property
    def ptr(self: Self) -> int:
        """
        The address of the first element in the buffer's own memory space.

        Raises
        ------
        RuntimeError
            If the buffer was freed.

        """
        self._check_alive()
        return self._ptr

    @property
    def device_ptr(self: Self) -> int:
        """
        The device-visible address of the buffer.

        For device buffers this is :attr:`ptr`; for mapped pinned host
        buffers it is the device alias of the host memory.

        Raises
        ------
        ValueError
            If the buffer is host memory that is not mapped into the device.

        """
        self._check_alive()
        if self._location == MemoryLocation.DEVICE:
            return self._ptr
        if self._mapped_ptr is not None:
            return self._mapped_ptr
        err_msg = "Host buffer is not mapped into the device address space."
        raise ValueError(err_msg)

    @property
    def device_visible(self: Self) -> bool:
        """Whether the device can read the buffer directly (device or mapped host memory)."""
        return self._location == MemoryLocation.DEVICE or self._mapped_ptr is not None

    @property
    def array(self: Self) -> np.ndarray:
        """
        The numpy array over a host buffer's memory (zero-copy).

        The array keeps the memory alive on its own, so it stays valid even
        after the Buffer is freed or garbage collected.

        Raises
        ------
        TypeError
            If the buffer resides on the device. Use :meth:`numpy` to copy it back.

        """
        self._check_alive()
        if self._array is None:
            err_msg = "Device buffers have no host array; use Buffer.numpy() to copy to the host."
            raise TypeError(err_msg)
        return self._array

    @property
    def pinned(self: Self) -> bool:
        """Whether a host buffer is pagelocked."""
        return self._pinned

    @property
    def mapped(self: Self) -> bool:
        """Whether a host buffer is mapped into the device address space."""
        return self._mapped_ptr is not None

    @property
    def readonly(self: Self) -> bool:
        """Whether the buffer must not be written."""
        return self._readonly

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------

    def view(self: Self, shape: tuple[int, ...] | list[int] | int) -> Buffer:
        """
        Get a view of the leading elements of the buffer with a new shape.

        The view covers the first ``prod(shape)`` elements, which must not
        exceed the buffer's size. This is how a max-shape allocation serves
        a smaller tensor: the smaller tensor is a C-contiguous prefix.

        Parameters
        ----------
        shape : tuple[int, ...] | list[int] | int
            The shape of the view. One dimension may be -1.

        Returns
        -------
        Buffer
            A view sharing this buffer's memory.

        Raises
        ------
        ValueError
            If the view would be larger than the buffer.

        """
        self._check_alive()
        dims = _normalize_shape(shape, self.size)
        count = 1
        for dim in dims:
            count *= dim
        if count > self.size:
            err_msg = (
                f"View of shape {dims} ({count} elements) exceeds buffer of {self.size} elements."
            )
            raise ValueError(err_msg)
        return self._subview(0, dims)

    def reshape(self: Self, shape: tuple[int, ...] | list[int] | int) -> Buffer:
        """
        Get a view of the whole buffer with a new shape.

        Parameters
        ----------
        shape : tuple[int, ...] | list[int] | int
            The new shape, with the same number of elements. One dimension may be -1.

        Returns
        -------
        Buffer
            A view sharing this buffer's memory.

        Raises
        ------
        ValueError
            If the number of elements differs.

        """
        dims = _normalize_shape(shape, self.size)
        count = 1
        for dim in dims:
            count *= dim
        if count != self.size:
            err_msg = f"Cannot reshape buffer of {self.size} elements into shape {dims}."
            raise ValueError(err_msg)
        return self.view(dims)

    def __len__(self: Self) -> int:
        """
        Get the size of the leading dimension.

        Returns
        -------
        int
            The size of the leading dimension.

        Raises
        ------
        TypeError
            If the buffer is 0-dimensional.

        """
        if not self._shape:
            err_msg = "len() of a 0-dimensional Buffer."
            raise TypeError(err_msg)
        return self._shape[0]

    def __getitem__(self: Self, key: int | slice) -> Buffer:
        """
        Get a view of a range along the leading dimension.

        Parameters
        ----------
        key : int | slice
            An index or a step-1 slice along the leading dimension.

        Returns
        -------
        Buffer
            A view sharing this buffer's memory.

        Raises
        ------
        IndexError
            If the buffer is 0-dimensional or the index is out of range.
        ValueError
            If the slice has a step other than 1.

        """
        self._check_alive()
        if not self._shape:
            err_msg = "Cannot index a 0-dimensional Buffer."
            raise IndexError(err_msg)
        inner = self._shape[1:]
        inner_size = 1
        for dim in inner:
            inner_size *= dim
        if isinstance(key, slice):
            if key.step not in (None, 1):
                err_msg = "Buffer views only support slices with step 1."
                raise ValueError(err_msg)
            start, stop, _ = key.indices(self._shape[0])
            return self._subview(start * inner_size, (max(0, stop - start), *inner))
        index = key + self._shape[0] if key < 0 else key
        if not 0 <= index < self._shape[0]:
            err_msg = f"Index {key} out of range for leading dimension of size {self._shape[0]}."
            raise IndexError(err_msg)
        return self._subview(index * inner_size, inner)

    def _subview(self: Self, offset_elems: int, shape: tuple[int, ...]) -> Buffer:
        offset = offset_elems * self._dtype.itemsize
        array: np.ndarray | None = None
        if self._array is not None:
            count = 1
            for dim in shape:
                count *= dim
            array = self._array.reshape(-1)[offset_elems : offset_elems + count].reshape(shape)
        return Buffer(
            self._location,
            self._dtype,
            shape,
            self._ptr + offset,
            array=array,
            owner=self._owner,
            pinned=self._pinned,
            mapped_ptr=self._mapped_ptr + offset if self._mapped_ptr is not None else None,
            readonly=self._readonly,
        )

    # ------------------------------------------------------------------
    # transfers
    # ------------------------------------------------------------------

    def copy_from(self: Self, src: Buffer, stream: cudart.cudaStream_t | None = None) -> None:
        """
        Copy the contents of another Buffer into this one.

        The direction (host/device on either side) is taken from the two
        buffers' locations. Both must hold the same dtype and number of
        elements; shapes may differ (both are C-contiguous).

        Parameters
        ----------
        src : Buffer
            The source of the data.
        stream : cudart.cudaStream_t, optional
            If given, copies involving the device are enqueued on the stream
            and the caller must synchronize it before relying on the result.
            Host-to-host copies are always synchronous.

        Raises
        ------
        TypeError
            If ``src`` is not a Buffer.
        ValueError
            If the dtypes or element counts differ, or this buffer is read-only.

        """
        if not isinstance(src, Buffer):
            err_msg = f"copy_from expects a Buffer, got {type(src).__name__}. Use Buffer.wrap()."
            raise TypeError(err_msg)
        if self._readonly:
            err_msg = "Cannot copy into a read-only Buffer."
            raise ValueError(err_msg)
        if src.dtype != self._dtype or src.size != self.size:
            err_msg = (
                f"Buffer mismatch: cannot copy {src.shape} {src.dtype} into "
                f"{self._shape} {self._dtype}."
            )
            raise ValueError(err_msg)
        if self.is_host and src.is_host:
            np.copyto(self.array, src.array.reshape(self._shape))
            return
        if src.is_host:
            kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        elif self.is_host:
            kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        else:
            kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("core::Buffer.copy")
        if stream is None:
            cuda_call(cudart.cudaMemcpy(self.ptr, src.ptr, self.nbytes, kind))
        else:
            cuda_call(cudart.cudaMemcpyAsync(self.ptr, src.ptr, self.nbytes, kind, stream))
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    def copy_to(self: Self, dst: Buffer, stream: cudart.cudaStream_t | None = None) -> None:
        """
        Copy the contents of this Buffer into another one.

        Parameters
        ----------
        dst : Buffer
            The destination.
        stream : cudart.cudaStream_t, optional
            See :meth:`copy_from`.

        """
        dst.copy_from(self, stream)

    def numpy(self: Self) -> np.ndarray:
        """
        Get the contents as a numpy array.

        Host buffers return their backing array (zero-copy). Device buffers
        are copied synchronously into a new pageable array.

        Returns
        -------
        np.ndarray
            The contents of the buffer.

        """
        if self.is_host:
            return self.array
        out = Buffer.wrap(np.empty(self._shape, dtype=self._dtype))
        out.copy_from(self)
        return out.array

    def __array__(
        self: Self,
        dtype: np.dtype | None = None,
        *,
        copy: bool | None = None,
    ) -> np.ndarray:
        """
        Support ``np.asarray`` on host buffers.

        Parameters
        ----------
        dtype : np.dtype, optional
            The dtype to convert to.
        copy : bool, optional
            Whether to copy (numpy 2 protocol).

        Returns
        -------
        np.ndarray
            The host data.

        Raises
        ------
        TypeError
            If the buffer resides on the device.

        """
        if not self.is_host:
            err_msg = "Cannot implicitly convert a device Buffer to numpy; use Buffer.numpy()."
            raise TypeError(err_msg)
        arr = self.array
        if dtype is not None and np.dtype(dtype) != arr.dtype:
            return arr.astype(dtype)
        return arr.copy() if copy else arr

    @property
    def __cuda_array_interface__(self: Self) -> dict[str, Any]:
        """
        The CUDA Array Interface (v3), for device and mapped host buffers.

        Returns
        -------
        dict[str, Any]
            The interface description.

        Raises
        ------
        AttributeError
            If the device cannot address the buffer, so ``hasattr`` checks
            by other libraries behave correctly.

        """
        if not self.device_visible or self._released:
            err_msg = "Buffer is not device-addressable."
            raise AttributeError(err_msg)
        return {
            "shape": self._shape,
            "typestr": self._dtype.str,
            "data": (self.device_ptr, self._readonly),
            "strides": None,
            "version": 3,
        }

    # ------------------------------------------------------------------
    # lifetime
    # ------------------------------------------------------------------

    def _check_alive(self: Self) -> None:
        if self._released:
            err_msg = "Buffer has been freed."
            raise RuntimeError(err_msg)

    def free(self: Self) -> None:
        """
        Release this Buffer's reference to its memory.

        The memory is returned to CUDA as soon as nothing else references
        it: other views of the same allocation and numpy arrays obtained
        from :attr:`array` keep it alive, so freeing can never leave them
        dangling. Using this Buffer after ``free`` raises ``RuntimeError``.
        """
        self._released = True
        self._array = None
        self._owner = None

    def __enter__(self: Self) -> Self:
        """
        Enter a context that frees the buffer on exit.

        Returns
        -------
        Buffer
            This buffer.

        """
        return self

    def __exit__(self: Self, *args: object) -> None:
        """
        Free the buffer.

        Parameters
        ----------
        *args : object
            Exception information, unused.

        """
        self.free()

    def __repr__(self: Self) -> str:
        """
        Get a string representation of the buffer.

        Returns
        -------
        str
            The representation.

        """
        kind = self._location.value
        if self._pinned:
            kind += ", pinned"
        if self._mapped_ptr is not None:
            kind += ", mapped"
        state = ", freed" if self._released else ""
        return f"Buffer(shape={self._shape}, dtype={self._dtype}, {kind}{state})"


class _ArrayInterface:
    """Expose foreign host memory to numpy while keeping its owner alive."""

    def __init__(self: Self, interface: dict[str, Any], owner: object | None) -> None:
        self.__array_interface__ = interface
        self._owner = owner


def as_buffers(data: object) -> list[Buffer]:
    """
    Validate that every element of a sequence is a Buffer.

    Parameters
    ----------
    data : object
        The sequence of inputs given to an engine.

    Returns
    -------
    list[Buffer]
        The inputs as a list.

    Raises
    ------
    TypeError
        If ``data`` is not a sequence of Buffers.

    """
    if isinstance(data, (Buffer, np.ndarray)) or not isinstance(data, (list, tuple)):
        err_msg = f"Engine inputs must be a list of Buffers, got {type(data).__name__}."
        raise TypeError(err_msg)
    for item in data:
        if not isinstance(item, Buffer):
            err_msg = (
                f"Engine inputs must be Buffers, got {type(item).__name__}. "
                "Wrap host or device data with trtutils.core.Buffer.wrap(...)."
            )
            raise TypeError(err_msg)
    return list(data)
