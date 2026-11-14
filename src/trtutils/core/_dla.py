# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
"""Find the engine inputs a DLA reads in place, and the memory it may read them from."""

from __future__ import annotations

import bisect
import json
from typing import TYPE_CHECKING, cast

from trtutils._flags import FLAGS
from trtutils.compat._libs import cuda, trt

from ._bindings import Binding
from ._buffer import DEVICE_ALIGNMENT

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import cudart

    from ._buffer import Buffer


def dla_input_names(
    engine: trt.ICudaEngine, input_names: list[str], dla_core: int | None
) -> list[str]:
    """
    Get the engine inputs that a DLA layer reads directly, with no GPU layer in between.

    Only an engine loaded with a ``dla_core`` is inspected, so GPU engines cost
    nothing. If it was built with ``ProfilingVerbosity.DETAILED``, the layer
    types name its DLA layers and their inputs are returned. Without that
    detail a DLA subgraph cannot be told apart from a GPU (Myelin) one, and
    every input is returned.

    Parameters
    ----------
    engine : trt.ICudaEngine
        The engine to inspect.
    input_names : list[str]
        The names of the engine inputs.
    dla_core : int, optional
        The DLA core the engine was loaded with.

    Returns
    -------
    list[str]
        The input names a DLA reads in place, in the order of ``input_names``.

    """
    if not FLAGS.HAS_DLA or dla_core is None:
        return []
    if engine.profiling_verbosity != trt.ProfilingVerbosity.DETAILED:
        return list(input_names)
    info = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    # an engine with no DLA layer skips the parse
    if '"DLA"' not in info:
        return []
    read = {
        tensor.get("Name")
        for layer in json.loads(info).get("Layers", [])
        if layer.get("LayerType") == "DLA"
        for tensor in layer.get("Inputs", [])
    }
    return [name for name in input_names if name in read]


class DLAMemoryGuard:
    """
    Decide whether a DLA may read an engine input in place, at a given device address.

    TensorRT registers the address of every DLA input with cuDLA on each
    enqueue and never unregisters it. cuDLA keeps that registration, keyed by
    address, for the life of the execution context, and treats a later
    registration of the same address as already done. Once the memory is
    freed and the address is handed out again with new pages, the DLA keeps
    reading the old pages, and the outputs are silently wrong.

    The guard records every allocation it lets a DLA input read in place and
    refuses any later allocation that overlaps one of them, so those inputs
    are copied into the engine's own binding instead. A recorded allocation
    is read in place for as long as it lives: a reused input buffer, or the
    blocks a PyTorch caching allocator hands out from a kept segment.
    """

    def __init__(self: Self) -> None:
        """Create a guard with no recorded allocations."""
        # the (start, end) of the recorded allocations, sorted and disjoint
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._buffer_ids: set[int] = set()

    def can_read(self: Self, ptr: int) -> bool:
        """
        Check whether a DLA may read the memory at ``ptr`` in place, recording it if so.

        Parameters
        ----------
        ptr : int
            The device address of the input.

        Returns
        -------
        bool
            True if the allocation at ``ptr`` was recorded before, or overlaps no
            recorded allocation. False if it is a newer allocation in place of a
            recorded one, or ``ptr`` is not memory CUDA knows.

        """
        # CUDA never reuses a buffer id within a process, even when it reuses the address
        err, buffer_id = cuda.cuPointerGetAttribute(
            cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_BUFFER_ID, ptr
        )
        if err != cuda.CUresult.CUDA_SUCCESS:
            return False
        if buffer_id in self._buffer_ids:
            return True
        attrs = [
            cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
            cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_SIZE,
        ]
        err, values = cuda.cuPointerGetAttributes(len(attrs), attrs, ptr)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return False
        start = int(values[0])
        end = start + int(values[1])
        i = bisect.bisect_right(self._starts, start)
        if (i > 0 and self._ends[i - 1] > start) or (
            i < len(self._starts) and self._starts[i] < end
        ):
            return False
        self._starts.insert(i, start)
        self._ends.insert(i, end)
        self._buffer_ids.add(buffer_id)
        return True


class DLAInputBinding(Binding):
    """
    An input :class:`Binding` that a DLA reads in place.

    :meth:`bind` reads device memory in place only when the binding's
    :class:`DLAMemoryGuard` allows it, and copies it into the binding
    otherwise. Inputs no DLA reads keep the plain :meth:`Binding.bind`.
    """

    _guard: DLAMemoryGuard

    @classmethod
    def convert(cls, binding: Binding, guard: DLAMemoryGuard) -> DLAInputBinding:
        """
        Turn an allocated input Binding into one whose device memory a guard checks.

        Parameters
        ----------
        binding : Binding
            The input binding, converted in place.
        guard : DLAMemoryGuard
            The guard shared by the engine's DLA inputs.

        Returns
        -------
        DLAInputBinding
            The same binding object.

        """
        binding.__class__ = cls
        dla_binding = cast("DLAInputBinding", binding)
        dla_binding._guard = guard  # noqa: SLF001
        return dla_binding

    @property
    def guard(self: Self) -> DLAMemoryGuard:
        """The guard that checks device memory before the DLA reads it in place."""
        return self._guard

    def bind(self: Self, src: Buffer, stream: cudart.cudaStream_t | None = None) -> Buffer:
        """
        Get device memory the DLA can read ``src`` from, staging into the binding when needed.

        Like :meth:`Binding.bind`, but device memory at an address that was freed
        and handed out again since TensorRT registered it with the DLA is copied
        into the binding too.

        Parameters
        ----------
        src : Buffer
            The data, on the host or the device.
        stream : cudart.cudaStream_t, optional
            The stream to enqueue a staging copy on.

        Returns
        -------
        Buffer
            ``src`` itself, or the device view of the staged data.

        """
        if (
            src.device_visible
            and src.device_ptr % DEVICE_ALIGNMENT == 0
            and self._guard.can_read(src.device_ptr)
        ):
            return src
        return self.stage(src, stream)
