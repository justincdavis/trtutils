# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
"""Find the engine inputs a DLA reads in place, and the memory it may read them from."""

from __future__ import annotations

import bisect
import json
from typing import TYPE_CHECKING

from trtutils._flags import FLAGS
from trtutils.compat._libs import cuda, trt

if TYPE_CHECKING:
    from typing_extensions import Self


def dla_input_names(engine: trt.ICudaEngine, input_names: list[str]) -> list[str]:
    """
    Get the engine inputs that a DLA layer reads directly, with no GPU layer in between.

    Uses the layer information of the engine inspector. When the engine has
    no per-layer detail (built below ``ProfilingVerbosity.DETAILED``), every
    input of an engine that may run on a DLA is returned.

    Parameters
    ----------
    engine : trt.ICudaEngine
        The engine to inspect.
    input_names : list[str]
        The names of the engine inputs.

    Returns
    -------
    list[str]
        The input names a DLA reads in place, in the order of ``input_names``.

    """
    if not FLAGS.HAS_DLA:
        return []
    info = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    # ProfilingVerbosity.NONE gives no information, LAYER_NAMES_ONLY a list of names
    layers = json.loads(info).get("Layers", []) if info else []
    if layers and all(isinstance(layer, dict) for layer in layers):
        read = {
            tensor.get("Name")
            for layer in layers
            if layer.get("LayerType") == "DLA"
            for tensor in layer.get("Inputs", [])
        }
        return [name for name in input_names if name in read]
    # DLA subgraphs are named ForeignNode, as are some GPU subgraphs
    if layers and not any("ForeignNode" in layer for layer in layers):
        return []
    return list(input_names)


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
