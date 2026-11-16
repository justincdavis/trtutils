# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
"""Report where the layers of a built TensorRT engine run: on a DLA or on the GPU."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trtutils.compat._libs import trt
from trtutils.core._engine import create_engine
from trtutils.core._stream import destroy_stream

from ._types import LayerInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

_REFORMAT_TYPES = frozenset({"Reformat", "NoOp"})
# layer types tensorrt usually folds or fuses away
_TRIVIAL_TYPES = frozenset({"CONSTANT", "IDENTITY", "SHUFFLE"})
_FOREIGN_NODE = re.compile(r"^\{ForeignNode\[(.*)\]\}$", re.DOTALL)
_MAX_NAME = 100


@dataclass(frozen=True)
class EngineLayer:
    """
    A layer of a built engine, as the TensorRT engine inspector reports it.

    Attributes
    ----------
    name : str
        The layer name. A DLA subgraph is named ``{ForeignNode[first...last]}`` after
        the first and last network layers it runs.
    layer_type : str
        The inspector ``LayerType``, e.g. ``"DLA"``, ``"Reformat"`` or ``"PointWiseV2"``.
    inputs : tuple[str, ...]
        The names of the tensors the layer reads.
    outputs : tuple[str, ...]
        The names of the tensors the layer writes.

    """

    name: str
    layer_type: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    @property
    def kind(self: Self) -> str:
        """
        Where the layer runs.

        One of ``"dla"``, ``"gpu_compute"``, ``"gpu_reformat"`` (a reformat or copy) or
        ``"gpu_constant"``. Only ``LayerType == "DLA"`` marks a DLA subgraph: GPU (Myelin)
        subgraphs are named ``{ForeignNode[...]}`` too.
        """
        if self.layer_type == "DLA":
            return "dla"
        if self.layer_type in _REFORMAT_TYPES:
            return "gpu_reformat"
        if self.layer_type == "Constant":
            return "gpu_constant"
        return "gpu_compute"


def _foreign_node_range(name: str) -> tuple[str, str] | None:
    match = _FOREIGN_NODE.match(name)
    if match is None:
        return None
    inner = match.group(1)
    first, last = inner.split("...", 1) if "..." in inner else (inner, inner)
    return first.split(" + ")[0].strip(), last.split(" + ")[-1].strip()


def _map_network_devices(
    engine_layers: Sequence[EngineLayer],
    network_layers: Sequence[LayerInfo],
    gpu_copies: Sequence[EngineLayer],
) -> dict[str, str]:
    devices: dict[str, str] = {}
    placed = [n for n in network_layers if n.layer_type not in _TRIVIAL_TYPES]
    gpu_names = [layer.name for layer in engine_layers if layer.kind == "gpu_compute"]
    copied = {tensor for copy in gpu_copies for tensor in copy.outputs}
    for n in placed:
        pattern = re.compile(r"(?<![\w/])" + re.escape(n.name) + r"(?!\w)")
        if any(pattern.search(name) for name in gpu_names) or copied.intersection(n.outputs):
            devices[n.name] = "gpu"
    index = {n.name: n.index for n in network_layers}
    for layer in sorted(engine_layers, key=lambda layer: layer.kind != "dla"):
        span = _foreign_node_range(layer.name)
        if span is None or span[0] not in index or span[1] not in index:
            continue
        lo, hi = sorted((index[span[0]], index[span[1]]))
        device = "dla" if layer.kind == "dla" else "gpu"
        for n in placed:
            if lo <= n.index <= hi and n.name not in devices:
                devices[n.name] = device
    return devices


@dataclass(frozen=True)
class PlacementReport:
    """
    Where the layers of a built engine run, from the TensorRT engine inspector.

    Attributes
    ----------
    engine_layers : list[EngineLayer]
        The layers of the built engine, in the order the inspector lists them.
    network_layers : list[LayerInfo]
        The layers of the network the engine was built from, empty if not given.
    io_names : tuple[str, ...]
        The names of the engine inputs and outputs.
    dla_core : int | None
        The DLA core the engine was loaded on, if given.
    inspected : bool
        False when the engine inspector returned nothing, so the placement is unknown.

    """

    engine_layers: list[EngineLayer]
    network_layers: list[LayerInfo] = field(default_factory=list)
    io_names: tuple[str, ...] = ()
    dla_core: int | None = None
    inspected: bool = True

    @classmethod
    def from_inspector_json(
        cls,
        info: dict[str, Any],
        network_layers: Sequence[LayerInfo] | None = None,
        dla_core: int | None = None,
    ) -> PlacementReport:
        """
        Build a report from engine inspector JSON.

        Parameters
        ----------
        info : dict[str, Any]
            The parsed output of ``IEngineInspector.get_engine_information`` in
            ``trt.LayerInformationFormat.JSON``, or of ``trtexec --exportLayerInfo``.
        network_layers : Sequence[LayerInfo], optional
            The layers of the network the engine was built from, as returned by
            :func:`inspect_onnx_layers`, to attribute each one to a device.
        dla_core : int, optional
            The DLA core the engine was loaded on.

        Returns
        -------
        PlacementReport
            The report.

        Raises
        ------
        ValueError
            If the layers are names only: the engine was not built with
            ``ProfilingVerbosity.DETAILED``.

        """
        layers = info.get("Layers", [])
        if any(isinstance(layer, str) for layer in layers):
            err_msg = (
                "Engine layer information has names only, build the engine with "
                "profiling_verbosity=trt.ProfilingVerbosity.DETAILED"
            )
            raise ValueError(err_msg)
        engine_layers = [
            EngineLayer(
                name=layer["Name"],
                layer_type=layer["LayerType"],
                inputs=tuple(tensor["Name"] for tensor in layer.get("Inputs", [])),
                outputs=tuple(tensor["Name"] for tensor in layer.get("Outputs", [])),
            )
            for layer in layers
        ]
        return cls(
            engine_layers=engine_layers,
            network_layers=list(network_layers) if network_layers is not None else [],
            io_names=tuple(info.get("Bindings", [])),
            dla_core=dla_core,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlacementReport:
        """
        Rebuild a report from :meth:`to_dict` output.

        Parameters
        ----------
        data : dict[str, Any]
            The output of :meth:`to_dict`, e.g. after a JSON round trip.

        Returns
        -------
        PlacementReport
            The report.

        """
        return cls(
            engine_layers=[
                EngineLayer(
                    name=layer["name"],
                    layer_type=layer["layer_type"],
                    inputs=tuple(layer["inputs"]),
                    outputs=tuple(layer["outputs"]),
                )
                for layer in data["engine_layers"]
            ],
            network_layers=[
                LayerInfo(
                    index=layer["index"],
                    name=layer["name"],
                    layer_type=layer["layer_type"],
                    precision=None
                    if layer["precision"] is None
                    else getattr(trt.DataType, layer["precision"]),
                    input_tensor_size=layer["input_tensor_size"],
                    output_tensor_size=layer["output_tensor_size"],
                    dla_compatible=layer["dla_compatible"],
                    outputs=tuple(layer["outputs"]),
                )
                for layer in data["network_layers"]
            ],
            io_names=tuple(data["io_names"]),
            dla_core=data["dla_core"],
            inspected=data["inspected"],
        )

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Get the report as a JSON-serializable dict.

        Returns
        -------
        dict[str, Any]
            The report fields, which :meth:`from_dict` reads back.

        """
        return {
            "engine_layers": [
                {**vars(layer), "inputs": list(layer.inputs), "outputs": list(layer.outputs)}
                for layer in self.engine_layers
            ],
            "network_layers": [
                {
                    **vars(layer),
                    "precision": None if layer.precision is None else layer.precision.name,
                    "outputs": list(layer.outputs),
                }
                for layer in self.network_layers
            ],
            "io_names": list(self.io_names),
            "dla_core": self.dla_core,
            "inspected": self.inspected,
        }

    def _of_kind(self: Self, kind: str) -> list[EngineLayer]:
        return [layer for layer in self.engine_layers if layer.kind == kind]

    @property
    def dla_subgraphs(self: Self) -> list[EngineLayer]:
        """The DLA subgraphs of the engine."""
        return self._of_kind("dla")

    @property
    def gpu_compute_layers(self: Self) -> list[EngineLayer]:
        """The layers that compute on the GPU."""
        return self._of_kind("gpu_compute")

    @property
    def reformat_layers(self: Self) -> list[EngineLayer]:
        """The reformats and copies that run on the GPU."""
        return self._of_kind("gpu_reformat")

    @property
    def io_reformat_layers(self: Self) -> list[EngineLayer]:
        """The GPU reformats that read an engine input or write an engine output."""
        io_names = set(self.io_names)
        return [r for r in self.reformat_layers if io_names.intersection(r.inputs + r.outputs)]

    @property
    def internal_reformat_layers(self: Self) -> list[EngineLayer]:
        """
        The GPU reformats and copies that are not on a chain to an engine input or output.

        In a DLA engine they sit between DLA subgraphs: a precision or format change, or
        a network layer TensorRT runs as plain copies on the GPU (e.g. a Concat on W).
        """
        reformats = self.reformat_layers
        io_side = {r.name for r in self.io_reformat_layers}
        tensors = {t for r in reformats if r.name in io_side for t in r.inputs + r.outputs}
        changed = True
        while changed:
            changed = False
            for r in reformats:
                if r.name not in io_side and tensors.intersection(r.inputs + r.outputs):
                    io_side.add(r.name)
                    tensors.update(r.inputs + r.outputs)
                    changed = True
        return [r for r in reformats if r.name not in io_side]

    @property
    def fully_on_dla(self: Self) -> bool:
        """True when only I/O reformats run on the GPU: no compute, no copies between DLA subgraphs."""
        return (
            bool(self.dla_subgraphs)
            and not self.gpu_compute_layers
            and not self.internal_reformat_layers
        )

    @property
    def zero_gpu(self: Self) -> bool:
        """True when the engine is nothing but DLA subgraphs."""
        return bool(self.engine_layers) and all(layer.kind == "dla" for layer in self.engine_layers)

    @property
    def transitions(self: Self) -> int:
        """The number of hops between DLA subgraphs and GPU work (compute or internal copies)."""
        internal = {r.name for r in self.internal_reformat_layers}
        runs: list[str] = []
        for layer in self.engine_layers:
            kind = "gpu_compute" if layer.name in internal else layer.kind
            if kind in ("dla", "gpu_compute") and (not runs or runs[-1] != kind):
                runs.append(kind)
        return max(len(runs) - 1, 0)

    @cached_property
    def network_devices(self: Self) -> dict[str, str]:
        """
        The device, ``"dla"`` or ``"gpu"``, of each network layer.

        A network layer runs on the GPU when a GPU compute layer names it or an internal
        GPU copy writes its output. Otherwise it runs where the subgraph whose
        ``{ForeignNode[first...last]}`` range holds it runs, DLA subgraphs first. Layers
        TensorRT folded or fused away are missing.
        """
        return _map_network_devices(
            self.engine_layers, self.network_layers, self.internal_reformat_layers
        )

    @property
    def dla_coverage(self: Self) -> float | None:
        """The fraction of placed network layers that run on the DLA, None if none are placed."""
        devices = list(self.network_devices.values())
        if not devices:
            return None
        return devices.count("dla") / len(devices)

    @property
    def blocking_layers(self: Self) -> list[LayerInfo]:
        """
        The network layers that keep the engine off the DLA.

        These are the layers placed on the GPU. When GPU compute layers exist but no
        network layer is placed on the GPU, they are the layers that failed the DLA check
        and were not placed on the DLA.
        """
        layers = [n for n in self.network_layers if n.layer_type not in _TRIVIAL_TYPES]
        on_gpu = [n for n in layers if self.network_devices.get(n.name) == "gpu"]
        if on_gpu or not self.gpu_compute_layers:
            return on_gpu
        return [
            n for n in layers if not n.dla_compatible and self.network_devices.get(n.name) != "dla"
        ]

    def summary(self: Self) -> str:
        """
        Get a one-line summary of the placement.

        Returns
        -------
        str
            The counts of DLA subgraphs, GPU layers and reformats, the DLA<->GPU
            transitions and the DLA coverage.

        """
        head = f"DLA core {self.dla_core}: " if self.dla_core is not None else ""
        if not self.inspected:
            return f"{head}placement unknown, the TensorRT engine inspector returned nothing"
        internal = len(self.internal_reformat_layers)
        between = f", {internal} between DLA subgraphs" if internal else ""
        coverage = self.dla_coverage
        return (
            f"{head}{len(self.dla_subgraphs)} DLA subgraph(s), "
            f"{len(self.gpu_compute_layers)} GPU compute layer(s), "
            f"{len(self.reformat_layers)} GPU reformat(s) "
            f"({len(self.io_reformat_layers)} at I/O{between}), "
            f"{self.transitions} DLA<->GPU transition(s), "
            f"coverage {'n/a' if coverage is None else f'{coverage:.0%}'}"
        )

    def __str__(self: Self) -> str:
        lines = [self.summary(), f"  {'kind':<13} {'type':<12} name"]
        for layer in self.engine_layers:
            name = layer.name
            if len(name) > _MAX_NAME:
                name = name[: _MAX_NAME - 3] + "..."
            lines.append(f"  {layer.kind:<13} {layer.layer_type:<12} {name}")
        blocking = self.blocking_layers
        if not self.fully_on_dla and blocking:
            names = ", ".join(f"{n.name} ({n.layer_type})" for n in blocking)
            lines.append(f"  on the GPU: {names}")
        return "\n".join(lines)


def inspect_placement(
    engine: trt.ICudaEngine | Path | str,
    dla_core: int | None = None,
    network_layers: Sequence[LayerInfo] | None = None,
) -> PlacementReport:
    """
    Find where each layer of a built engine runs: on a DLA or on the GPU.

    Works for DLA and GPU engines. The engine must be built with
    ``profiling_verbosity=trt.ProfilingVerbosity.DETAILED``, since with less detail a
    DLA subgraph cannot be told apart from a GPU one.

    Parameters
    ----------
    engine : trt.ICudaEngine | Path | str
        A loaded engine, or the path to a serialized one.
    dla_core : int, optional
        The DLA core to load the engine on, recorded in the report.
    network_layers : Sequence[LayerInfo], optional
        The layers of the network the engine was built from, as returned by
        :func:`inspect_onnx_layers`, to attribute each one to a device.

    Returns
    -------
    PlacementReport
        The report. ``inspected`` is False when the engine inspector returned nothing,
        which TensorRT 8.6 does for some engines that build and run fine.

    Raises
    ------
    ValueError
        If the engine was not built with ``ProfilingVerbosity.DETAILED``.

    """
    stream = None
    if isinstance(engine, (Path, str)):
        engine, _context, _logger, stream = create_engine(engine, dla_core=dla_core)
    detailed = engine.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    info = ""
    if detailed:
        inspector = engine.create_engine_inspector()
        info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    if stream is not None:
        destroy_stream(stream)
    if not detailed:
        err_msg = (
            "Engine was not built with ProfilingVerbosity.DETAILED, rebuild it with "
            "profiling_verbosity=trt.ProfilingVerbosity.DETAILED"
        )
        raise ValueError(err_msg)
    layers = list(network_layers) if network_layers is not None else []
    if not info:
        return PlacementReport([], layers, dla_core=dla_core, inspected=False)
    return PlacementReport.from_inspector_json(json.loads(info), layers, dla_core)
