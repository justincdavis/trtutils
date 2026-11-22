# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engines for HaX-CoNN: layer-group engines and per-schedule mixed GPU/DLA engines."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from trtutils._log import LOG
from trtutils.builder._build import build_engine
from trtutils.builder._utils import needs_dynamic_range
from trtutils.compat._libs import trt

from ._groups import constant_layers
from ._simulate import encode_assignment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from trtutils.builder._batcher import AbstractBatcher

    from ._types import ModelTraits, Schedule

_NOMINAL_RANGE = 4.0


def nominal_ranges_hook(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
    """
    Give every tensor without an INT8 dynamic range a fixed one, for timing builds.

    INT8 kernels run as fast whatever their scales, so engines built with this hook time
    like calibrated ones, but their outputs are meaningless. Use it only to measure.

    Parameters
    ----------
    network : trt.INetworkDefinition
        The network, from ``build_engine(hooks=[...])``.

    Returns
    -------
    trt.INetworkDefinition
        The same network.

    """
    tensors = [network.get_input(i) for i in range(network.num_inputs)]
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        tensors.extend(layer.get_output(k) for k in range(layer.num_outputs))
    for tensor in tensors:
        if tensor.dynamic_range is None and needs_dynamic_range(tensor):
            tensor.set_dynamic_range(-_NOMINAL_RANGE, _NOMINAL_RANGE)
    return network


def slice_hook(first: int, last: int) -> Callable[[trt.INetworkDefinition], trt.INetworkDefinition]:
    """
    Make a build hook that keeps only the network layers ``first`` to ``last``.

    This is the paper's ``markOutput`` / ``addInput`` method of profiling a layer group
    on its own. Every data tensor the kept layers read from outside the range becomes
    a network input, and every tensor they write that a layer outside the range reads,
    or that was a network output, becomes a network output. TensorRT drops the other
    layers when it builds; the original inputs stay as unused engine inputs.

    Parameters
    ----------
    first : int
        The index of the first network layer to keep.
    last : int
        The index of the last network layer to keep.

    Returns
    -------
    Callable[[trt.INetworkDefinition], trt.INetworkDefinition]
        The hook, for ``build_engine(hooks=[...])``. It must run before any hook that
        adds layers.

    Raises
    ------
    ValueError
        When the hook runs, if the range is outside the network, a new input would have
        a dynamic shape, or the range produces nothing used elsewhere.

    """

    def _hook(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        if not 0 <= first <= last < network.num_layers:
            err_msg = f"layers {first}-{last} are outside the network's {network.num_layers}"
            raise ValueError(err_msg)
        layers = [network.get_layer(i) for i in range(network.num_layers)]
        ins = [
            [t.name for t in (layer.get_input(k) for k in range(layer.num_inputs)) if t is not None]
            for layer in layers
        ]
        outs = [[layer.get_output(k).name for k in range(layer.num_outputs)] for layer in layers]
        constant = constant_layers([str(layer.type).split(".")[-1] for layer in layers], ins, outs)
        producer = {name: i for i, names in enumerate(outs) for name in names}

        new_inputs: dict[str, trt.ITensor] = {}
        for i in range(first, last + 1):
            layer = layers[i]
            for k in range(layer.num_inputs):
                tensor = layer.get_input(k)
                if tensor is None:
                    continue
                source = producer.get(tensor.name)
                if source is None or first <= source <= last or source in constant:
                    continue
                if tensor.name not in new_inputs:
                    if any(d < 0 for d in tensor.shape):
                        err_msg = f"slicing needs static shapes, {tensor.name} is {tensor.shape}"
                        raise ValueError(err_msg)
                    new_inputs[tensor.name] = network.add_input(
                        f"{tensor.name}__in", tensor.dtype, tensor.shape
                    )
                layer.set_input(k, new_inputs[tensor.name])

        originals = [network.get_output(i) for i in range(network.num_outputs)]
        original_names = {t.name for t in originals}
        for tensor in originals:
            network.unmark_output(tensor)
        used_outside = {
            name for i, names in enumerate(ins) if not first <= i <= last for name in names
        }
        marked = 0
        for i in range(first, last + 1):
            layer = layers[i]
            for k in range(layer.num_outputs):
                tensor = layer.get_output(k)
                if tensor.name in used_outside or tensor.name in original_names:
                    network.mark_output(tensor)
                    marked += 1
        if marked == 0:
            err_msg = f"layers {first}-{last} produce nothing used outside them"
            raise ValueError(err_msg)
        return network

    return _hook


def assignment_code(accelerators: Sequence[str]) -> str:
    """
    Name an assignment for engine file names.

    Parameters
    ----------
    accelerators : Sequence[str]
        The accelerator of each layer group.

    Returns
    -------
    str
        One letter per group, "G" for the GPU and "D" for a DLA.

    """
    return "".join("D" if acc == "dla" else "G" for acc in accelerators)


def build_engines(
    models: Mapping[str, Path | str],
    traits: Mapping[str, ModelTraits],
    schedule: Schedule,
    output_dir: Path | str,
    dla_core: int = 0,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    data_batchers: Mapping[str, AbstractBatcher] | None = None,
    *,
    verbose: bool | None = None,
) -> dict[str, Path]:
    """
    Build one mixed GPU/DLA engine per DNN for a schedule.

    Each engine runs every layer of a group on the group's accelerator, in the
    precision the DNN's traits were measured in, with GPU fallback for layers a DLA
    cannot run. INT8 engines are calibrated with the DNN's data batcher; without one
    they get :func:`nominal_ranges_hook` ranges, which time correctly but compute
    nothing useful. An engine that already exists in ``output_dir`` for the same
    assignment is reused, so going back to a schedule seen before costs nothing.

    Parameters
    ----------
    models : Mapping[str, Path | str]
        The ONNX file of each DNN.
    traits : Mapping[str, ModelTraits]
        The traits of each DNN, which hold its layer groups.
    schedule : Schedule
        The schedule to build.
    output_dir : Path | str
        The directory for the engines.
    dla_core : int, optional
        The DLA core the DLA layers run on. Default is 0.
    workspace : float, optional
        The TensorRT workspace in GiB. Default is 4.0.
    timing_cache : Path | str, optional
        A timing cache file shared by the builds. Default is None.
    data_batchers : Mapping[str, AbstractBatcher], optional
        The INT8 calibration data of each DNN, by name. Default is None.
    verbose : bool, optional
        Whether to log the builds. Default is None (False).

    Returns
    -------
    dict[str, Path]
        The engine of each DNN.

    Raises
    ------
    ValueError
        If ``models``, ``traits`` and the schedule do not name the same DNNs, or an
        assignment does not fit its DNN's layer groups.

    """
    if not set(models) == set(traits) == set(schedule.assignments):
        err_msg = (
            f"models {sorted(models)}, traits {sorted(traits)} and schedule "
            f"{sorted(schedule.assignments)} must name the same DNNs"
        )
        raise ValueError(err_msg)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    batchers = dict(data_batchers) if data_batchers is not None else {}
    engines: dict[str, Path] = {}
    for name, onnx in models.items():
        accs = schedule.assignments[name]
        encode_assignment(traits[name], accs)
        int8 = traits[name].precision == "int8"
        batcher = batchers.get(name) if int8 else None
        suffix = ("int8" if batcher is not None else "int8nominal") if int8 else "fp16"
        # the same letters mean another engine for another model or other group bounds
        source = Path(onnx).resolve()
        key = (
            str(source),
            source.stat().st_mtime_ns,
            [(g.first, g.last) for g in traits[name].groups],
        )
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:10]
        path = directory / f"{name}_{assignment_code(accs)}_{suffix}_{digest}.engine"
        engines[name] = path
        if path.exists():
            continue
        if int8 and batcher is None:
            LOG.warning(f"HaX-CoNN: {name} has no data batcher, its INT8 outputs are not usable")
        devices: list[tuple[int, trt.DeviceType | None]] = []
        for group, acc in zip(traits[name].groups, accs):
            device = trt.DeviceType.DLA if acc == "dla" else trt.DeviceType.GPU
            devices.extend((i, device) for i in range(group.first, group.last + 1))
        has_dla = "dla" in accs
        if verbose:
            LOG.info(f"HaX-CoNN: building {path.name}")
        # constant layers of DLA groups fall back to the GPU, which only warns
        with LOG.with_level("ERROR"):
            build_engine(
                onnx,
                path,
                default_device=trt.DeviceType.DLA if has_dla else trt.DeviceType.GPU,
                workspace=workspace,
                dla_core=dla_core if has_dla else None,
                data_batcher=batcher,
                layer_device=devices,
                hooks=[nominal_ranges_hook] if int8 and batcher is None else None,
                timing_cache=timing_cache,
                gpu_fallback=True,
                fp16=True,
                int8=int8,
                verbose=verbose,
            )
    return engines
