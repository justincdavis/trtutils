# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Offline characterization for HaX-CoNN: layer groups of a DNN and platform contention."""

from __future__ import annotations

import gc
import math
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
from jetsontools import TegraStats, parse_tegrastats

from trtutils._engine import TRTEngine
from trtutils._log import LOG
from trtutils.builder._build import build_engine
from trtutils.compat._libs import trt
from trtutils.core._stream import stream_synchronize
from trtutils.inspect._onnx import inspect_onnx_layers
from trtutils.inspect._placement import inspect_placement
from trtutils.profiling._profiler import profile_engine

from ._build import nominal_ranges_hook, slice_hook
from ._groups import LayerGroup, fused_spans, group_layers, match_engine_layers
from ._types import Calibration, GroupTraits, ModelTraits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trtutils.profiling._profiler import LayerTiming

_TEGRA_INTERVAL_MS = 20
_SETTLE_S = 0.3
_CORUN_TRIM_S = (0.2, 0.1)
_REPEATS = 4
_REFORMAT = "Reformatting CopyNode"
_FOREIGN_NODE = "{ForeignNode["
_EMC_KEY = "EMC_FREQ"


class _Measured(NamedTuple):
    # compute samples (ms), transition in/out (ms), memory samples, on a DLA only, the
    # profiled compute layers and the factor that rescales their times
    time_ms: list[float]
    transition_in_ms: float
    transition_out_ms: float
    memory: list[float]
    on_dla: bool
    layers: list[LayerTiming]
    scale: float


def _load(path: Path, dla_core: int | None) -> TRTEngine:
    return TRTEngine(path, dla_core=dla_core, cuda_graph=False, warmup=False)


def _busy_step(engine: TRTEngine) -> Callable[[], None]:
    # inputs stay in the bindings' own device memory; several enqueues per sync keep
    # the accelerator busy, so memory samples need no idle correction
    inputs = [binding.device for binding in engine.input_bindings]

    def _step() -> None:
        for _ in range(_REPEATS):
            engine.raw_exec(inputs)
        stream_synchronize(engine.stream)

    return _step


def _free_run(step: Callable[[], None], seconds: float) -> list[tuple[float, float]]:
    stamps: list[tuple[float, float]] = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        step()
        stamps.append((t0, time.perf_counter()))
    return stamps


def _per_execute_ms(stamps: Sequence[tuple[float, float]]) -> list[float]:
    return [(t1 - t0) * 1000.0 / _REPEATS for t0, t1 in stamps]


def _check_tegrastats() -> None:
    if os.geteuid() == 0:
        return
    # tegrastats reports memory-controller utilisation only as root
    result = subprocess.run(["sudo", "-n", "true"], check=False, capture_output=True)  # noqa: S607
    if result.returncode != 0:
        err_msg = (
            "measuring memory throughput needs tegrastats as root: allow passwordless sudo "
            "for tegrastats (sudo -n must work) or run as root"
        )
        raise RuntimeError(err_msg)


def measure_memory(step: Callable[[], None], seconds: float = 1.5) -> list[float]:
    """
    Sample memory-controller (EMC) utilisation while a step runs in a loop.

    Parameters
    ----------
    step : Callable[[], None]
        One unit of work, called repeatedly. It should keep its accelerator busy.
    seconds : float, optional
        How long to loop. The first 0.3 s are discarded. Default is 1.5.

    Returns
    -------
    list[float]
        The EMC utilisation samples in percent, one per tegrastats line (about every
        30 ms). Lock the memory clock (``sudo jetson_clocks``) so percentages are
        comparable across runs.

    Raises
    ------
    RuntimeError
        If tegrastats cannot run as root or reports no EMC utilisation.

    """
    _check_tegrastats()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tegrastats.txt"
        with TegraStats(path, interval=_TEGRA_INTERVAL_MS, sudo=os.geteuid() != 0):
            t0 = time.time()
            while time.time() - t0 < seconds:
                step()
            t1 = time.time()
        rows = parse_tegrastats(path)
    window = [row for row in rows if t0 + _SETTLE_S <= float(row["timestamp"]) <= t1]
    if not window or _EMC_KEY not in window[0]:
        err_msg = (
            "tegrastats reported no EMC_FREQ samples; it prints memory-controller "
            "utilisation only as root"
        )
        raise RuntimeError(err_msg)
    return [float(row[_EMC_KEY].split("%")[0]) for row in window]


def _build(
    onnx: Path | str | Callable,
    output: Path,
    *,
    dla_core: int | None,
    hooks: list | None,
    workspace: float,
    timing_cache: Path | str,
    int8: bool,
    verbose: bool | None,
) -> None:
    dla = dla_core is not None
    # timing only: int8 kernels run as fast with nominal scales as with calibrated ones
    hooks = [*(hooks or []), nominal_ranges_hook] if int8 else hooks
    with LOG.with_level("ERROR"):
        build_engine(
            onnx,
            output,
            default_device=trt.DeviceType.DLA if dla else trt.DeviceType.GPU,
            workspace=workspace,
            dla_core=dla_core,
            hooks=hooks,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            timing_cache=timing_cache,
            gpu_fallback=dla,
            fp16=True,
            int8=int8,
            verbose=verbose,
        )


def _measure_group(
    path: Path,
    dla_core: int | None,
    iterations: int,
    warmup_iterations: int,
    memory_seconds: float | None,
) -> _Measured:
    engine = _load(path, dla_core)
    report = inspect_placement(engine.engine, dla_core) if dla_core is not None else None
    step = _busy_step(engine)
    for _ in range(warmup_iterations):
        step()
    stamps: list[tuple[float, float]] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        step()
        stamps.append((t0, time.perf_counter()))
    wall = float(np.median(_per_execute_ms(stamps)))
    memory = measure_memory(step, memory_seconds) if memory_seconds is not None else []
    # the profiler stays attached, so it runs last on this engine
    profile = profile_engine(engine, iterations, warmup_iterations)
    del engine
    gc.collect()

    names = [layer.name for layer in profile.layers]
    head = next((i for i, n in enumerate(names) if not n.startswith(_REFORMAT)), len(names))
    tail = next(
        (len(names) - i for i, n in enumerate(reversed(names)) if not n.startswith(_REFORMAT)),
        head,
    )
    middle = profile.layers[head:tail]
    on_dla = False
    if report is not None:
        # tensorrt 8.6 sometimes inspects nothing: then the profile names decide
        on_dla = (
            report.fully_on_dla
            if report.inspected
            else (bool(middle) and all(layer.name.startswith(_FOREIGN_NODE) for layer in middle))
        )
    total = sum(layer.mean for layer in profile.layers)
    # the profiler slows GPU layers down; never scale times up to the wall time
    scale = min(1.0, wall / total) if total > 0.0 else 1.0
    # a group shorter than its launch leaves the accelerator idle between runs, and
    # the memory controller idles with it (idle EMC utilisation is about 0)
    duty = min(1.0, total / wall) if wall > 0.0 else 1.0
    if duty > 0.0:
        memory = [min(100.0, value / duty) for value in memory]
    samples = [scale * sum(per_layer) for per_layer in zip(*(layer.raw for layer in middle))]
    return _Measured(
        samples,
        scale * sum(layer.mean for layer in profile.layers[:head]),
        scale * sum(layer.mean for layer in profile.layers[tail:]),
        memory,
        on_dla,
        list(middle),
        scale,
    )


def characterize(
    onnx: Path | str,
    max_groups: int | None = None,
    dla_core: int = 0,
    iterations: int = 200,
    warmup_iterations: int = 20,
    workspace: float = 4.0,
    timing_cache: Path | str | None = None,
    *,
    int8: bool = False,
    memory: bool = False,
    memory_seconds: float = 1.5,
    verbose: bool | None = None,
) -> ModelTraits:
    """
    Characterize the layer groups of one DNN on the GPU and a DLA (HaX-CoNN Sec. 3.1-3.3).

    The network is split into layer groups at its transition points
    (:func:`group_layers`). Every group is built on its own, by slicing the network
    (:func:`slice_hook`), for the GPU and, if it can run there, for the DLA, in FP16 or
    INT8 (with nominal scales: INT8 timing does not depend on them).
    The full GPU engine's per-layer profile gives each group's GPU time, as the layers
    run together; each group engine's profile gives the reformats that a transition into
    and out of the group costs, and on a DLA also the group's time. A group counts as DLA-capable only
    if its DLA engine runs fully on the DLA. The DLA task launch cost, measured from an
    engine spanning several DLA groups, moves from each group's DLA time into its DLA
    transition-in cost, so a run of DLA groups pays it once. With ``memory``, each
    group's memory throughput is sampled with tegrastats, which needs root.

    Parameters
    ----------
    onnx : Path | str
        The ONNX model, with static input shapes.
    max_groups : int, optional
        The most layer groups to keep, merging the cheapest neighbours. Default is
        None (every transition point).
    dla_core : int, optional
        The DLA core to characterize on. Default is 0.
    iterations : int, optional
        Timed executions per group engine. Default is 200.
    warmup_iterations : int, optional
        Untimed executions before timing. Default is 20.
    workspace : float, optional
        The TensorRT workspace in GiB. Default is 4.0.
    timing_cache : Path | str, optional
        A timing cache file shared by every build; slices of a network build much
        faster with it. Default is None (a temporary one).
    int8 : bool, optional
        Whether to measure INT8 engines (the paper's Orin setting) instead of FP16.
        Default is False.
    memory : bool, optional
        Whether to measure memory throughput, needed for contention-aware scheduling.
        Default is False.
    memory_seconds : float, optional
        Seconds of memory sampling per group engine. Default is 1.5.
    verbose : bool, optional
        Whether to log progress. Default is None (False).

    Returns
    -------
    ModelTraits
        The traits of the DNN.

    Raises
    ------
    RuntimeError
        If ``memory`` is set and tegrastats cannot report memory throughput.

    """
    if memory:
        _check_tegrastats()
    layers = inspect_onnx_layers(onnx)
    mem_seconds = memory_seconds if memory else None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cache = timing_cache if timing_cache is not None else tmp_dir / "timing.cache"
        settings = {
            "workspace": workspace,
            "timing_cache": cache,
            "int8": int8,
            "verbose": verbose,
        }

        full_path = tmp_dir / "full.engine"
        _build(onnx, full_path, dla_core=None, hooks=None, **settings)
        full = _measure_group(full_path, None, iterations, warmup_iterations, None)
        names = [layer.name for layer in full.layers]
        matches = match_engine_layers(names, layers)
        weights = [0.0] * len(layers)
        for layer, matched in zip(full.layers, matches):
            for index in matched:
                weights[index] += layer.mean / len(matched)
        groups = group_layers(layers, weights, fused_spans(names, layers), max_groups)
        gpu_times = _split_times(full, matches, groups)
        if verbose:
            LOG.info(f"HaX-CoNN characterize {onnx}: {len(layers)} layers, {len(groups)} groups")

        measured: list[dict[str, _Measured]] = []
        for k, group in enumerate(groups):
            hooks = [slice_hook(group.first, group.last)]
            per_acc: dict[str, _Measured] = {}
            gpu_path = tmp_dir / f"group{k}_gpu.engine"
            _build(onnx, gpu_path, dla_core=None, hooks=hooks, **settings)
            # the full engine times the group's layers as they run together; a slice
            # alone loses the optimizations across its boundaries, so it only gives the
            # transitions and the memory throughput
            per_acc["gpu"] = _measure_group(
                gpu_path, None, iterations, warmup_iterations, mem_seconds
            )._replace(time_ms=gpu_times[k])
            gpu_path.unlink()
            if group.dla:
                dla_path = tmp_dir / f"group{k}_dla.engine"
                _build(onnx, dla_path, dla_core=dla_core, hooks=hooks, **settings)
                result = _measure_group(
                    dla_path, dla_core, iterations, warmup_iterations, mem_seconds
                )
                dla_path.unlink()
                if result.on_dla:
                    per_acc["dla"] = result
            measured.append(per_acc)
            if verbose:
                times = {acc: f"{np.mean(m.time_ms):.3f} ms" for acc, m in per_acc.items()}
                LOG.info(f"  group {group.first}-{group.last}: {times}")

        launch = _dla_launch(
            onnx, groups, measured, tmp_dir, dla_core, iterations, warmup_iterations, settings
        )
        if verbose:
            LOG.info(f"  DLA launch cost: {launch:.3f} ms")

    traits: list[GroupTraits] = []
    for group, per_acc in zip(groups, measured):
        time_ms = {acc: m.time_ms for acc, m in per_acc.items()}
        t_in = {acc: m.transition_in_ms for acc, m in per_acc.items()}
        if "dla" in per_acc:
            time_ms["dla"] = [max(t - launch, 0.0) for t in time_ms["dla"]]
            t_in["dla"] += launch
        traits.append(
            GroupTraits(
                first=group.first,
                last=group.last,
                time_ms=time_ms,
                transition_in_ms=t_in,
                transition_out_ms={acc: m.transition_out_ms for acc, m in per_acc.items()},
                memory={acc: m.memory for acc, m in per_acc.items() if m.memory},
            )
        )
    return ModelTraits(groups=traits, dla_launch_ms=launch, precision="int8" if int8 else "fp16")


def _split_times(
    full: _Measured,
    matches: Sequence[Sequence[int]],
    groups: Sequence[LayerGroup],
) -> list[list[float]]:
    # per group, per iteration: the full engine's layer times, a fused layer split
    # across groups by how many of its network layers each holds
    owner = {i: k for k, group in enumerate(groups) for i in range(group.first, group.last + 1)}
    iterations = len(full.layers[0].raw) if full.layers else 0
    times = [np.zeros(iterations) for _ in groups]
    for layer, matched in zip(full.layers, matches):
        for index in matched:
            times[owner[index]] += full.scale * np.asarray(layer.raw) / len(matched)
    return [t.tolist() for t in times]


def _dla_launch(
    onnx: Path | str,
    groups: Sequence[LayerGroup],
    measured: Sequence[dict[str, _Measured]],
    tmp_dir: Path,
    dla_core: int,
    iterations: int,
    warmup_iterations: int,
    settings: dict,
) -> float:
    # the longest run of adjacent DLA groups: one engine for the run pays the launch once
    best: list[int] = []
    run: list[int] = []
    for k, per_acc in enumerate(measured):
        run = [*run, k] if "dla" in per_acc else []
        if len(run) > len(best):
            best = run
    if len(best) < 2:  # noqa: PLR2004
        return 0.0
    path = tmp_dir / "span_dla.engine"
    hooks = [slice_hook(groups[best[0]].first, groups[best[-1]].last)]
    _build(onnx, path, dla_core=dla_core, hooks=hooks, **settings)
    span = _measure_group(path, dla_core, iterations, warmup_iterations, None)
    parts = sum(float(np.mean(measured[k]["dla"].time_ms)) for k in best)
    return max((parts - float(np.mean(span.time_ms))) / (len(best) - 1), 0.0)


def _probe(size: int, kernel: int, channels: int, depth: int) -> Callable:
    # a stack of random-weight convolutions; the size and kernel set its memory demand
    def _define(network: trt.INetworkDefinition, _config: trt.IBuilderConfig) -> list[np.ndarray]:
        rng = np.random.default_rng(0)
        tensor = network.add_input("input", trt.float32, (1, channels, size, size))
        arrays: list[np.ndarray] = []
        for _ in range(depth):
            weight = (rng.standard_normal((channels, channels, kernel, kernel)) * 0.01).astype(
                np.float32
            )
            bias = np.zeros(channels, dtype=np.float32)
            conv = network.add_convolution_nd(
                tensor, channels, (kernel, kernel), trt.Weights(weight), trt.Weights(bias)
            )
            conv.padding_nd = (kernel // 2, kernel // 2)
            tensor = conv.get_output(0)
            arrays.extend((weight, bias))
        network.mark_output(tensor)
        return arrays

    return _define


def calibrate(
    dla_core: int = 0,
    sizes: Sequence[int] = (64, 128, 192, 256),
    kernels: Sequence[int] = (1, 3),
    channels: int = 64,
    depth: int = 4,
    seconds: float = 1.5,
    workspace: float = 4.0,
    *,
    int8: bool = False,
    verbose: bool | None = None,
) -> dict[str, Calibration]:
    """
    Calibrate the contention between the GPU and a DLA for PCCS (HaX-CoNN Sec. 3.3).

    Synthetic convolution probes (every size and kernel) are built for the GPU and the
    DLA. Each probe's standalone latency and memory throughput are measured, then every
    GPU probe runs concurrently with every DLA probe, both in free-running loops on
    their own threads, and both speeds relative to standalone are recorded. Needs
    tegrastats as root.

    Parameters
    ----------
    dla_core : int, optional
        The DLA core to calibrate. Default is 0.
    sizes : Sequence[int], optional
        The probe input sizes (square). Default is (64, 128, 192, 256).
    kernels : Sequence[int], optional
        The probe kernel sizes. Default is (1, 3).
    channels : int, optional
        The probe channels. Default is 64.
    depth : int, optional
        The convolutions of the largest probe; smaller probes get proportionally more,
        so every probe does about the same work and runs long enough that launch
        overhead (and two loops sharing the CPU) does not slow it down. Default is 4.
    seconds : float, optional
        Seconds per measurement loop. Default is 1.5.
    workspace : float, optional
        The TensorRT workspace in GiB. Default is 4.0.
    int8 : bool, optional
        Whether the probes run in INT8 instead of FP16; match the DNNs' precision.
        Default is False.
    verbose : bool, optional
        Whether to log progress. Default is None (False).

    Returns
    -------
    dict[str, Calibration]
        The calibration of the "gpu" and "dla", for :meth:`PCCSModel.from_calibration`.

    Raises
    ------
    RuntimeError
        If tegrastats cannot report memory throughput.

    """
    _check_tegrastats()
    probes = [(size, kernel) for size in sizes for kernel in kernels]
    largest = max(sizes)
    engines: dict[str, list[TRTEngine]] = {"gpu": [], "dla": []}
    solo: dict[str, list[float]] = {"gpu": [], "dla": []}
    demand: dict[str, list[float]] = {"gpu": [], "dla": []}
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "timing.cache"
        for acc, core in (("gpu", None), ("dla", dla_core)):
            for size, kernel in probes:
                path = Path(tmp) / f"probe_{acc}_{size}_{kernel}.engine"
                _build(
                    _probe(size, kernel, channels, math.ceil(depth * (largest / size) ** 2)),
                    path,
                    dla_core=core,
                    hooks=None,
                    workspace=workspace,
                    timing_cache=cache,
                    int8=int8,
                    verbose=verbose,
                )
                engine = _load(path, core)
                step = _busy_step(engine)
                step()
                solo[acc].append(float(np.median(_per_execute_ms(_free_run(step, seconds)))))
                demand[acc].append(float(np.median(measure_memory(step, seconds))))
                engines[acc].append(engine)
                if verbose:
                    LOG.info(
                        f"HaX-CoNN probe {acc} {size}x{kernel}: {solo[acc][-1]:.3f} ms, "
                        f"EMC {demand[acc][-1]:.1f}%"
                    )

    speed = {"gpu": np.zeros((len(probes), len(probes))), "dla": np.zeros((len(probes),) * 2)}
    for i, gpu in enumerate(engines["gpu"]):
        for j, dla in enumerate(engines["dla"]):
            gpu_ms, dla_ms = _corun(_busy_step(gpu), _busy_step(dla), seconds)
            speed["gpu"][i, j] = min(100.0, 100.0 * solo["gpu"][i] / gpu_ms)
            speed["dla"][j, i] = min(100.0, 100.0 * solo["dla"][j] / dla_ms)
    engines.clear()
    gc.collect()

    calibration: dict[str, Calibration] = {}
    for acc, other in (("gpu", "dla"), ("dla", "gpu")):
        rows = np.argsort(demand[acc], kind="stable")
        cols = np.argsort(demand[other], kind="stable")
        calibration[acc] = Calibration(
            accelerator=acc,
            demand=[demand[acc][i] for i in rows],
            external=[demand[other][j] for j in cols],
            relative_speed=speed[acc][np.ix_(rows, cols)].tolist(),
        )
    return calibration


def _corun(
    first: Callable[[], None],
    second: Callable[[], None],
    seconds: float,
) -> tuple[float, float]:
    # free-running loops on two threads; only iterations while both ran count
    stamps: list[list[tuple[float, float]]] = [[], []]

    def _loop(index: int, step: Callable[[], None]) -> None:
        stamps[index] = _free_run(step, seconds)

    threads = [
        threading.Thread(target=_loop, args=(i, step)) for i, step in enumerate((first, second))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lo = max(s[0][0] for s in stamps) + _CORUN_TRIM_S[0]
    hi = min(s[-1][1] for s in stamps) - _CORUN_TRIM_S[1]
    medians = []
    for loop in stamps:
        inside = [s for s in loop if s[0] >= lo and s[1] <= hi] or loop
        medians.append(float(np.median(_per_execute_ms(inside))))
    return medians[0], medians[1]
