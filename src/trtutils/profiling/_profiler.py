# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

from trtutils._engine import TRTEngine
from trtutils._log import LOG
from trtutils.compat._libs import trt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self


@dataclass
class LayerTiming:
    """
    A dataclass to store per-layer profiling statistics.

    Attributes
    ----------
    name : str
        The name of the layer.
    mean : float
        The mean execution time in milliseconds.
    median : float
        The median execution time in milliseconds.
    min : float
        The minimum execution time in milliseconds.
    max : float
        The maximum execution time in milliseconds.
    raw : list[float]
        The raw execution times in milliseconds across all iterations.

    """

    name: str
    mean: float
    median: float
    min: float
    max: float
    raw: list[float]

    def __str__(self: Self) -> str:
        return f"{self.name}: mean={self.mean:.3f}ms, median={self.median:.3f}ms, min={self.min:.3f}ms, max={self.max:.3f}ms"

    def __repr__(self: Self) -> str:
        return f"LayerTiming(name={self.name!r}, mean={self.mean}, median={self.median}, min={self.min}, max={self.max})"


@dataclass
class ProfilerResult:
    """
    A dataclass to store the complete profiling results.

    Attributes
    ----------
    layers : list[LayerTiming]
        The per-layer timing statistics.
    total_time : LayerTiming
        The total execution time statistics across all layers.
    iterations : int
        The number of profiling iterations performed.

    """

    layers: Sequence[LayerTiming]
    total_time: LayerTiming
    iterations: int

    def __str__(self: Self) -> str:
        return f"ProfilerResult(layers={len(self.layers)}, total_time={self.total_time.mean:.3f}ms, iterations={self.iterations})"

    def __repr__(self: Self) -> str:
        return f"ProfilerResult(layers={self.layers!r}, total_time={self.total_time!r}, iterations={self.iterations})"


class LayerProfiler(trt.IProfiler):
    """
    A profiler that implements TensorRT's IProfiler interface.

    This class collects per-layer execution times across multiple inference iterations
    and can aggregate statistics for each layer.
    """

    def __init__(self: Self) -> None:
        """Initialize the LayerProfiler."""
        super().__init__()
        # Store timings for each layer across iterations
        # Key: layer_name, Value: list of timings in milliseconds
        self._timings: dict[str, list[float]] = defaultdict(list)
        # Store current iteration timings (reset after each report_to_profiler call)
        self._current_iteration: dict[str, float] = {}

    def report_layer_time(self: Self, layer_name: str, ms: float) -> None:
        """
        Record the execution time for a layer.

        This method is called by TensorRT once per layer after inference, only
        if the profiler is not added to the context.

        Parameters
        ----------
        layer_name : str
            The name of the layer.
        ms : float
            The execution time in milliseconds.

        """
        self._current_iteration[layer_name] = ms

    def finalize_iteration(self: Self) -> None:
        """
        Finalize the current iteration by storing all layer timings.

        This should be called after context.report_to_profiler() to commit
        the current iteration's timings to the aggregate storage.
        """
        for layer_name, time_ms in self._current_iteration.items():
            self._timings[layer_name].append(time_ms)
        self._current_iteration.clear()

    def get_statistics(self: Self) -> list[LayerTiming]:
        """
        Compute statistics for each layer across all iterations.

        Returns
        -------
        list[LayerTiming]
            A list of LayerTiming objects, one per layer, with aggregated statistics.

        """
        layer_stats: list[LayerTiming] = []

        for layer_name, times in self._timings.items():
            if not times:
                continue

            layer_timing = LayerTiming(
                name=layer_name,
                mean=mean(times),
                median=median(times),
                min=min(times),
                max=max(times),
                raw=times.copy(),
            )
            layer_stats.append(layer_timing)

        return layer_stats

    def reset(self: Self) -> None:
        """Reset all stored timings."""
        self._timings.clear()
        self._current_iteration.clear()


def profile_engine(
    engine: Path | str | TRTEngine,
    iterations: int = 100,
    warmup_iterations: int = 10,
    dla_core: int | None = None,
    device: int | None = None,
    *,
    warmup: bool | None = None,
    verbose: bool | None = None,
) -> ProfilerResult:
    """
    Profile a TensorRT engine layer-by-layer.

    This function runs inference multiple times and collects per-layer execution
    times using TensorRT's IProfiler interface. It returns aggregated statistics
    (mean, median, min, max) for each layer across all iterations.

    Notes
    -----
    For best results, build the engine with profiling_verbosity set to DETAILED
    when calling build_engine. Otherwise, layer names may be numeric indices.

    Parameters
    ----------
    engine : Path | str | TRTEngine
        The engine to profile. Either a TRTEngine object or path to the engine file.
        If a path is given, then a TRTEngine will be created automatically.
    iterations : int, optional
        The number of profiling iterations to run, by default 100.
    warmup_iterations : int, optional
        The number of warmup iterations to run before profiling, by default 10.
    dla_core : int, optional
        The DLA core to assign DLA layers of the engine to. Default is None.
        If None, any DLA layers will be assigned to DLA core 0.
    device : int, optional
        The CUDA device index to use for the engine. Default is None,
        which uses the current device.
    warmup : bool, optional
        Whether to do warmup iterations, by default None.
        If None, warmup will be set to True.
    verbose : bool, optional
        Whether to output additional information to stdout.
        Default None/False.

    Returns
    -------
    ProfilerResult
        A dataclass containing per-layer timing statistics and total execution time.

    """
    if verbose:
        LOG.info("Starting engine profiling")

    if warmup is None:
        warmup = True

    engine_loaded = False
    if isinstance(engine, (Path, str)):
        engine = TRTEngine(
            engine,
            dla_core=dla_core,
            device=device,
            warmup=False,
            verbose=verbose,
        )
        engine_loaded = True

    # issue warning if not build with detailed
    engine_verbosity = engine.engine.profiling_verbosity
    if engine_verbosity != trt.ProfilingVerbosity.DETAILED and verbose:
        LOG.warning(
            "Engine profiling verbosity is not DETAILED. Layer names may be numeric indices. "
            "Rebuild the engine with profiling_verbosity=trt.ProfilingVerbosity.DETAILED for best results.",
        )

    # attach profiler
    profiler = LayerProfiler()
    engine.context.profiler = profiler

    # do warmup iterations
    # always do a single pass regardless of warmup_iterations
    engine.mock_execute(verbose=False)
    if warmup:
        for _ in range(warmup_iterations):
            engine.mock_execute(verbose=False)
    # report_layer_time is called by the context, so reset after warmup
    profiler.reset()

    if verbose:
        LOG.info(f"Running {iterations} profiling iterations")

    for idx in range(iterations):
        engine.mock_execute(verbose=False)
        profiler.finalize_iteration()

        if verbose and (idx + 1) % 10 == 0:
            LOG.info(f"Completed {idx + 1}/{iterations} iterations")

    layer_stats = profiler.get_statistics()

    if verbose:
        LOG.info(f"Profiling complete: {len(layer_stats)} layers profiled")

    total_times: list[float] = []
    for idx in range(iterations):
        iteration_total = sum(layer.raw[idx] for layer in layer_stats if idx < len(layer.raw))
        total_times.append(iteration_total)

    total_timing = LayerTiming(
        name="TOTAL",
        mean=mean(total_times) if total_times else 0.0,
        median=median(total_times) if total_times else 0.0,
        min=min(total_times) if total_times else 0.0,
        max=max(total_times) if total_times else 0.0,
        raw=total_times,
    )

    # if loaded here, delete engine
    if engine_loaded:
        del engine

    return ProfilerResult(
        layers=layer_stats,
        total_time=total_timing,
        iterations=iterations,
    )
