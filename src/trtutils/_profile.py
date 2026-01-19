# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from trtutils._engine import TRTEngine
from trtutils.jetson._profile import profile_engine as jetson_profile_engine
from trtutils.profiling._profiler import ProfilerResult, profile_engine as inspect_profile_engine

if TYPE_CHECKING:
    from trtutils.jetson._profile import JetsonProfilerResult


def profile_engine(
    engine: Path | str | TRTEngine,
    iterations: int = 100,
    warmup_iterations: int = 10,
    dla_core: int | None = None,
    tegra_interval: int = 5,
    *,
    jetson: bool = False,
    warmup: bool | None = None,
    verbose: bool | None = None,
) -> ProfilerResult | JetsonProfilerResult:
    """
    Profile a TensorRT engine layer-by-layer.

    This is a dispatcher function that calls either the standard profiler or
    the Jetson-specific profiler based on the jetson parameter.

    This function runs inference multiple times and collects per-layer execution
    times using TensorRT's IProfiler interface. On Jetson devices with jetson=True,
    it also collects power and energy metrics. It returns aggregated statistics
    (mean, median, min, max) for each layer across all iterations.

    Notes
    -----
    For best results, build the engine with profiling_verbosity set to DETAILED
    when calling build_engine. Otherwise, layer names may be numeric indices.

    When jetson=True, the Jetson profiler function has a default of 10000 iterations
    (instead of 100) to ensure adequate tegrastats sampling coverage across all layers.
    You can override this by explicitly providing the iterations parameter.

    Parameters
    ----------
    engine : Path | str | TRTEngine
        The engine to profile. Either a TRTEngine object or path to the engine file.
        If a path is given, then a TRTEngine will be created automatically.
    iterations : int, optional
        The number of profiling iterations to run, by default 100 for standard profiling.
        Note: The Jetson profiler uses 10000 by default if not explicitly specified.
    warmup_iterations : int, optional
        The number of warmup iterations to run before profiling, by default 10.
    dla_core : int, optional
        The DLA core to assign DLA layers of the engine to. Default is None.
        If None, any DLA layers will be assigned to DLA core 0.
    tegra_interval : int, optional
        The interval in milliseconds between tegrastats samples (Jetson only),
        by default 5. Only used when jetson=True.
    jetson : bool, optional
        Whether to use Jetson-specific profiling with power/energy metrics,
        by default False.
    warmup : bool, optional
        Whether to do warmup iterations, by default None.
        If None, warmup will be set to True.
    verbose : bool, optional
        Whether to output additional information to stdout.
        Default None/False.

    Returns
    -------
    ProfilerResult | JetsonProfilerResult
        If jetson=False: ProfilerResult containing per-layer timing statistics
        and total execution time.
        If jetson=True: JetsonProfilerResult containing per-layer timing statistics
        with power/energy data, total execution time, overall power draw, and overall
        energy consumption.

    Raises
    ------
    ImportError
        If jetson=True but the jetson module is not available.

    """
    if jetson:
        return jetson_profile_engine(
            engine=engine,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            tegra_interval=tegra_interval,
            dla_core=dla_core,
            warmup=warmup,
            verbose=verbose,
        )
    else:
        return inspect_profile_engine(
            engine=engine,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            dla_core=dla_core,
            warmup=warmup,
            verbose=verbose,
        )
