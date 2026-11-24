# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jetsontools import TegraStats, filter_data, get_powerdraw, parse_tegrastats

from trtutils._benchmark import Metric, _check_iterations
from trtutils._engine import TRTEngine
from trtutils._log import LOG
from trtutils.parallel._parallel_engines import ParallelTRTEngines

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self


@dataclass
class JetsonBenchmarkResult:
    latency: Metric
    power_draw: Metric
    energy: Metric

    def __str__(self: Self) -> str:
        return f"JetsonBenchmarkResult(latency={self.latency}, power_draw={self.power_draw}, energy={self.energy})"

    def __repr__(self: Self) -> str:
        return f"JetsonBenchmarkResult(latency={self.latency!r}, power_draw={self.power_draw!r}, energy={self.energy!r})"


def _collect_power_data(
    temp_file: Path,
    start_stop_times: list[tuple[float, float]],
    tegra_interval: int,
) -> tuple[list[float], list[float]]:
    """
    Parse tegrastats output and extract power draw and per-inference energy.

    Parameters
    ----------
    temp_file : Path
        The tegrastats output file. It is deleted after parsing.
    start_stop_times : list[tuple[float, float]]
        The (start, stop) time of each inference.
    tegra_interval : int
        The tegrastats sampling interval in milliseconds, used in error messages.

    Returns
    -------
    tuple[list[float], list[float]]
        The power draw samples and the energy of each inference with at least one sample.

    Raises
    ------
    RuntimeError
        If no tegrastats sample fell inside any inference window.

    """
    tegradata = parse_tegrastats(temp_file)
    temp_file.unlink(missing_ok=True)

    # filter the data by actual times during execution
    filtered_data, per_inference = filter_data(tegradata, start_stop_times)
    if len(filtered_data) == 0:
        err_msg = (
            "No tegrastats samples were captured during inference, so power draw and energy "
            f"cannot be computed (tegra_interval={tegra_interval} ms, "
            f"iterations={len(start_stop_times)}). "
            "Lower tegra_interval or increase iterations."
        )
        raise RuntimeError(err_msg)

    power_draw = get_powerdraw(filtered_data)["VDD_TOTAL"].raw

    # for energy values need to compute powerdraw per inference, then compute energy
    energy = [
        get_powerdraw(inf_data)["VDD_TOTAL"].mean * (inf_stop - inf_start)
        for (inf_start, inf_stop), inf_data in per_inference
        if len(inf_data) > 0
    ]
    if len(energy) < len(start_stop_times):
        LOG.warning(
            f"Only {len(energy)} of {len(start_stop_times)} inferences had a tegrastats sample. "
            "Energy is computed over fewer iterations than latency; "
            "lower tegra_interval for full coverage."
        )
    return power_draw, energy


def benchmark_engine(
    engine: TRTEngine | Path | str,
    iterations: int = 1000,
    warmup_iterations: int = 50,
    tegra_interval: int = 5,
    dla_core: int | None = None,
    *,
    warmup: bool | None = None,
    cuda_graph: bool = False,
    verbose: bool | None = None,
) -> JetsonBenchmarkResult:
    """
    Benchmark a TensorRT engine on a Jetson device.

    Parameters
    ----------
    engine : TRTEngine | Path | str
        The engine to benchmark. Either a TRTEngine object or path to the engine file.
        If a path is given, then a TRTEngine will be created automatically.
    iterations : int, optional
        The number of iterations to run the benchmark for, by default 1000.
    warmup_iterations : int, optional
        The number of warmup iterations to run before the benchmark, by default 50.
    tegra_interval : int, optional
        The number of milliseconds between each tegrastats sampling.
        The smaller the number, the more samples per second are generated.
        By default 5 milliseconds between samples.
    dla_core : int, optional
        The DLA core to assign DLA layers of the engine to. Default is None.
        If None, any DLA layers will be assigned to DLA core 0.
    warmup : bool, optional
        Whether to do warmup iterations, by default None
        If None, warmup will be set to True.
    cuda_graph : bool, optional
        Whether to enable CUDA graph capture for the underlying TRTEngine.
        Default is False. Leave False for engines with DLA layers, as DLA
        does not support CUDA graphs.
    verbose : bool, optional
        Whether ot not to output additional information to stdout.
        Default None/False.

    Returns
    -------
    JetsonBenchmarkResult
        A dataclass containing the results of the benchmark.

    Raises
    ------
    ValueError
        If iterations is less than 1.
    RuntimeError
        If no tegrastats sample falls inside any inference, e.g. a fast engine with a large
        tegra_interval. Lower tegra_interval or increase iterations.

    """
    _check_iterations(iterations)
    warmup = warmup if warmup is not None else True
    loaded_engine: bool = False
    if isinstance(engine, (Path, str)):
        engine = TRTEngine(
            engine,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
            dla_core=dla_core,
            cuda_graph=cuda_graph,
            verbose=verbose,
        )
        loaded_engine = True
    else:
        if warmup:
            for _ in range(warmup_iterations):
                engine.mock_execute()

    # list of metrics
    metric_names = ["latency", "power_draw", "energy"]
    raw: dict[str, list[float]] = {metric: [] for metric in metric_names}

    # pre-generate the false data
    false_data = engine.get_random_input(verbose=verbose)

    # create temp file location for data to go
    temp_file = Path(Path.cwd()) / "temptegra.txt"
    # store the start/stop times of the engine execution
    start_stop_times: list[tuple[float, float]] = []
    with TegraStats(temp_file, interval=tegra_interval):
        for _ in range(iterations):
            t0 = time.time()
            engine.mock_execute(false_data, verbose=verbose)
            t1 = time.time()
            raw["latency"].append(t1 - t0)
            start_stop_times.append((t0, t1))

    raw["power_draw"], raw["energy"] = _collect_power_data(
        temp_file, start_stop_times, tegra_interval
    )

    # calculate the metrics
    metrics: dict[str, Metric] = {}
    for metric_name in metric_names:
        data = raw[metric_name]
        metric = Metric(data)
        metrics[metric_name] = metric
        LOG.debug(f"{metric_name}: {metric}")

    if loaded_engine:
        del engine
        gc.collect()

    return JetsonBenchmarkResult(
        latency=metrics["latency"],
        power_draw=metrics["power_draw"],
        energy=metrics["energy"],
    )


def benchmark_engines(
    engines: Sequence[TRTEngine | Path | str | tuple[TRTEngine | Path | str, int]],
    iterations: int = 1000,
    warmup_iterations: int = 50,
    tegra_interval: int = 5,
    *,
    warmup: bool | None = None,
    cuda_graph: bool = False,
    parallel: bool | None = None,
    verbose: bool | None = None,
) -> list[JetsonBenchmarkResult]:
    """
    Benchmark a TensorRT engine.

    Parameters
    ----------
    engines : Sequence[TRTEngine | Path | str | tuple[TRTEngine | Path | str, int]]
        The engines to benchmark as paths to the engine files.
    iterations : int, optional
        The number of iterations to run the benchmark for, by default 1000.
    warmup_iterations : int, optional
        The number of warmup iterations to run before the benchmark, by default 50.
    tegra_interval : int, optional
        The number of milliseconds between each tegrastats sampling.
        The smaller the number, the more samples per second are generated.
        By default 5 milliseconds between samples.
    warmup : bool, optional
        Whether to do warmup iterations, by default None
        If None, warmup will be set to True.
    cuda_graph : bool, optional
        Whether to enable CUDA graph capture for the underlying TRTEngines.
        Default is False. Leave False for engines with DLA layers, as DLA
        does not support CUDA graphs.
    parallel : bool, optional
        Whether or not to process the engines in parallel.
        Useful for assessing concurrent execution performance.
        Will execute the engines in lockstep.
        If None, will benchmark each engine individually.
    verbose : bool, optional
        Whether ot not to output additional information to stdout.
        Default None/False.

    Returns
    -------
    list[JetsonBenchmarkResult]
        A list of dataclasses containing the results of the benchmark.
        If parallel was True, will only contain one item.

    Raises
    ------
    ValueError
        If iterations is less than 1.
    RuntimeError
        If no tegrastats sample falls inside any inference, e.g. a fast engine with a large
        tegra_interval. Lower tegra_interval or increase iterations.

    """
    _check_iterations(iterations)
    warmup = warmup if warmup is not None else True
    temp_engines: list[Path | TRTEngine] = []
    dla_assignments: list[int | None] = []
    for engine_info in engines:
        dla_core: int | None = None
        if isinstance(engine_info, tuple):
            engine, dla_core = engine_info
        else:
            engine = engine_info
        if isinstance(engine, str):
            engine = Path(engine)
        temp_engines.append(engine)
        dla_assignments.append(dla_core)

    if not parallel:
        return [
            benchmark_engine(
                engine,
                iterations,
                warmup_iterations,
                tegra_interval,
                dla_core=dla_core,
                warmup=warmup,
                cuda_graph=cuda_graph,
                verbose=verbose,
            )
            for engine, dla_core in zip(temp_engines, dla_assignments)
        ]

    # otherwise we need a parallel setup
    trt_engines = ParallelTRTEngines(
        [(ep, dc) if dc is not None else ep for ep, dc in zip(temp_engines, dla_assignments)],
        warmup_iterations=warmup_iterations,
        warmup=warmup,
        cuda_graph=cuda_graph,
    )

    # list of metrics
    metric_names = ["latency", "power_draw", "energy"]
    raw: dict[str, list[float]] = {metric: [] for metric in metric_names}

    # pre-generate the false data
    false_data = trt_engines.get_random_input()

    # create temp file location for data to go
    temp_file = Path(Path.cwd()) / "temptegra.txt"
    # store the start/stop times of the engine execution
    start_stop_times: list[tuple[float, float]] = []
    with TegraStats(temp_file, interval=tegra_interval):
        for _ in range(iterations):
            t0 = time.time()
            trt_engines.submit(false_data)
            trt_engines.retrieve()
            t1 = time.time()
            raw["latency"].append(t1 - t0)
            start_stop_times.append((t0, t1))

    raw["power_draw"], raw["energy"] = _collect_power_data(
        temp_file, start_stop_times, tegra_interval
    )

    # calculate the metrics
    metrics: dict[str, Metric] = {}
    for metric_name in metric_names:
        data = raw[metric_name]
        metric = Metric(data)
        metrics[metric_name] = metric
        LOG.debug(f"{metric_name}: {metric}")

    # no need to check if we loaded the engines, since passed engines are not deleted
    # explicitly by the ParallelTRTEngines class
    del trt_engines
    gc.collect()

    return [
        JetsonBenchmarkResult(
            latency=metrics["latency"],
            power_draw=metrics["power_draw"],
            energy=metrics["energy"],
        ),
    ]
