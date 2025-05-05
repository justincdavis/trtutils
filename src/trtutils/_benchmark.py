# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

from ._engine import ParallelTRTEngines, TRTEngine
from ._log import LOG

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self


@dataclass
class Metric:
    """A dataclass to store the results of a benchmark."""

    raw: list[float | int]
    mean: float | int = -1.0
    median: float | int = -1.0
    min: float | int = -1.0
    max: float | int = -1.0

    def __post_init__(self: Self) -> None:
        if not self.raw:
            err_msg = "Raw data cannot be empty"
            raise ValueError(err_msg)

        self.min = min(self.raw)
        self.median = median(self.raw)
        self.max = max(self.raw)
        self.mean = mean(self.raw)

    def __str__(self: Self) -> str:
        return f"Metric(mean={self.mean:.3f}, median={self.median:.3f}, min={self.min:.3f}, max={self.max:.3f})"

    def __repr__(self: Self) -> str:
        return f"Metric(mean={self.mean},median={self.median},min={self.min},max={self.max})"


@dataclass
class BenchmarkResult:
    """A dataclass to store the results of a benchmark."""

    latency: Metric

    def __str__(self: Self) -> str:
        return f"BenchmarkResult(latency={self.latency})"

    def __repr__(self: Self) -> str:
        return f"BenchmarkResult(latency={self.latency!r})"


def benchmark_engine(
    engine: TRTEngine | Path | str,
    iterations: int = 1000,
    warmup_iterations: int = 50,
    dla_core: int | None = None,
    *,
    warmup: bool | None = None,
    verbose: bool | None = None,
) -> BenchmarkResult:
    """
    Benchmark a TensorRT engine.

    Parameters
    ----------
    engine : TRTEngine | Path | str
        The engine to benchmark. Either a TRTEngine object or path to the engine file.
        If a path is given, then a TRTEngine will be created automatically.
    iterations : int, optional
        The number of iterations to run the benchmark for, by default 1000.
    warmup_iterations : int, optional
        The number of warmup iterations to run before the benchmark, by default 50.
    dla_core : int, optional
        The DLA core to assign DLA layers of the engine to. Default is None.
        If None, any DLA layers will be assigned to DLA core 0.
    warmup : bool, optional
        Whether to do warmup iterations, by default None
        If None, warmup will be set to True.
    verbose : bool, optional
        Whether ot not to output additional information to stdout.
        Default None/False.

    Returns
    -------
    BenchmarkResult
        A dataclass containing the results of the benchmark.

    """
    if verbose:
        LOG.debug("Running benchmark_engine")

    if isinstance(engine, (Path, str)):
        engine = TRTEngine(
            engine,
            warmup_iterations=warmup_iterations,
            dla_core=dla_core,
            warmup=warmup,
            verbose=verbose,
        )
    else:
        if warmup:
            for _ in range(warmup_iterations):
                engine.mock_execute(verbose=verbose)

    # list of metrics
    metric_names = ["latency"]

    # allocate spot for raw data
    raw: dict[str, list[float]] = {metric: [] for metric in metric_names}

    # pre-generate the false data
    false_data = engine.get_random_input(verbose=verbose)

    for _ in range(iterations):
        t0 = time.time()
        engine.mock_execute(false_data, verbose=verbose)
        t1 = time.time()

        raw["latency"].append(t1 - t0)

    # calculate the metrics
    metrics: dict[str, Metric] = {}
    for metric_name in metric_names:
        data = raw[metric_name]
        metric = Metric(data)
        metrics[metric_name] = metric
        LOG.debug(f"{metric_name}: {metric}")

    return BenchmarkResult(
        latency=metrics["latency"],
    )


def benchmark_engines(
    engines: Sequence[TRTEngine | Path | str | tuple[TRTEngine | Path | str, int]],
    iterations: int = 1000,
    warmup_iterations: int = 50,
    *,
    warmup: bool | None = None,
    parallel: bool | None = None,
    verbose: bool | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark a TensorRT engine.

    Parameters
    ----------
    engines : Sequence[TRTEngine | Path | str | tuple[TRTEngine | Path | str, int]],
        The engines to benchmark as paths to the engine files.
    iterations : int, optional
        The number of iterations to run the benchmark for, by default 1000.
    warmup_iterations : int, optional
        The number of warmup iterations to run before the benchmark, by default 50.
    warmup : bool, optional
        Whether to do warmup iterations, by default None
        If None, warmup will be set to True.
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
    list[BenchmarkResult]
        A list of dataclasses containing the results of the benchmark.
        If parallel was True, will only contain one item.

    """
    temp_engines: list[Path | TRTEngine] = []
    dla_assignments: list[int | None] = []
    for engine_info in engines:
        engine: TRTEngine | Path | str
        dla_core: int | None = None
        if isinstance(engine_info, tuple):
            engine = engine_info[0]
            dla_core = engine_info[1]
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
                dla_core=dla_core,
                warmup=warmup,
                verbose=verbose,
            )
            for engine, dla_core in zip(temp_engines, dla_assignments)
        ]

    # otherwise we need a parallel setup
    trt_engines = ParallelTRTEngines(
        [
            (ep, dc) if dc is not None else ep
            for ep, dc in zip(temp_engines, dla_assignments)
        ],
        warmup_iterations=warmup_iterations,
        warmup=warmup,
    )

    # list of metrics
    metric_names = ["latency"]

    # allocate spot for raw data
    raw: dict[str, list[float]] = {metric: [] for metric in metric_names}

    # pre-generate the false data
    false_data = trt_engines.get_random_input()

    for _ in range(iterations):
        t0 = time.time()
        trt_engines.submit(false_data)
        trt_engines.retrieve()
        t1 = time.time()

        raw["latency"].append(t1 - t0)

    # calculate the metrics
    metrics: dict[str, Metric] = {}
    for metric_name in metric_names:
        data = raw[metric_name]
        metric = Metric(data)
        metrics[metric_name] = metric
        LOG.debug(f"{metric_name}: {metric}")

    return [
        BenchmarkResult(
            latency=metrics["latency"],
        ),
    ]
