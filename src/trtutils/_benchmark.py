# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

from ._engine import TRTEngine

if TYPE_CHECKING:
    from typing_extensions import Self

_log = logging.getLogger(__name__)


@dataclass
class Metric:
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


@dataclass
class BenchmarkResult:
    """A dataclass to store the results of a benchmark."""

    latency: Metric


def benchmark_engine(
    engine: TRTEngine | Path | str,
    iterations: int = 1000,
    warmup_iterations: int = 50,
    *,
    warmup: bool | None = None,
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
    warmup : bool, optional
        Whether to do warmup iterations, by default None
        If None, warmup will be set to True.

    Returns
    -------
    BenchmarkResult
        A dataclass containing the results of the benchmark.

    """
    if isinstance(engine, (Path, str)):
        engine = TRTEngine(engine, warmup_iterations=warmup_iterations, warmup=warmup)
    else:
        if warmup:
            for _ in range(warmup_iterations):
                engine.mock_execute()

    # list of metrics
    metric_names = ["latency"]

    # allocate spot for raw data
    raw: dict[str, list[float]] = {metric: [] for metric in metric_names}

    # pre-generate the false data
    false_data = engine.get_random_input()

    for _ in range(iterations):
        t0 = time.time()
        engine.mock_execute(false_data)
        t1 = time.time()

        raw["latency"].append(t1 - t0)

    # calculate the metrics
    metrics: dict[str, Metric] = {}
    for metric_name in metric_names:
        data = raw[metric_name]
        metric = Metric(data)
        metrics[metric_name] = metric
        _log.info(
            f"{metric_name}: mean={metric.mean:.6f}, median={metric.median:.6f}, min={metric.min:.6f}, max={metric.max:.6f}",
        )

    return BenchmarkResult(
        latency=metrics["latency"],
    )
