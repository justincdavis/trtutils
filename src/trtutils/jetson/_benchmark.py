# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jetsontools import Tegrastats, parse_tegrastats, get_energy

from trtutils._benchmark import Metric
from trtutils._engine import TRTEngine

_log = logging.getLogger(__name__)


@dataclass
class JetsonBenchmarkResult:
    latency: Metric
    energy: Metric


def benchmark_engine(
    engine: TRTEngine | Path | str,
    iterations: int = 1000,
    warmup_iterations: int = 50,
    tegra_interval: int = 5,
    *,
    warmup: bool | None = None,
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

    # create temp file location for data to go
    temp_file = Path("/temp/tegra.txt")
    # store the start/stop times of the engine execution
    start_stop: list[tuple[float, float]] = []
    with Tegrastats(temp_file, interval=tegra_interval):
        for _ in range(iterations):
            t0 = time.time()
            engine.mock_execute(false_data)
            t1 = time.time()
            raw["latency"].append(t1 - t0)
            start_stop.append((t0, t1))

    # parse the tegra data
    tegradata = parse_tegrastats(temp_file)

    # filter out all entries which are not during execution
    # TODO: implement
    filtered_data = tegradata

    # get the energy values
    energy_data = get_energy(filtered_data)

    # calculate the metrics
    metrics: dict[str, Metric] = {}
    for metric in metric_names:
        data = np.array(raw[metric])
        avg = float(np.mean(data))
        min_ = float(np.min(data))
        max_ = float(np.max(data))

        _log.info(f"{metric}: avg={avg:.6f}, min={min_:.6f}, max={max_:.6f}")
        metrics[metric] = Metric(avg, min_, max_)

    return JetsonBenchmarkResult(
        latency=metrics["latency"],
    )
