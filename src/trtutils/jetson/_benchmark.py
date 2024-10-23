# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jetsontools import Tegrastats, filter_data, get_powerdraw, parse_tegrastats

from trtutils._benchmark import Metric
from trtutils._engine import TRTEngine

if TYPE_CHECKING:
    from jetsontools._parsing import Metric as JMetric  # typing fix

_log = logging.getLogger(__name__)


@dataclass
class JetsonBenchmarkResult:
    latency: Metric
    power_draw: Metric
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
    metric_names = ["latency", "power_draw", "energy"]
    raw: dict[str, list[float]] = {metric: [] for metric in metric_names}

    # pre-generate the false data
    false_data = engine.get_random_input()

    # create temp file location for data to go
    temp_file = Path(Path.cwd()) / "temptegra.txt"
    # store the start/stop times of the engine execution
    start_stop_times: list[tuple[float, float]] = []
    with Tegrastats(temp_file, interval=tegra_interval):
        for _ in range(iterations):
            t0 = time.time()
            engine.mock_execute(false_data)
            t1 = time.time()
            raw["latency"].append(t1 - t0)
            start_stop_times.append((t0, t1))

    # parse the tegra data
    tegradata = parse_tegrastats(temp_file)

    # delete the temp file
    temp_file.unlink()

    # filter the data by actual times during execution
    filtered_data, per_inference = filter_data(tegradata, start_stop_times)

    # get the energy values
    powerdraw_data: dict[str, JMetric] = get_powerdraw(filtered_data)
    raw["power_draw"] = powerdraw_data["VDD_TOTAL"].raw

    # compute energy values
    # for energy values need to compute powerdraw per infernece
    # then compute energy
    energy_data = [
        get_powerdraw(inf_data)["VDD_TOTAL"].mean * (inf_stop - inf_start)
        for (inf_start, inf_stop), inf_data in per_inference
        if len(inf_data) > 0
    ]
    raw["energy"] = energy_data

    # calculate the metrics
    metrics: dict[str, Metric] = {}
    for metric_name in metric_names:
        data = raw[metric_name]
        metric = Metric(data)
        metrics[metric_name] = metric
        _log.info(
            f"{metric}: mean={metric.mean:.6f}, median={metric.median:.6f}, min={metric.min:.6f}, max={metric.max:.6f}",
        )

    return JetsonBenchmarkResult(
        latency=metrics["latency"],
        power_draw=metrics["power_draw"],
        energy=metrics["energy"],
    )
