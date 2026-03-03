# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Timing utilities for benchmark scripts."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tqdm import tqdm

from trtutils import Metric

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def benchmark_loop(
    fn: Callable[[], Any],
    warmup_iters: int,
    bench_iters: int,
    desc: str = "",
) -> list[float]:
    """
    Run warmup iterations, then time bench_iters calls to fn.

    Parameters
    ----------
    fn : Callable[[], Any]
        The function to benchmark. Called with no arguments.
    warmup_iters : int
        Number of warmup calls (not timed).
    bench_iters : int
        Number of timed calls.
    desc : str, optional
        Description for the tqdm progress bar.

    Returns
    -------
    list[float]
        Raw wall-clock timings in seconds.

    """
    for _ in range(warmup_iters):
        fn()

    timings: list[float] = []
    for _ in tqdm(range(bench_iters), desc=desc):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)

    return timings


def compute_results(
    timings: list[float],
    batch_size: int = 1,
) -> dict[str, float]:
    """
    Compute statistics from raw timings using trtutils.Metric.

    Parameters
    ----------
    timings : list[float]
        Raw wall-clock timings in seconds.
    batch_size : int, optional
        Batch size for throughput calculation, by default 1.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: mean, median, min, max, std, ci95 (all ms)
        and throughput (images/sec).

    """
    metric = Metric(timings)
    mean_s = metric.mean
    throughput = batch_size / mean_s if mean_s > 0 else 0.0

    return {
        "mean": metric.mean * 1000.0,
        "median": metric.median * 1000.0,
        "min": metric.min * 1000.0,
        "max": metric.max * 1000.0,
        "std": metric.std * 1000.0,
        "ci95": metric.ci95 * 1000.0,
        "throughput": throughput,
    }
