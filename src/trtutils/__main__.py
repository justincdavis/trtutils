# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: PLC0415
"""Main entry point for trtutils CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from trtutils import set_log_level

if TYPE_CHECKING:
    from types import SimpleNamespace

_log = logging.getLogger("trtutils")


def _benchmark(args: SimpleNamespace) -> None:
    mpath = Path(args.model)
    if not mpath.exists():
        err_msg = f"Cannot find provided model: {mpath}"
        raise FileNotFoundError(err_msg)

    set_log_level("INFO")

    if args.jetson:
        from trtutils.jetson import benchmark_engine as benchmark_jetson

        j_result = benchmark_jetson(
            mpath,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            tegra_interval=1,
            warmup=True,
        )
        latency = j_result.latency
        energy = j_result.energy
        power = j_result.power_draw
        latency_in_ms = {
            "mean": latency.mean * 1000.0,
            "median": latency.median * 1000.0,
            "min": latency.min * 1000.0,
            "max": latency.max * 1000.0,
        }
        energy_in_joules = {
            "mean": energy.mean / 1000.0,
            "median": energy.median / 1000.0,
            "min": energy.min / 1000.0,
            "max": energy.max / 1000.0,
        }
        power_in_watts = {
            "mean": power.mean / 1000.0,
            "median": power.median / 1000.0,
            "min": power.min / 1000.0,
            "max": power.max / 1000.0,
        }

        _log.info(f"Benchmarking result for: {mpath.stem}")
        _log.info("=" * 40)
        _log.info("Latency (ms):")
        _log.info(f"  Mean   : {latency_in_ms['mean']:.2f}")
        _log.info(f"  Median : {latency_in_ms['median']:.2f}")
        _log.info(f"  Min    : {latency_in_ms['min']:.2f}")
        _log.info(f"  Max    : {latency_in_ms['max']:.2f}")
        _log.info("")
        _log.info("Energy Consumption (J):")
        _log.info(f"  Mean   : {energy_in_joules['mean']:.3f}")
        _log.info(f"  Median : {energy_in_joules['median']:.3f}")
        _log.info(f"  Min    : {energy_in_joules['min']:.3f}")
        _log.info(f"  Max    : {energy_in_joules['max']:.3f}")
        _log.info("")
        _log.info("Power Draw (W):")
        _log.info(f"  Mean   : {power_in_watts['mean']:.3f}")
        _log.info(f"  Median : {power_in_watts['median']:.3f}")
        _log.info(f"  Min    : {power_in_watts['min']:.3f}")
        _log.info(f"  Max    : {power_in_watts['max']:.3f}")
        _log.info("=" * 40)

    else:
        from trtutils import benchmark_engine

        result = benchmark_engine(
            mpath,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            warmup=True,
        )
        latency = result.latency
        latency_in_ms = {
            "mean": latency.mean * 1000.0,
            "median": latency.median * 1000.0,
            "min": latency.min * 1000.0,
            "max": latency.max * 1000.0,
        }

        _log.info(f"Benchmarking result for: {mpath.stem}")
        _log.info("=" * 40)
        _log.info("Latency (ms):")
        _log.info(f"  Mean   : {latency_in_ms['mean']:.2f}")
        _log.info(f"  Median : {latency_in_ms['median']:.2f}")
        _log.info(f"  Min    : {latency_in_ms['min']:.2f}")
        _log.info(f"  Max    : {latency_in_ms['max']:.2f}")
        _log.info("=" * 40)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Utilities for TensorRT.")

    # create subparser for each command
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        required=True,
    )

    # benchmark command, will use benchmark engine by default, can pass jetson to use the jetson benchmarker
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark a given TensorRT engine.",
    )
    benchmark_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to the engine file.",
    )
    benchmark_parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=1000,
        help="Number of iterations to measure over.",
    )
    benchmark_parser.add_argument(
        "--warmup_iterations",
        "-wi",
        type=int,
        default=100,
        help="The number of iterations to warmup the model before measuring.",
    )
    benchmark_parser.add_argument(
        "--jetson",
        "-j",
        action="store_true",
        help="If True, will use the trtutils.jetson submodule benchmarker to record energy and pwoerdraw as well.",
    )
    benchmark_parser.set_defaults(func=_benchmark)

    # parse args and call the function
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
