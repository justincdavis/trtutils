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

import trtutils
from trtutils.trtexec._cli import cli_trtexec

if TYPE_CHECKING:
    from types import SimpleNamespace

_log = logging.getLogger("trtutils")


def _benchmark(args: SimpleNamespace) -> None:
    mpath = Path(args.model)
    if not mpath.exists():
        err_msg = f"Cannot find provided model: {mpath}"
        raise FileNotFoundError(err_msg)

    if args.jetson:
        j_result = trtutils.jetson.benchmark_engine(
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
        result = trtutils.benchmark_engine(
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


def _build(args: SimpleNamespace) -> None:
    trtutils.build_engine(
        onnx=Path(args.onnx),
        output=Path(args.output),
        timing_cache=Path(args.timing_cache),
        log_level=args.log_level,
        workspace=args.workspace,
        dla_core=args.dla_core,
        gpu_fallback=args.gpu_fallback,
        direct_io=args.direct_io,
        prefer_precision_constraints=args.prefer_precision_constraints,
        reject_empty_algorithms=args.reject_empty_algorithms,
        ignore_timing_mismatch=args.ignore_timing_mismatch,
        fp16=args.fp16,
    )


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

    # trtexec parser
    trtexec_parser = subparsers.add_parser(
        "trtexec",
        help="Run trtexec.",
    )
    trtexec_parser.set_defaults(func=cli_trtexec)

    # build_engine parser
    build_parser = subparsers.add_parser(
        "build",
        help="Build a TensorRT engine from an ONNX model.",
    )
    build_parser.add_argument(
        "--onnx",
        "-o",
        required=True,
        help="Path to the ONNX model file.",
    )
    build_parser.add_argument(
        "--output",
        "-out",
        required=True,
        help="Path to save the TensorRT engine file.",
    )
    build_parser.add_argument(
        "--timing_cache",
        "-tc",
        default="timing.cache",
        help="Path to store timing cache data. Default is 'timing.cache'.",
    )
    build_parser.add_argument(
        "--log_level",
        "-ll",
        type=int,
        help="Log level to use if the logger is None. Default is WARNING.",
    )
    build_parser.add_argument(
        "--workspace",
        "-w",
        type=float,
        default=4.0,
        help="Workspace size in GB. Default is 4.0.",
    )
    build_parser.add_argument(
        "--dla_core",
        type=int,
        help="Specify the DLA core. By default, the engine is built for GPU.",
    )
    build_parser.add_argument(
        "--gpu_fallback",
        action="store_true",
        help="Allow GPU fallback for unsupported layers when building for DLA.",
    )
    build_parser.add_argument(
        "--direct_io",
        action="store_true",
        help="Use direct IO for the engine.",
    )
    build_parser.add_argument(
        "--prefer_precision_constraints",
        action="store_true",
        help="Prefer precision constraints.",
    )
    build_parser.add_argument(
        "--reject_empty_algorithms",
        action="store_true",
        help="Reject empty algorithms.",
    )
    build_parser.add_argument(
        "--ignore_timing_mismatch",
        action="store_true",
        help="Allow different CUDA device timing caches to be used.",
    )
    build_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Quantize the engine to FP16 precision.",
    )
    build_parser.set_defaults(func=_build)

    # parse args and call the function
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    trtutils.set_log_level("INFO")
    _main()
