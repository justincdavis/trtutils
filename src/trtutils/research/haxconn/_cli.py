# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CLI subcommands for the HaX-CoNN research module."""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from trtutils._log import LOG

from ._build import assignment_code, prebuild_engines
from ._characterize import calibrate, characterize
from ._haxconn import HaxConn
from ._pccs import PCCSModel
from ._scheduler import HaxConnScheduler
from ._types import (
    _CONTENTIONS,
    _OBJECTIVES,
    _SOLVERS,
    HaxConnConfig,
    Schedule,
    load_calibration,
    load_schedule,
    load_traits,
    save_calibration,
    save_schedule,
    save_traits,
)

if TYPE_CHECKING:
    from types import SimpleNamespace

    from ._types import ModelTraits

_METRICS = ("mean", "median", "min", "max")
_CONFIG_CHOICES = {
    "objective": _OBJECTIVES,
    "contention": _CONTENTIONS,
    "solver": _SOLVERS,
    "latency_metric": _METRICS,
    "memory_metric": _METRICS,
}


def _optional_int(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def _optional_float(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def _dla_core(args: SimpleNamespace) -> int:
    return args.dla_core if args.dla_core is not None else 0


def _make_config(args: SimpleNamespace) -> HaxConnConfig:
    """
    Build the config from a preset and per-field overrides.

    Parameters
    ----------
    args : SimpleNamespace
        Arguments with ``preset`` and an attribute for each ``HaxConnConfig`` field that
        was given.

    Returns
    -------
    HaxConnConfig
        The preset with the given fields replaced.

    """
    config = HaxConnConfig.upstream() if args.preset == "upstream" else HaxConnConfig()
    overrides = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(HaxConnConfig)
        if hasattr(args, f.name)
    }
    if "update_ms" in overrides:
        values = overrides["update_ms"]
        none = len(values) == 1 and values[0].lower() == "none"
        overrides["update_ms"] = None if none else tuple(float(v) for v in values)
    return dataclasses.replace(config, **overrides)


def _contention(path: str | None) -> dict[str, PCCSModel] | None:
    if path is None:
        return None
    return {acc: PCCSModel.from_calibration(c) for acc, c in load_calibration(path).items()}


def _log_schedule(label: str, schedule: Schedule) -> None:
    LOG.info(f"{label}: objective {schedule.objective:.3f}, optimal={schedule.optimal}")
    for name, accs in schedule.assignments.items():
        LOG.info(f"  {name:<20} {assignment_code(accs)}  {schedule.latency_ms[name]:8.3f} ms")


def _characterize(args: SimpleNamespace) -> None:
    """
    Characterize the layer groups of one DNN and save its traits.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments: the general ones (``log_level``, ``verbose``, ``nvtx``),
        and ``onnx``, ``output``, ``max_groups``, ``dla_core``, ``iterations``,
        ``warmup_iterations``, ``workspace``, ``timing_cache``, ``int8``, ``memory``,
        ``memory_seconds``.

    """
    traits = characterize(
        args.onnx,
        max_groups=args.max_groups,
        dla_core=_dla_core(args),
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        workspace=args.workspace,
        timing_cache=args.timing_cache,
        int8=args.int8,
        memory=args.memory,
        memory_seconds=args.memory_seconds,
        verbose=args.verbose,
    )
    save_traits(traits, args.output)
    LOG.info("HaX-CoNN Characterization Complete")
    LOG.info("=" * 60)
    for group in traits.groups:
        times = "  ".join(f"{acc} {group.time(acc):.3f} ms" for acc in group.time_ms)
        LOG.info(f"layers {group.first:>4}-{group.last:<4} {times}")
    LOG.info(f"DLA launch cost: {traits.dla_launch_ms:.3f} ms")
    LOG.info(f"Saved to: {args.output}")


def _calibrate(args: SimpleNamespace) -> None:
    """
    Calibrate the GPU/DLA contention and save it.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments: the general ones, and ``output``, ``dla_core``,
        ``sizes``, ``kernels``, ``seconds``, ``int8``.

    """
    calibration = calibrate(
        dla_core=_dla_core(args),
        sizes=args.sizes,
        kernels=args.kernels,
        seconds=args.seconds,
        int8=args.int8,
        verbose=args.verbose,
    )
    save_calibration(calibration, args.output)
    LOG.info("HaX-CoNN Calibration Complete")
    for acc, c in calibration.items():
        LOG.info(f"{acc}: {PCCSModel.from_calibration(c)}")
    LOG.info(f"Saved to: {args.output}")


def _solve(args: SimpleNamespace) -> None:
    """
    Solve for a schedule and compare it with the naive baselines.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments: the general ones, ``model`` (``NAME TRAITS`` pairs),
        ``calibration``, ``output``, ``preset`` and one attribute per config field.

    """
    traits = {name: load_traits(path) for name, path in args.model}
    scheduler = HaxConnScheduler(traits, _contention(args.calibration), _make_config(args))
    schedule = scheduler.solve()
    for label, baseline in scheduler.baselines().items():
        _log_schedule(label, baseline)
    _log_schedule("HaX-CoNN", schedule)
    if args.output is not None:
        save_schedule(schedule, args.output)
        LOG.info(f"Saved to: {args.output}")


def _measure(runtime: HaxConn, iterations: int, warmup_iterations: int) -> dict[str, float]:
    for _ in range(warmup_iterations):
        runtime.mock_execute()
    samples: dict[str, list[float]] = {}
    for _ in range(iterations):
        runtime.mock_execute()
        for name, ms in (runtime.last if runtime.last is not None else {}).items():
            samples.setdefault(name, []).append(ms)
    return {name: statistics.mean(values) for name, values in samples.items()}


def _run(args: SimpleNamespace) -> None:
    """
    Run DNNs concurrently on a schedule and report predicted against measured latency.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments: the general ones, ``model`` (``NAME ONNX TRAITS``
        triples), ``calibration``, ``schedule``, ``engine_dir``, ``dla_core``,
        ``iterations``, ``warmup_iterations``, ``baselines``, ``dynamic``, ``preset`` and
        one attribute per config field.

    """
    models = {name: onnx for name, onnx, _ in args.model}
    traits: dict[str, ModelTraits] = {name: load_traits(path) for name, _, path in args.model}
    contention = _contention(args.calibration)
    config = _make_config(args)
    schedules: dict[str, Schedule | None] = {}
    if args.baselines:
        schedules.update(HaxConnScheduler(traits, contention, config).baselines())
    schedules["HaX-CoNN"] = load_schedule(args.schedule) if args.schedule is not None else None

    LOG.info("HaX-CoNN Run")
    LOG.info("=" * 60)
    # one engine directory for every schedule, so a DNN assignment shared by two is built once
    with tempfile.TemporaryDirectory() as tmp:
        engine_dir = args.engine_dir if args.engine_dir is not None else tmp
        for label, given in schedules.items():
            runtime = HaxConn(
                models,
                traits,
                contention,
                config,
                schedule=given,
                engine_dir=engine_dir,
                dla_core=_dla_core(args),
                dynamic=args.dynamic and given is None,
                verbose=args.verbose,
            )
            measured = _measure(runtime, args.iterations, args.warmup_iterations)
            schedule = runtime.schedule
            runtime.close()
            LOG.info(f"{label}: predicted max {max(schedule.latency_ms.values()):.3f} ms")
            for name, accs in schedule.assignments.items():
                LOG.info(
                    f"  {name:<20} {assignment_code(accs)}  "
                    f"predicted {schedule.latency_ms[name]:8.3f} ms  "
                    f"measured {measured[name]:8.3f} ms"
                )
            slowest = max(measured.values())
            LOG.info(f"  measured max {slowest:.3f} ms ({1000.0 / slowest:.1f} rounds/s)")
    LOG.info("=" * 60)


def _pool(args: SimpleNamespace) -> tuple[dict[str, str], dict[str, ModelTraits]]:
    models = {name: onnx for name, onnx, _ in args.model}
    traits = {name: load_traits(path) for name, _, path in args.model}
    return models, traits


def _prebuild(args: SimpleNamespace) -> None:
    """
    Build the engine of every assignment a solver can pick, for D-HaX-CoNN.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments: the general ones, ``model`` (``NAME ONNX TRAITS``
        triples), ``engine_dir``, ``dla_core``, ``max_transitions``, ``workspace`` and
        ``timing_cache``.

    """
    models, traits = _pool(args)
    built = prebuild_engines(
        models,
        traits,
        args.engine_dir,
        max_transitions=args.max_transitions,
        dla_core=_dla_core(args),
        workspace=args.workspace,
        timing_cache=args.timing_cache,
        verbose=args.verbose,
    )
    LOG.info("HaX-CoNN Prebuild Complete")
    for name, paths in built.items():
        LOG.info(f"  {name:<20} {len(paths)} engines")
    LOG.info(f"Saved to: {args.engine_dir}")


def _segment(rounds: list[dict], start: float) -> dict:
    # what one stretch of a single DNN set did: where it started, where it ended, how fast
    first, final = rounds[0]["codes"], rounds[-1]["codes"]
    reached = next(r["t"] for r in rounds if r["codes"] == final)
    on_final = [r for r in rounds if r["codes"] == final]
    searching = [r["max_ms"] for r in on_final if r["searching"]]
    settled = [r["max_ms"] for r in on_final if not r["searching"]]
    return {
        "workload": rounds[0]["workload"],
        "rounds": len(rounds),
        "first": first,
        "first_ms": statistics.median(r["max_ms"] for r in rounds if r["codes"] == first),
        "final": final,
        "final_ms": statistics.median(r["max_ms"] for r in on_final),
        "switch_s": reached - start,
        "optimal": rounds[-1]["optimal"],
        "searching_ms": statistics.median(searching) if searching else None,
        "settled_ms": statistics.median(settled) if settled else None,
    }


def _record(runtime: HaxConn, t: float, cycle: int, workload: str, *, searching: bool) -> dict:
    schedule = runtime.schedule
    latency = runtime.last if runtime.last is not None else {}
    return {
        "t": t,
        "cycle": cycle,
        "workload": workload,
        "codes": {name: assignment_code(accs) for name, accs in schedule.assignments.items()},
        "predicted": schedule.objective,
        "optimal": schedule.optimal,
        "searching": searching,
        "latency_ms": dict(latency),
        "max_ms": max(latency.values()),
    }


def _log_segment(seg: dict) -> None:
    LOG.info(
        f"cycle {seg['cycle']} {seg['workload']}: {seg['rounds']} rounds, optimal={seg['optimal']}"
    )
    LOG.info(f"  first {seg['first']}  {seg['first_ms']:8.3f} ms")
    LOG.info(f"  final {seg['final']}  {seg['final_ms']:8.3f} ms after {seg['switch_s']:.3f} s")
    if seg["searching_ms"] is not None and seg["settled_ms"] is not None:
        overhead = 100.0 * (seg["searching_ms"] / seg["settled_ms"] - 1.0)
        LOG.info(
            f"  final schedule while searching {seg['searching_ms']:.3f} ms, "
            f"after {seg['settled_ms']:.3f} ms ({overhead:+.1f}%)"
        )


def _dynamic(args: SimpleNamespace) -> None:
    """
    Run D-HaX-CoNN on a changing set of DNNs and report how each set converges.

    Each workload (a set of DNNs) runs for ``period`` seconds, in order, ``cycles``
    times. Every round is recorded: the time, the schedule, whether the solver was
    still searching, and the measured latencies.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments: the general ones, ``model`` (``NAME ONNX TRAITS``
        triples), ``workload``, ``period``, ``cycles``, ``calibration``, ``engine_dir``,
        ``dla_core``, ``prebuild``, ``build``, ``solver_core``, ``output``, ``preset``
        and one attribute per config field.

    Raises
    ------
    ValueError
        If a workload names a DNN without a model, or ``prebuild`` is given without a
        transition limit.

    """
    models, traits = _pool(args)
    workloads = [[name.strip() for name in w.split(",") if name.strip()] for w in args.workload]
    for names in workloads:
        unknown = sorted(set(names) - set(models))
        if len(names) == 0 or unknown:
            err_msg = f"workload {names} must name DNNs from --model, unknown: {unknown}"
            raise ValueError(err_msg)
    config = _make_config(args)
    if args.prebuild and config.max_transitions is None:
        err_msg = "--prebuild needs a transition limit"
        raise ValueError(err_msg)
    rounds: list[dict] = []
    segments: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        engine_dir = args.engine_dir if args.engine_dir is not None else tmp
        if args.prebuild and config.max_transitions is not None:
            prebuild_engines(
                models,
                traits,
                engine_dir,
                max_transitions=config.max_transitions,
                dla_core=_dla_core(args),
                verbose=args.verbose,
            )
        runtime = HaxConn(
            models,
            traits,
            _contention(args.calibration),
            config,
            engine_dir=engine_dir,
            dla_core=_dla_core(args),
            active=workloads[0],
            dynamic=True,
            build=args.build,
            solver_core=args.solver_core,
            verbose=args.verbose,
        )
        t0 = time.monotonic()
        try:
            for cycle in range(args.cycles):
                for i, names in enumerate(workloads):
                    if cycle > 0 or i > 0:
                        runtime.set_active(names)
                    start = time.monotonic() - t0
                    segment: list[dict] = []
                    while time.monotonic() - t0 - start < args.period:
                        searching = runtime.searching
                        runtime.mock_execute()
                        t = time.monotonic() - t0
                        workload = ",".join(names)
                        segment.append(_record(runtime, t, cycle, workload, searching=searching))
                    rounds.extend(segment)
                    segments.append({"cycle": cycle, **_segment(segment, start)})
        finally:
            runtime.close()

    LOG.info("D-HaX-CoNN Run")
    LOG.info("=" * 60)
    for seg in segments:
        _log_segment(seg)
    LOG.info("=" * 60)
    if args.output is not None:
        with Path(args.output).open("w") as f:
            json.dump({"segments": segments, "rounds": rounds}, f, indent=2)
        LOG.info(f"Saved to: {args.output}")


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=["paper", "upstream"],
        default="paper",
        help=(
            "Hyperparameter preset: the HaX-CoNN paper's choices or the reference "
            "implementation's. Flags below override single values. Default is paper."
        ),
    )
    for field in dataclasses.fields(HaxConnConfig):
        name = field.name
        help_text = f"Override {name} of the preset."
        if name in _CONFIG_CHOICES:
            parser.add_argument(
                f"--{name}", choices=_CONFIG_CHOICES[name], default=argparse.SUPPRESS, help=help_text
            )
        elif name == "update_ms":
            parser.add_argument(
                f"--{name}",
                nargs="+",
                default=argparse.SUPPRESS,
                help=f"{help_text} Milliseconds; 'none': switch on every improvement.",
            )
        elif name == "max_transitions":
            parser.add_argument(
                f"--{name}",
                type=_optional_int,
                default=argparse.SUPPRESS,
                help=f"{help_text} 'none': no limit.",
            )
        else:
            parser.add_argument(
                f"--{name}",
                type=_optional_float,
                default=argparse.SUPPRESS,
                help=f"{help_text} 'none': search until proven optimal.",
            )


def register_cli(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """
    Register HaX-CoNN CLI subcommands under the research parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action from the research CLI parser.
    parents : list[argparse.ArgumentParser]
        Parent parsers; the first (general arguments) and second (``--dla_core``) are used.

    """
    haxconn_parser = subparsers.add_parser(
        "haxconn",
        help="HaX-CoNN: contention-aware concurrent DNN scheduling across GPU and DLA.",
    )
    haxconn_subparsers = haxconn_parser.add_subparsers(
        title="haxconn commands",
        dest="haxconn_command",
        required=False,
    )
    haxconn_parser.set_defaults(func=lambda _args: haxconn_parser.print_help())
    general_dla = parents[:2]

    characterize_parser = haxconn_subparsers.add_parser(
        "characterize",
        help="Characterize the layer groups of one ONNX model on the GPU and a DLA.",
        parents=general_dla,
    )
    characterize_parser.add_argument("--onnx", required=True, help="The ONNX model.")
    characterize_parser.add_argument(
        "--output", required=True, help="JSON file to write the traits to."
    )
    characterize_parser.add_argument(
        "--max_groups", type=int, default=None, help="Most layer groups to keep. Default all."
    )
    characterize_parser.add_argument(
        "--iterations", type=int, default=200, help="Timed runs per group engine. Default 200."
    )
    characterize_parser.add_argument(
        "--warmup_iterations", type=int, default=20, help="Untimed runs first. Default 20."
    )
    characterize_parser.add_argument(
        "--workspace", type=float, default=4.0, help="TensorRT workspace in GiB. Default 4.0."
    )
    characterize_parser.add_argument(
        "--timing_cache", default=None, help="Timing cache shared by the builds."
    )
    characterize_parser.add_argument(
        "--int8",
        action="store_true",
        help="Measure INT8 engines (the paper's Orin setting) instead of FP16.",
    )
    characterize_parser.add_argument(
        "--memory",
        action="store_true",
        help="Measure memory throughput with tegrastats (needs root or passwordless sudo).",
    )
    characterize_parser.add_argument(
        "--memory_seconds", type=float, default=1.5, help="Memory sampling per group. Default 1.5."
    )
    characterize_parser.set_defaults(func=_characterize)

    calibrate_parser = haxconn_subparsers.add_parser(
        "calibrate",
        help="Calibrate GPU/DLA memory contention (needs root or passwordless sudo).",
        parents=general_dla,
    )
    calibrate_parser.add_argument(
        "--output", required=True, help="JSON file to write the calibration to."
    )
    calibrate_parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128, 192, 256], help="Probe input sizes."
    )
    calibrate_parser.add_argument(
        "--kernels", type=int, nargs="+", default=[1, 3], help="Probe kernel sizes."
    )
    calibrate_parser.add_argument(
        "--seconds", type=float, default=1.5, help="Seconds per measurement. Default 1.5."
    )
    calibrate_parser.add_argument(
        "--int8", action="store_true", help="Run the probes in INT8 to match INT8 traits."
    )
    calibrate_parser.set_defaults(func=_calibrate)

    contention_help = "JSON file written by 'haxconn calibrate'. Needed unless --contention none."
    solve_parser = haxconn_subparsers.add_parser(
        "solve", help="Solve for a schedule from saved traits.", parents=parents[:1]
    )
    solve_parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        required=True,
        metavar=("NAME", "TRAITS"),
        help="A DNN and its traits JSON. Repeat for every DNN.",
    )
    solve_parser.add_argument("--calibration", default=None, help=contention_help)
    solve_parser.add_argument("--output", default=None, help="JSON file to write the schedule to.")
    _add_config_flags(solve_parser)
    solve_parser.set_defaults(func=_solve)

    run_parser = haxconn_subparsers.add_parser(
        "run",
        help="Run DNNs concurrently on a schedule and report the latencies.",
        parents=general_dla,
    )
    run_parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "ONNX", "TRAITS"),
        help="A DNN, its ONNX model and its traits JSON. Repeat for every DNN.",
    )
    run_parser.add_argument("--calibration", default=None, help=contention_help)
    run_parser.add_argument(
        "--schedule", default=None, help="Schedule JSON from 'haxconn solve'. Default: solve."
    )
    run_parser.add_argument(
        "--engine_dir", default=None, help="Directory that keeps the engines. Default temporary."
    )
    run_parser.add_argument("--iterations", type=int, default=200, help="Timed rounds. Default 200.")
    run_parser.add_argument(
        "--warmup_iterations", type=int, default=20, help="Untimed rounds first. Default 20."
    )
    run_parser.add_argument(
        "--baselines", action="store_true", help="Also run the GPU-only and naive schedules."
    )
    run_parser.add_argument(
        "--dynamic",
        action="store_true",
        help="D-HaX-CoNN: start on a naive schedule and switch as better ones are found.",
    )
    _add_config_flags(run_parser)
    run_parser.set_defaults(func=_run)

    model_help = "A DNN, its ONNX model and its traits JSON. Repeat for every DNN."
    prebuild_parser = haxconn_subparsers.add_parser(
        "prebuild",
        help="Build the engine of every assignment a solver can pick (for D-HaX-CoNN).",
        parents=general_dla,
    )
    prebuild_parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "ONNX", "TRAITS"),
        help=model_help,
    )
    prebuild_parser.add_argument(
        "--engine_dir", required=True, help="Directory for the engines (the runtime's)."
    )
    prebuild_parser.add_argument(
        "--max_transitions", type=int, default=1, help="Transition limit. Default 1."
    )
    prebuild_parser.add_argument(
        "--workspace", type=float, default=4.0, help="TensorRT workspace in GiB. Default 4.0."
    )
    prebuild_parser.add_argument(
        "--timing_cache", default=None, help="Timing cache shared by the builds."
    )
    prebuild_parser.set_defaults(func=_prebuild)

    dynamic_parser = haxconn_subparsers.add_parser(
        "dynamic",
        help="D-HaX-CoNN on a changing set of DNNs: report how each set converges.",
        parents=general_dla,
    )
    dynamic_parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "ONNX", "TRAITS"),
        help=model_help,
    )
    dynamic_parser.add_argument(
        "--workload",
        action="append",
        required=True,
        metavar="NAMES",
        help="Comma-separated DNNs that run together. Repeat; they run in order.",
    )
    dynamic_parser.add_argument(
        "--period", type=float, default=10.0, help="Seconds per workload. Default 10 (paper)."
    )
    dynamic_parser.add_argument(
        "--cycles", type=int, default=1, help="Times to run the workload sequence. Default 1."
    )
    dynamic_parser.add_argument("--calibration", default=None, help=contention_help)
    dynamic_parser.add_argument(
        "--engine_dir", default=None, help="Directory that keeps the engines. Default temporary."
    )
    dynamic_parser.add_argument(
        "--prebuild",
        action="store_true",
        help="Build every engine the solver can pick before running.",
    )
    dynamic_parser.add_argument(
        "--build",
        choices=["background", "prebuilt"],
        default="background",
        help=(
            "background: build a better schedule's engines while running; prebuilt: only "
            "switch to schedules whose engines exist. Default background."
        ),
    )
    dynamic_parser.add_argument(
        "--solver_core", type=int, default=None, help="CPU core to pin the solver to."
    )
    dynamic_parser.add_argument(
        "--output", default=None, help="JSON file to write every round and the summary to."
    )
    _add_config_flags(dynamic_parser)
    dynamic_parser.set_defaults(func=_dynamic)
