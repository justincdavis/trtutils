# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CLI subcommands for the DRIFT research module."""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2

from trtutils._log import LOG
from trtutils.image import Detector

from ._characterize import characterize
from ._drift import Drift
from ._types import DriftConfig, load_characterization, save_characterization

if TYPE_CHECKING:
    import argparse
    from types import SimpleNamespace

    import numpy as np

    from trtutils.image.interfaces import DetectorInterface
    from trtutils.image.types import Bbox

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_MODEL_SPEC_LENGTHS = (2, 3)
_LABEL_FIELDS = 5
_METRICS = ["mean", "median", "min", "max"]
_CONFIG_CHOICES = {
    "counter_decrement": ["all", "empty"],
    "dense_groups": ["rectangles", "blocks"],
    "anneal_counter": ["frames", "packs"],
    "confidence_metric": _METRICS,
    "box_size_metric": _METRICS,
    "accuracy_scale_metric": _METRICS,
    "latency_metric": _METRICS,
    "lp_latency_test": ["sum", "schedule"],
}


def _parse_models(
    specs: list[list[str]],
    input_ranges: list[list[str]] | None = None,
) -> dict[str, Callable[[], DetectorInterface]]:
    """
    Turn ``--model NAME ENGINE [DLA_CORE]`` specs into detector factories.

    Parameters
    ----------
    specs : list[list[str]]
        One ``[name, engine]`` or ``[name, engine, dla_core]`` list per model.
    input_ranges : list[list[str]], optional
        ``[name, low, high]`` lists overriding a model's input range, e.g. YOLOX
        models expect ``0 255``. Default is None.

    Returns
    -------
    dict[str, Callable[[], DetectorInterface]]
        A factory per model name.

    Raises
    ------
    ValueError
        If a spec has the wrong number of values, a name repeats, or an input range
        names an unknown model.

    """
    ranges: dict[str, tuple[float, float]] = {}
    for spec in input_ranges or []:
        name, low, high = spec
        ranges[name] = (float(low), float(high))
    models: dict[str, Callable[[], DetectorInterface]] = {}
    for spec in specs:
        if len(spec) not in _MODEL_SPEC_LENGTHS:
            err_msg = f"--model takes NAME ENGINE [DLA_CORE], got {spec}"
            raise ValueError(err_msg)
        name, engine = spec[0], Path(spec[1])
        if name in models:
            err_msg = f"--model name {name!r} is used more than once"
            raise ValueError(err_msg)
        dla_core = int(spec[2]) if len(spec) == max(_MODEL_SPEC_LENGTHS) else None
        kwargs: dict[str, Any] = {}
        if name in ranges:
            kwargs["input_range"] = ranges[name]
        # dla engines cannot use cuda graphs
        models[name] = partial(
            Detector,
            engine,
            dla_core=dla_core,
            cuda_graph=False if dla_core is not None else None,
            warmup=True,
            **kwargs,
        )
    unknown = sorted(set(ranges) - set(models))
    if unknown:
        err_msg = f"--input_range names unknown models {unknown}"
        raise ValueError(err_msg)
    return models


def _read_labels(path: Path, width: int, height: int) -> list[tuple[Bbox, int]]:
    """
    Read a YOLO-format label file as pixel boxes with class ids.

    Parameters
    ----------
    path : Path
        The ``.txt`` file with ``class cx cy w h`` lines, normalized to [0, 1].
        A missing file means no objects.
    width : int
        The image width.
    height : int
        The image height.

    Returns
    -------
    list[tuple[Bbox, int]]
        The (box, class id) pairs, boxes as (x1, y1, x2, y2) pixels.

    Raises
    ------
    ValueError
        If a line does not have exactly five fields.

    """
    if not path.exists():
        return []
    labels: list[tuple[Bbox, int]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        values = line.split()
        if len(values) == 0:
            continue
        if len(values) != _LABEL_FIELDS:
            err_msg = f"{path}:{number}: expected 'class cx cy w h', got {line!r}"
            raise ValueError(err_msg)
        cx, cy, w, h = (float(v) for v in values[1:])
        box = (
            round((cx - w / 2) * width),
            round((cy - h / 2) * height),
            round((cx + w / 2) * width),
            round((cy + h / 2) * height),
        )
        labels.append((box, int(values[0])))
    return labels


def _read_frames(source: Path, limit: int | None = None) -> list[np.ndarray]:
    """
    Read an image, or up to ``limit`` frames of a video.

    Parameters
    ----------
    source : Path
        An image or video file.
    limit : int, optional
        The maximum number of video frames. Default is None (all).

    Returns
    -------
    list[np.ndarray]
        The frames, BGR.

    Raises
    ------
    FileNotFoundError
        If nothing can be read.

    """
    frames: list[np.ndarray] = []
    if source.suffix.lower() in _IMAGE_SUFFIXES:
        image = cv2.imread(str(source))
        if image is not None:
            frames.append(image)
    else:
        cap = cv2.VideoCapture(str(source))
        while limit is None or len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    if len(frames) == 0:
        err_msg = f"Cannot read any frame from {source}"
        raise FileNotFoundError(err_msg)
    return frames


def _characterize(args: SimpleNamespace) -> None:
    """
    Characterize detection models on a labeled image directory.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:

        General (inherited):
        - log_level : str
            Logging level. Default is "INFO".
        - verbose : bool
            Enable verbose output.
        - nvtx : bool
            Enable NVTX markers.

        DRIFT-specific:
        - model : list[list[str]]
            One ``NAME ENGINE [DLA_CORE]`` list per model.
        - input_range : list[list[str]] | None
            ``NAME LOW HIGH`` input-range overrides.
        - images : str
            Directory of images.
        - labels : str
            Directory of YOLO-format ``.txt`` labels, one per image stem.
        - latency_input : str | None
            Video or image the latency loops use. None uses the labeled images.
        - latency_frames : int
            Frames read from ``latency_input``.
        - output : str
            JSON file to write the characterization to.
        - iterations : int
            Timed calls per model and per pair.
        - warmup_iterations : int
            Untimed calls before each timing loop.
        - iou_threshold : float
            IoU for a detection to match a label.

    Raises
    ------
    FileNotFoundError
        If the image directory holds no images.

    """
    models = _parse_models(args.model, args.input_range)
    image_paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(image_paths) == 0:
        err_msg = f"No images found in {args.images}"
        raise FileNotFoundError(err_msg)
    images: list[np.ndarray] = []
    ground_truth: list[list[tuple[Bbox, int]]] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            err_msg = f"Cannot read image: {path}"
            raise FileNotFoundError(err_msg)
        images.append(image)
        height, width = image.shape[:2]
        ground_truth.append(_read_labels(Path(args.labels) / f"{path.stem}.txt", width, height))
    latency_images = None
    if args.latency_input is not None:
        latency_images = _read_frames(Path(args.latency_input), args.latency_frames)

    traits = characterize(
        models,
        images,
        ground_truth,
        latency_images=latency_images,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        iou_threshold=args.iou_threshold,
        verbose=args.verbose,
    )
    save_characterization(traits, args.output)

    LOG.info("DRIFT Characterization Complete")
    LOG.info("=" * 60)
    for name, trait in traits.items():
        f1 = trait.frame_f1()
        energy = sum(trait.energy_mj) / len(trait.energy_mj) if trait.energy_mj else 0.0
        accel = "DLA" if trait.dla else "GPU"
        LOG.info(
            f"{name:<24} {accel} {trait.latency():8.2f} ms {energy:9.2f} mJ "
            f"mean F1 {sum(f1) / len(f1):.3f}"
        )
        for lp in trait.pair_latency_ms:
            LOG.info(f"  + {lp:<20} {trait.pair_latency(lp):8.2f} ms (concurrent)")
    LOG.info(f"Saved to: {args.output}")
    LOG.info("=" * 60)


def _make_config(args: SimpleNamespace) -> DriftConfig:
    """
    Build the config from a preset and per-field overrides.

    Parameters
    ----------
    args : SimpleNamespace
        Arguments with ``preset`` and one attribute per ``DriftConfig`` field, None
        when not given.

    Returns
    -------
    DriftConfig
        The preset with the given fields replaced.

    """
    config = DriftConfig.upstream() if args.preset == "upstream" else DriftConfig()
    overrides = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(DriftConfig)
        if getattr(args, f.name) is not None
    }
    return dataclasses.replace(config, **overrides)


def _run(args: SimpleNamespace) -> None:
    """
    Run DRIFT over a video or image and report which schedules ran.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:

        General (inherited):
        - log_level : str
            Logging level. Default is "INFO".
        - verbose : bool
            Enable verbose output.
        - nvtx : bool
            Enable NVTX markers.

        DRIFT-specific:
        - model : list[list[str]]
            One ``NAME ENGINE [DLA_CORE]`` list per model.
        - input_range : list[list[str]] | None
            ``NAME LOW HIGH`` input-range overrides.
        - characterization : str
            JSON file written by ``characterize``.
        - input : str
            A video or image file.
        - priority : list[int]
            The high-priority class ids.
        - baseline : str | None
            The baseline model, None for the most accurate GPU model.
        - seed : int | None
            Seed of the frame packer's sampler.
        - preset : str
            "paper" or "upstream".
        - one attribute per ``DriftConfig`` field, None when not overridden.

    Raises
    ------
    FileNotFoundError
        If the input cannot be read.

    """
    source = Path(args.input)
    if not source.exists():
        err_msg = f"Cannot find input: {source}"
        raise FileNotFoundError(err_msg)
    drift = Drift(
        _parse_models(args.model, args.input_range),
        load_characterization(args.characterization),
        args.priority,
        baseline=args.baseline,
        config=_make_config(args),
        seed=args.seed,
        verbose=args.verbose,
    )

    usage: Counter[str] = Counter()
    elapsed = 0.0
    baseline_frames = 0
    parallel_frames = 0

    def _step(frame: np.ndarray) -> None:
        nonlocal elapsed, baseline_frames, parallel_frames
        t0 = time.perf_counter()
        drift.end2end(frame)
        elapsed += time.perf_counter() - t0
        last = drift.last
        if last is None:
            return
        schedule = last.schedule
        if schedule.lp is None:
            baseline_frames += 1
            usage[f"{schedule.hp} (baseline)"] += 1
        else:
            parallel_frames += int(schedule.parallel)
            usage[f"{schedule.hp} + {schedule.lp}"] += 1

    if source.suffix.lower() in _IMAGE_SUFFIXES:
        _step(_read_frames(source)[0])
    else:
        cap = cv2.VideoCapture(str(source))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _step(frame)
        cap.release()
    drift.close()

    frames = sum(usage.values())
    if frames == 0:
        err_msg = f"No frames read from {source}"
        raise FileNotFoundError(err_msg)
    LOG.info("DRIFT Run Complete")
    LOG.info("=" * 60)
    LOG.info(f"Frames           : {frames}")
    LOG.info(f"Mean frame time  : {elapsed / frames * 1000.0:.2f} ms")
    LOG.info(f"Baseline frames  : {baseline_frames / frames:.1%}")
    LOG.info(f"Split frames     : {(frames - baseline_frames) / frames:.1%}")
    LOG.info(f"Parallel frames  : {parallel_frames / frames:.1%}")
    for name, count in usage.most_common():
        LOG.info(f"{name:<40} {count:6d} frames ({count / frames:.1%})")
    LOG.info("=" * 60)


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    """
    Add one override flag per ``DriftConfig`` field.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ``run`` parser.

    """
    for field in dataclasses.fields(DriftConfig):
        name = field.name
        help_text = f"Override {name} of the preset."
        if name in _CONFIG_CHOICES:
            parser.add_argument(
                f"--{name}", choices=_CONFIG_CHOICES[name], default=None, help=help_text
            )
        elif field.type == "bool":
            parser.add_argument(
                f"--{name}", dest=name, action="store_const", const=True, default=None,
                help=help_text,
            )  # fmt: skip
            parser.add_argument(
                f"--no_{name}", dest=name, action="store_const", const=False, default=None,
                help=f"Turn {name} off.",
            )  # fmt: skip
        elif str(field.type).startswith("tuple"):
            parser.add_argument(
                f"--{name}",
                nargs="+",
                type=str,
                default=None,
                help=help_text,
                metavar="NAME",
            )
        else:
            parser.add_argument(
                f"--{name}",
                type=int if field.type == "int" else float,
                default=None,
                help=help_text,
            )


def register_cli(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """
    Register DRIFT CLI subcommands under the research parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action from the research CLI parser.
    parents : list[argparse.ArgumentParser]
        Parent parsers; only the first (general arguments) is used, since DRIFT
        does not build engines.

    """
    drift_parser = subparsers.add_parser(
        "drift",
        help="DRIFT: intra-frame parallel object detection across GPU and DLA.",
    )
    drift_subparsers = drift_parser.add_subparsers(
        title="drift commands",
        dest="drift_command",
        required=False,
    )
    drift_parser.set_defaults(func=lambda _args: drift_parser.print_help())

    model_help = (
        "A detection model as NAME ENGINE [DLA_CORE]. Repeat for every model. "
        "Give DLA_CORE for engines that run on a DLA."
    )
    range_help = (
        "Input range of one model as NAME LOW HIGH, e.g. '0 255' for YOLOX. "
        "Repeatable. Default is each detector's default (0 1)."
    )

    characterize_parser = drift_subparsers.add_parser(
        "characterize",
        help="Characterize detection models on labeled images.",
        parents=parents[:1],
    )
    characterize_parser.add_argument(
        "--model", action="append", nargs="+", required=True, help=model_help
    )
    characterize_parser.add_argument(
        "--input_range", action="append", nargs=3, default=None, help=range_help
    )
    characterize_parser.add_argument(
        "--images", required=True, help="Directory of images (.jpg, .jpeg, .png, .bmp)."
    )
    characterize_parser.add_argument(
        "--labels",
        required=True,
        help=(
            "Directory of YOLO-format labels (class cx cy w h, normalized), one .txt "
            "per image stem, class ids in the models' id space. A missing file means "
            "the image has no objects."
        ),
    )
    characterize_parser.add_argument(
        "--output", required=True, help="JSON file to write the characterization to."
    )
    characterize_parser.add_argument(
        "--latency_input",
        default=None,
        help="Video or image the latency loops use. Default uses the labeled images.",
    )
    characterize_parser.add_argument(
        "--latency_frames",
        type=int,
        default=30,
        help="Frames read from --latency_input. Default is 30.",
    )
    characterize_parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Timed calls per model and per GPU+DLA pair. Default is 200.",
    )
    characterize_parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=20,
        help="Untimed calls before each timing loop. Default is 20.",
    )
    characterize_parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU for a detection to match a label. Default is 0.5.",
    )
    characterize_parser.set_defaults(func=_characterize)

    run_parser = drift_subparsers.add_parser(
        "run",
        help="Run DRIFT over a video or image.",
        parents=parents[:1],
    )
    run_parser.add_argument("--model", action="append", nargs="+", required=True, help=model_help)
    run_parser.add_argument("--input_range", action="append", nargs=3, default=None, help=range_help)
    run_parser.add_argument(
        "--characterization",
        required=True,
        help="JSON file written by 'drift characterize'.",
    )
    run_parser.add_argument("--input", required=True, help="A video or image file.")
    run_parser.add_argument(
        "--priority",
        type=int,
        nargs="+",
        required=True,
        metavar="CLASS_ID",
        help="The high-priority class ids, e.g. 0 for COCO person.",
    )
    run_parser.add_argument(
        "--baseline",
        default=None,
        help="The baseline GPU model. Default is the GPU model with the highest mean F1.",
    )
    run_parser.add_argument(
        "--seed", type=int, default=None, help="Seed of the frame packer's sampler."
    )
    run_parser.add_argument(
        "--preset",
        choices=["paper", "upstream"],
        default="paper",
        help=(
            "Hyperparameter preset: the DRIFT paper's values or the reference "
            "implementation's. Flags below override single values. Default is paper."
        ),
    )
    _add_config_flags(run_parser)
    run_parser.set_defaults(func=_run)
