# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CLI subcommands for the SHIFT research module."""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2

from trtutils._log import LOG
from trtutils.image import Detector

from ._characterize import _read_image, characterize
from ._shift import Shift
from ._types import (
    _REDUCERS,
    _SIMILARITIES,
    ShiftConfig,
    load_characterization,
    save_characterization,
)

if TYPE_CHECKING:
    import argparse
    from types import SimpleNamespace

    import numpy as np

    from trtutils.image.interfaces import DetectorInterface
    from trtutils.image.types import Bbox

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_MODEL_SPEC_LENGTHS = (2, 3)
_LABEL_FIELDS = 5
_CONFIG_CHOICES = {
    "similarity": list(_SIMILARITIES),
    "confidence_metric": list(_REDUCERS),
    "accuracy_metric": list(_REDUCERS),
}


def _parse_models(specs: list[list[str]]) -> dict[str, Callable[[], DetectorInterface]]:
    """
    Turn ``--model NAME ENGINE [DLA_CORE]`` specs into detector factories.

    Parameters
    ----------
    specs : list[list[str]]
        One ``[name, engine]`` or ``[name, engine, dla_core]`` list per pair.

    Returns
    -------
    dict[str, Callable[[], DetectorInterface]]
        A factory per pair name.

    Raises
    ------
    ValueError
        If a spec has the wrong number of values or a name repeats.

    """
    models: dict[str, Callable[[], DetectorInterface]] = {}
    for spec in specs:
        if len(spec) not in _MODEL_SPEC_LENGTHS:
            err_msg = f"--model takes NAME ENGINE [DLA_CORE], got {spec}"
            raise ValueError(err_msg)
        name, engine, *rest = spec
        if name in models:
            err_msg = f"--model name {name!r} is used more than once"
            raise ValueError(err_msg)
        dla_core = int(rest[0]) if rest else None
        # dla engines cannot use cuda graphs
        models[name] = partial(
            Detector,
            Path(engine),
            dla_core=dla_core,
            cuda_graph=False if dla_core is not None else None,
            warmup=True,
        )
    return models


def _read_labels(path: Path, width: int, height: int) -> list[Bbox]:
    """
    Read a YOLO-format label file as pixel boxes.

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
    list[Bbox]
        The boxes as (x1, y1, x2, y2) pixels.

    Raises
    ------
    ValueError
        If a line does not have exactly five fields.

    """
    if not path.exists():
        return []
    boxes: list[Bbox] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        values = line.split()
        if len(values) == 0:
            continue
        if len(values) != _LABEL_FIELDS:
            err_msg = f"{path}:{number}: expected 'class cx cy w h', got {line!r}"
            raise ValueError(err_msg)
        cx, cy, w, h = (float(v) for v in values[1:])
        boxes.append(
            (
                round((cx - w / 2) * width),
                round((cy - h / 2) * height),
                round((cx + w / 2) * width),
                round((cy + h / 2) * height),
            )
        )
    return boxes


def _characterize(args: SimpleNamespace) -> None:
    """
    Characterize (model, accelerator) pairs on a labeled image directory.

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

        SHIFT-specific:
        - model : list[list[str]]
            One ``NAME ENGINE [DLA_CORE]`` list per pair.
        - images : str
            Directory of images.
        - labels : str
            Directory of YOLO-format ``.txt`` labels, one per image stem.
        - output : str
            JSON file to write the characterization to.
        - iterations : int
            Benchmark iterations per pair.
        - warmup_iterations : int
            Benchmark warmup iterations per pair.

    Raises
    ------
    FileNotFoundError
        If the image directory holds no images.

    """
    models = _parse_models(args.model)
    image_paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(image_paths) == 0:
        err_msg = f"No images found in {args.images}"
        raise FileNotFoundError(err_msg)
    images: list[np.ndarray] = []
    ground_truth: list[list[Bbox]] = []
    for path in image_paths:
        image = _read_image(path)
        images.append(image)
        height, width = image.shape[:2]
        ground_truth.append(_read_labels(Path(args.labels) / f"{path.stem}.txt", width, height))

    traits = characterize(
        models,
        images,
        ground_truth,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        verbose=args.verbose,
    )
    save_characterization(traits, args.output)

    LOG.info("SHIFT Characterization Complete")
    LOG.info("=" * 60)
    for name, trait in traits.items():
        LOG.info(
            f"{name:<24} {trait.latency_ms:8.2f} ms {trait.energy_mj:9.2f} mJ "
            f"mean IoU {sum(trait.frame_accuracies()) / len(images):.3f}"
        )
    LOG.info(f"Saved to: {args.output}")
    LOG.info("=" * 60)


def _make_config(args: SimpleNamespace) -> ShiftConfig:
    """
    Build the scheduler config from a preset and per-field overrides.

    Parameters
    ----------
    args : SimpleNamespace
        Arguments with ``preset`` and one attribute per ``ShiftConfig`` field,
        None when not given.

    Returns
    -------
    ShiftConfig
        The preset with the given fields replaced.

    """
    config = ShiftConfig.shiftdetector() if args.preset == "shiftdetector" else ShiftConfig()
    overrides = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(ShiftConfig)
        if getattr(args, f.name) is not None
    }
    return dataclasses.replace(config, **overrides)


def _run(args: SimpleNamespace) -> None:
    """
    Run SHIFT over a video or image and report which pairs ran.

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

        SHIFT-specific:
        - model : list[list[str]]
            One ``NAME ENGINE [DLA_CORE]`` list per pair.
        - characterization : str
            JSON file written by ``characterize``.
        - input : str
            A video or image file.
        - preset : str
            "paper" or "shiftdetector".
        - memory_budget : float | None
            Memory budget in MB, or None to load every pair up front.
        - one attribute per ``ShiftConfig`` field, None when not overridden.

    Raises
    ------
    FileNotFoundError
        If the input cannot be read.

    """
    source = Path(args.input)
    if not source.exists():
        err_msg = f"Cannot find input: {source}"
        raise FileNotFoundError(err_msg)
    shift = Shift(
        _parse_models(args.model),
        load_characterization(args.characterization),
        _make_config(args),
        memory_budget=args.memory_budget,
        verbose=args.verbose,
    )

    usage: Counter[str] = Counter()
    swaps = 0
    elapsed = 0.0

    def _step(frame: np.ndarray) -> None:
        nonlocal swaps, elapsed
        previous = shift.last
        t0 = time.perf_counter()
        shift.end2end(frame)
        elapsed += time.perf_counter() - t0
        ran = shift.last
        if ran is not None:
            usage[ran] += 1
            swaps += int(previous is not None and ran != previous)

    if source.suffix.lower() in _IMAGE_SUFFIXES:
        _step(_read_image(source))
    else:
        cap = cv2.VideoCapture(str(source))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _step(frame)
        cap.release()
    shift.close()

    frames = sum(usage.values())
    if frames == 0:
        err_msg = f"No frames read from {source}"
        raise FileNotFoundError(err_msg)
    LOG.info("SHIFT Run Complete")
    LOG.info("=" * 60)
    LOG.info(f"Frames           : {frames}")
    LOG.info(f"Mean frame time  : {elapsed / frames * 1000.0:.2f} ms")
    LOG.info(f"Swaps            : {swaps}")
    for name, count in usage.most_common():
        LOG.info(f"{name:<24} {count:6d} frames ({count / frames:.1%})")
    LOG.info("=" * 60)


def register_cli(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """
    Register SHIFT CLI subcommands under the research parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action from the research CLI parser.
    parents : list[argparse.ArgumentParser]
        Parent parsers; only the first (general arguments) is used, since SHIFT
        does not build engines.

    """
    shift_parser = subparsers.add_parser(
        "shift",
        help="SHIFT: context-aware multi-model object detection scheduling.",
    )
    shift_subparsers = shift_parser.add_subparsers(
        title="shift commands",
        dest="shift_command",
        required=False,
    )
    shift_parser.set_defaults(func=lambda _args: shift_parser.print_help())

    model_help = (
        "A (model, accelerator) pair as NAME ENGINE [DLA_CORE]. "
        "Repeat for every pair. Give DLA_CORE for engines with DLA layers."
    )

    characterize_parser = shift_subparsers.add_parser(
        "characterize",
        help="Characterize (model, accelerator) pairs on labeled images.",
        parents=parents[:1],
    )
    characterize_parser.add_argument(
        "--model", action="append", nargs="+", required=True, help=model_help
    )
    characterize_parser.add_argument(
        "--images", required=True, help="Directory of images (.jpg, .jpeg, .png, .bmp)."
    )
    characterize_parser.add_argument(
        "--labels",
        required=True,
        help=(
            "Directory of YOLO-format labels (class cx cy w h, normalized), one .txt "
            "per image stem. A missing file means the image has no objects."
        ),
    )
    characterize_parser.add_argument(
        "--output", required=True, help="JSON file to write the characterization to."
    )
    characterize_parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Benchmark iterations per pair. Default is 1000.",
    )
    characterize_parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=50,
        help="Benchmark warmup iterations per pair. Default is 50.",
    )
    characterize_parser.set_defaults(func=_characterize)

    run_parser = shift_subparsers.add_parser(
        "run",
        help="Run SHIFT over a video or image.",
        parents=parents[:1],
    )
    run_parser.add_argument("--model", action="append", nargs="+", required=True, help=model_help)
    run_parser.add_argument(
        "--characterization",
        required=True,
        help="JSON file written by 'shift characterize'.",
    )
    run_parser.add_argument("--input", required=True, help="A video or image file.")
    run_parser.add_argument(
        "--preset",
        choices=["paper", "shiftdetector"],
        default="paper",
        help=(
            "Hyperparameter preset: the SHIFT paper's values or the shiftdetector "
            "reference implementation's. Flags below override single values. "
            "Default is paper."
        ),
    )
    run_parser.add_argument(
        "--memory_budget",
        type=float,
        default=None,
        help=(
            "Memory budget in MB for loaded pairs. Pairs then load on demand in the "
            "background and the least recently used are evicted. "
            "Default loads every pair up front."
        ),
    )
    for field in dataclasses.fields(ShiftConfig):
        run_parser.add_argument(
            f"--{field.name}",
            choices=_CONFIG_CHOICES.get(field.name),
            type=None if field.name in _CONFIG_CHOICES else int if field.type == "int" else float,
            default=None,
            help=f"Override {field.name} of the preset.",
        )
    run_parser.set_defaults(func=_run)
