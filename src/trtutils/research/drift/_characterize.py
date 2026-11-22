# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Offline characterization of detection models for DRIFT."""

from __future__ import annotations

import gc
import tempfile
import time
from collections.abc import Sized
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
from jetsontools import TegraStats, filter_data, get_powerdraw, parse_tegrastats

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._types import ModelTraits

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import numpy as np

    from trtutils.image.interfaces import DetectorInterface
    from trtutils.image.types import Bbox, Detection

_TEGRA_INTERVAL_MS = 5
_WARMUP_CALLS = 3
_CONFIDENCE_DECIMALS = 4


def _iou(box1: Bbox, box2: Bbox) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def _true_positives(
    detections: Sequence[Detection],
    ground_truth: Sequence[tuple[Bbox, int]],
    iou_threshold: float,
) -> int:
    # greedy, highest confidence first: each detection takes the best unmatched
    # ground-truth box of its class
    unmatched = list(range(len(ground_truth)))
    matches = 0
    for box, _, cls in sorted(detections, key=lambda d: -d[1]):
        best, best_iou = -1, 0.0
        for index in unmatched:
            gt_box, gt_cls = ground_truth[index]
            if gt_cls != cls:
                continue
            iou = _iou(box, gt_box)
            if iou > best_iou:
                best, best_iou = index, iou
        if best >= 0 and best_iou >= iou_threshold:
            unmatched.remove(best)
            matches += 1
    return matches


def _load_image(image: np.ndarray | Path | str) -> np.ndarray:
    if not isinstance(image, (Path, str)):
        return image
    frame = cv2.imread(str(image))
    if frame is None:
        err_msg = f"Cannot read image: {image}"
        raise FileNotFoundError(err_msg)
    return frame


def _labeled_images(
    images: Iterable[np.ndarray | Path | str],
    ground_truth: Iterable[Sequence[tuple[Bbox, int]]],
) -> Iterator[tuple[np.ndarray, Sequence[tuple[Bbox, int]]]]:
    # decodes one image at a time; generators have no length, so their mismatch is
    # only found when one side runs out
    sized = isinstance(images, Sized) and isinstance(ground_truth, Sized)
    if sized and len(images) != len(ground_truth):
        err_msg = f"got {len(images)} images but {len(ground_truth)} ground-truth entries"
        raise ValueError(err_msg)
    truth = iter(ground_truth)
    for image in images:
        boxes = next(truth, None)
        if boxes is None:
            err_msg = "ground_truth has fewer entries than images"
            raise ValueError(err_msg)
        yield _load_image(image), boxes
    if next(truth, None) is not None:
        err_msg = "ground_truth has more entries than images"
        raise ValueError(err_msg)


def _measure(
    step: Callable[[int], object],
    iterations: int,
    warmup_iterations: int,
) -> tuple[list[float], list[float]]:
    # returns raw latency (ms) and energy (mJ) samples; energy needs tegrastats
    for index in range(warmup_iterations):
        step(index)
    stamps: list[tuple[float, float]] = []
    if not FLAGS.IS_JETSON:
        for index in range(iterations):
            t0 = time.perf_counter()
            step(index)
            stamps.append((t0, time.perf_counter()))
        return [(t1 - t0) * 1000.0 for t0, t1 in stamps], []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tegrastats.txt"
        with TegraStats(path, interval=_TEGRA_INTERVAL_MS):
            for index in range(iterations):
                t0 = time.time()
                step(index)
                stamps.append((t0, time.time()))
        data = parse_tegrastats(path)
    _, per_call = filter_data(data, stamps)
    energy = [
        get_powerdraw(samples)["VDD_TOTAL"].mean * (t1 - t0)
        for (t0, t1), samples in per_call
        if len(samples) > 0
    ]
    return [(t1 - t0) * 1000.0 for t0, t1 in stamps], energy


def _run_locked(detector: DetectorInterface, image: np.ndarray) -> list[Detection]:
    # every dla call runs on the worker (here and in the runtime) under this lock: work
    # on the legacy default stream breaks a CUDA graph capture on the caller thread;
    # only captures contend for it, and the warmup calls do every capture up front
    with TRTEngine._capture_lock:  # noqa: SLF001
        return detector.end2end(image)


def _pair_step(
    executor: ThreadPoolExecutor,
    gpu: DetectorInterface,
    accel: DetectorInterface,
    frames: Sequence[np.ndarray],
    index: int,
) -> None:
    # the runtime's concurrent schedule: DLA model on the worker, GPU model here
    frame = frames[index % len(frames)]
    future = executor.submit(_run_locked, accel, frame)
    try:
        gpu.end2end(frame)
    finally:
        future.result()


def characterize(
    models: Mapping[str, Callable[[], DetectorInterface]],
    images: Iterable[np.ndarray | Path | str],
    ground_truth: Iterable[Sequence[tuple[Bbox, int]]],
    latency_images: Iterable[np.ndarray | Path | str] | None = None,
    iterations: int = 200,
    warmup_iterations: int = 20,
    iou_threshold: float = 0.5,
    *,
    latency_frames: int = 30,
    verbose: bool | None = None,
) -> dict[str, ModelTraits]:
    """
    Characterize detection models for DRIFT.

    Every model is built once and stays resident, as at runtime. The labeled images
    are streamed: each is decoded once and run through every model before the next
    one is read, so one labeled image is in memory at a time. Per labeled image it
    records every detection's confidence, the number of true positives (greedy,
    class-aware, at ``iou_threshold``) and the number of ground-truth boxes. Latency
    is end to end (``end2end``: preprocess, inference, postprocess), per model and
    for every GPU model running concurrently with every DLA model, exactly as the
    runtime runs them, cycling through a small set of frames held in memory. Energy
    is measured with tegrastats on Jetson.

    Parameters
    ----------
    models : Mapping[str, Callable[[], DetectorInterface]]
        A factory per model name that builds the detector, e.g.
        ``{"yolox_s_dla": lambda: Detector("s.engine", dla_core=0, cuda_graph=False)}``.
        Use the same factories at runtime.
    images : Iterable[np.ndarray | Path | str]
        The labeled images, as BGR arrays or paths, e.g. a list or a generator.
        Paths are decoded one at a time, when their turn comes.
    ground_truth : Iterable[Sequence[tuple[Bbox, int]]]
        Per image, the ground-truth (box, class id) pairs, boxes as (x1, y1, x2, y2)
        pixels and class ids in the models' output id space. Same length as
        ``images``.
    latency_images : Iterable[np.ndarray | Path | str], optional
        The images the latency loops cycle through, e.g. frames of the target video.
        All of them are decoded up front and held in memory, so keep the set small.
        Default is None: the first ``latency_frames`` labeled images.
    iterations : int, optional
        Timed calls per model and per pair. Default is 200.
    warmup_iterations : int, optional
        Untimed calls before each timing loop. Default is 20.
    iou_threshold : float, optional
        IoU for a detection to match a ground-truth box. Default is 0.5.
    latency_frames : int, optional
        How many labeled images the latency loops cycle through when
        ``latency_images`` is None. Default is 30.
    verbose : bool, optional
        Whether to log progress. Default is None (False).

    Returns
    -------
    dict[str, ModelTraits]
        The traits of each model, keyed by model name.

    Raises
    ------
    ValueError
        If no models or images are given, images and ground truth differ in length,
        ``latency_images`` is empty, or ``latency_frames`` is below one. When either
        of images and ground truth has no length, a mismatch is raised once the
        shorter one runs out.
    FileNotFoundError
        If an image path cannot be read.

    """
    return _characterize_labeled(
        models,
        _labeled_images(images, ground_truth),
        latency_images,
        iterations,
        warmup_iterations,
        iou_threshold,
        latency_frames=latency_frames,
        verbose=verbose,
    )


def _characterize_labeled(
    models: Mapping[str, Callable[[], DetectorInterface]],
    labeled: Iterator[tuple[np.ndarray, Sequence[tuple[Bbox, int]]]],
    latency_images: Iterable[np.ndarray | Path | str] | None,
    iterations: int,
    warmup_iterations: int,
    iou_threshold: float,
    *,
    latency_frames: int,
    verbose: bool | None,
) -> dict[str, ModelTraits]:
    # characterize over decoded (image, ground truth) pairs, pulled one at a time;
    # the cli builds its own pairs, since labels are scaled by their image's size
    if len(models) == 0:
        err_msg = "characterize needs at least one model"
        raise ValueError(err_msg)
    if latency_frames < 1:
        err_msg = f"latency_frames must be at least 1, got {latency_frames}"
        raise ValueError(err_msg)
    first = next(labeled, None)
    if first is None:
        err_msg = "characterize needs at least one labeled image"
        raise ValueError(err_msg)
    timing_frames: list[np.ndarray] = []
    if latency_images is not None:
        timing_frames = [_load_image(image) for image in latency_images]
        if len(timing_frames) == 0:
            err_msg = "latency_images is empty"
            raise ValueError(err_msg)

    detectors = {name: factory() for name, factory in models.items()}
    dla = {name: d.engine.dla_core is not None for name, d in detectors.items()}
    executor = ThreadPoolExecutor(max_workers=1) if any(dla.values()) else None

    def _run(name: str, image: np.ndarray) -> list[Detection]:
        if executor is not None and dla[name]:
            return executor.submit(_run_locked, detectors[name], image).result()
        return detectors[name].end2end(image)

    def _solo(name: str, index: int) -> None:
        _run(name, timing_frames[index % len(timing_frames)])

    traits: dict[str, ModelTraits] = {}
    try:
        for name in detectors:
            for _ in range(_WARMUP_CALLS):
                _run(name, first[0])

        # image outer, model inner: each labeled image is decoded once and dropped
        # after every model ran on it, except the few kept for the latency loops
        confidences: dict[str, list[list[float]]] = {name: [] for name in detectors}
        true_positives: dict[str, list[int]] = {name: [] for name in detectors}
        ground_truths: list[int] = []
        for frame, boxes in chain([first], labeled):
            for name in detectors:
                detections = _run(name, frame)
                confidences[name].append(
                    [round(float(d[1]), _CONFIDENCE_DECIMALS) for d in detections]
                )
                true_positives[name].append(_true_positives(detections, boxes, iou_threshold))
            ground_truths.append(len(boxes))
            if latency_images is None and len(timing_frames) < latency_frames:
                timing_frames.append(frame)

        for name, detector in detectors.items():
            latency, energy = _measure(partial(_solo, name), iterations, warmup_iterations)
            width, height = detector.input_shape
            traits[name] = ModelTraits(
                input_size=(int(width), int(height)),
                dla=dla[name],
                latency_ms=latency,
                confidences=confidences[name],
                true_positives=true_positives[name],
                ground_truths=list(ground_truths),
                energy_mj=energy,
            )
            if verbose:
                LOG.info(f"DRIFT characterize {name}: {traits[name].latency():.2f} ms")

        pairs = [(g, a) for g in detectors for a in detectors if not dla[g] and dla[a]]
        if executor is not None:
            for gpu, accel in pairs:
                step = partial(_pair_step, executor, detectors[gpu], detectors[accel], timing_frames)
                latency, energy = _measure(step, iterations, warmup_iterations)
                traits[gpu].pair_latency_ms[accel] = latency
                traits[gpu].pair_energy_mj[accel] = energy
                if verbose:
                    LOG.info(
                        f"DRIFT characterize {gpu} + {accel}: "
                        f"{traits[gpu].pair_latency(accel):.2f} ms"
                    )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        detectors.clear()
        gc.collect()
    return traits
