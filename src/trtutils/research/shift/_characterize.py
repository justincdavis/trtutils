# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Offline characterization of (model, accelerator) pairs for SHIFT."""

from __future__ import annotations

import gc
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2

from trtutils._benchmark import benchmark_engine
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.compat._libs import cudart
from trtutils.core._cuda import cuda_call
from trtutils.jetson._benchmark import benchmark_engine as jetson_benchmark_engine

from ._types import ModelTraits

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence

    import numpy as np

    from trtutils._engine import TRTEngine
    from trtutils.image.interfaces import DetectorInterface
    from trtutils.image.types import Bbox


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


def _read_image(path: Path | str) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        err_msg = f"Cannot read image: {path}"
        raise FileNotFoundError(err_msg)
    return image


def _prefetch(
    images: Iterable[np.ndarray | Path | str],
    depth: int,
    workers: int,
) -> Generator[np.ndarray, None, None]:
    """
    Yield images in order as BGR arrays, decoding paths ahead in background threads.

    Up to ``depth`` upcoming images are queued (decoding or decoded) while the
    caller works on the current one, so at most ``depth + 2`` decoded frames are
    alive: the queue, the frame in use and the one it replaces. Arrays pass
    through untouched. Decoding starts on the first ``next`` and the threads are
    joined when the generator ends or is closed, so close it (e.g. with
    ``contextlib.closing``) when stopping early.

    Parameters
    ----------
    images : Iterable[np.ndarray | Path | str]
        The images, as BGR arrays or paths.
    depth : int
        How many upcoming images to queue, at least 1.
    workers : int
        How many threads decode, at least 1.

    Yields
    ------
    np.ndarray
        Each image, in input order.

    Raises
    ------
    FileNotFoundError
        If an image path cannot be read; raised when that image is reached.

    """
    # cv2.imread releases the gil, so the threads overlap the caller's work
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shift-prefetch") as pool:

        def _submit(image: np.ndarray | Path | str) -> Future[np.ndarray] | np.ndarray:
            return pool.submit(_read_image, image) if isinstance(image, (Path, str)) else image

        items = iter(images)
        pending = deque(_submit(image) for image in islice(items, depth))
        try:
            while pending:
                item = pending.popleft()
                frame = item.result() if isinstance(item, Future) else item
                pending.extend(_submit(image) for image in islice(items, 1))
                yield frame
        finally:
            # python 3.8 has no shutdown(cancel_futures=True)
            for item in pending:
                if isinstance(item, Future):
                    item.cancel()


def _used_memory_mb() -> float:
    # unified memory on jetson, so this also sees host allocations
    free, total = cuda_call(cudart.cudaMemGetInfo())
    return (total - free) / (1024 * 1024)


def _engine_memory_mb(engine: TRTEngine) -> float:
    # activation memory plus i/o bindings: a floor that excludes weights
    bindings = [*engine.input_bindings, *engine.output_bindings]
    return (engine.memsize + sum(b.nbytes for b in bindings)) / (1024 * 1024)


def _measure(
    engine: TRTEngine,
    iterations: int,
    warmup_iterations: int,
    *,
    verbose: bool | None,
) -> tuple[float, float]:
    # returns (latency ms, energy mj); energy needs tegrastats
    if FLAGS.IS_JETSON:
        jresult = jetson_benchmark_engine(
            engine=engine,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            warmup=True,
            verbose=verbose,
        )
        return jresult.latency.mean * 1000.0, jresult.energy.mean
    result = benchmark_engine(
        engine=engine,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        warmup=True,
        verbose=verbose,
    )
    return result.latency.mean * 1000.0, 0.0


def characterize(
    models: Mapping[str, Callable[[], DetectorInterface]],
    images: Iterable[np.ndarray | Path | str],
    ground_truth: Iterable[Sequence[Bbox]],
    iterations: int = 1000,
    warmup_iterations: int = 50,
    *,
    prefetch: int = 4,
    decode_workers: int = 2,
    verbose: bool | None = None,
) -> dict[str, ModelTraits]:
    """
    Characterize (model, accelerator) pairs for SHIFT.

    Each pair is built, run over the labeled images, benchmarked, and freed
    before the next one, so only one pair is in memory at a time. Every pair needs
    every image, so images given as paths are streamed: during each pair's image
    pass, background threads decode the next ``prefetch`` images while the pair
    runs on the current one, and each frame is dropped after use. Memory holds a
    few frames instead of the whole set, at the cost of one decode per image per
    pair, which overlaps inference. The threads are stopped before the pair is
    benchmarked, so decoding never runs inside a measurement. Per image it records
    every detection's confidence and, for every ground-truth box, the best IoU of
    any detection (class agnostic). Latency comes from the engine alone (no
    pre/postprocessing); energy is measured with tegrastats on Jetson and is 0.0
    elsewhere.

    Parameters
    ----------
    models : Mapping[str, Callable[[], DetectorInterface]]
        A factory per pair name that builds the detector, e.g.
        ``{"yolov10n_dla": lambda: Detector("yolov10n_dla.engine", dla_core=0, cuda_graph=False)}``.
    images : Iterable[np.ndarray | Path | str]
        The labeled images, as BGR arrays or paths. Any iterable works, a generator
        included: it is read once and its items kept, which is cheap for paths
        (decoded per pair, as above) while arrays stay in memory throughout.
    ground_truth : Iterable[Sequence[Bbox]]
        Per image, the ground-truth boxes as (x1, y1, x2, y2) pixels. Any iterable
        works and is read once.
    iterations : int, optional
        Benchmark iterations per pair. Default is 1000.
    warmup_iterations : int, optional
        Benchmark warmup iterations per pair. Default is 50.
    prefetch : int, optional
        How many upcoming images to decode ahead of the pair, at least 1. At most
        ``prefetch + 2`` decoded frames are alive. Default is 4.
    decode_workers : int, optional
        How many threads decode images given as paths, at least 1. Default is 2.
    verbose : bool, optional
        Whether to log progress. Default is None (False).

    Returns
    -------
    dict[str, ModelTraits]
        The traits of each pair, keyed by pair name.

    Raises
    ------
    ValueError
        If no models or no images are given, images and ground truth differ in
        length, or ``prefetch`` or ``decode_workers`` is below 1.
    FileNotFoundError
        If an image path cannot be read; raised while the first pair runs.

    """
    if len(models) == 0:
        err_msg = "characterize needs at least one model"
        raise ValueError(err_msg)
    if prefetch < 1 or decode_workers < 1:
        err_msg = f"prefetch and decode_workers must be >= 1, got {prefetch}, {decode_workers}"
        raise ValueError(err_msg)
    # every pair walks the images again, so keep the items (paths, not decoded frames)
    images = list(images)
    ground_truth = list(ground_truth)
    if len(images) == 0:
        err_msg = "characterize needs at least one image"
        raise ValueError(err_msg)
    if len(images) != len(ground_truth):
        err_msg = f"got {len(images)} images but {len(ground_truth)} ground-truth entries"
        raise ValueError(err_msg)

    # one throwaway build so one-time cuda/tensorrt setup is not billed to the first pair
    next(iter(models.values()))()
    gc.collect()

    traits: dict[str, ModelTraits] = {}
    for name, factory in models.items():
        used = _used_memory_mb()
        t0 = time.perf_counter()
        detector = factory()
        load_time = time.perf_counter() - t0
        # the system-wide delta is noisy (unified memory on jetson), so never go below
        # what the engine itself allocates
        load_memory = max(_used_memory_mb() - used, _engine_memory_mb(detector.engine))

        confidences: list[list[float]] = []
        ious: list[list[float]] = []
        # closing joins the decode threads before measuring, even if the pass raises
        with closing(_prefetch(images, prefetch, decode_workers)) as frames:
            for frame, boxes in zip(frames, ground_truth):
                detections = detector.end2end(frame)
                confidences.append([float(d[1]) for d in detections])
                ious.append(
                    [max((_iou(box, d[0]) for d in detections), default=0.0) for box in boxes]
                )

        latency, energy = _measure(detector.engine, iterations, warmup_iterations, verbose=verbose)
        traits[name] = ModelTraits(
            latency_ms=latency,
            energy_mj=energy,
            load_time_s=load_time,
            load_memory_mb=load_memory,
            confidences=confidences,
            ious=ious,
        )
        if verbose:
            LOG.info(
                f"SHIFT characterize {name}: {latency:.2f} ms, {energy:.2f} mJ, "
                f"load {load_time:.2f} s / {load_memory:.1f} MB"
            )
        del detector
        gc.collect()
    return traits
