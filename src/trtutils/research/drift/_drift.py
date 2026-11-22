# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""DRIFT runtime: split each frame and run the high- and low-priority models concurrently."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Callable

import numpy as np

from trtutils._log import LOG
from trtutils.image.postprocessors._common import nms

from ._characterize import _WARMUP_CALLS, _run_locked
from ._packer import FramePacker
from ._scheduler import DriftScheduler, hp_region
from ._types import DriftConfig, FrameResult

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from concurrent.futures import Future

    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface
    from trtutils.image.types import Bbox, Detection

    from ._types import ModelTraits, Schedule


def _square(region: Bbox, image_size: tuple[int, int]) -> Bbox:
    # grow the short side by the difference, half on each side, clipped to the frame
    x1, y1, x2, y2 = region
    width, height = image_size
    diff = (x2 - x1) - (y2 - y1)
    if diff > 0:
        y1, y2 = max(0, y1 - diff // 2), min(height, y2 + diff - diff // 2)
    elif diff < 0:
        x1, x2 = max(0, x1 + diff // 2), min(width, x2 - diff + diff // 2)
    return x1, y1, x2, y2


class Drift:
    """
    Intra-frame parallel object detection across a GPU and a DLA.

    Each frame is split into the high-priority region (the padded box around the
    previous frame's priority-class detections), which a GPU model detects on a crop,
    and the rest of the frame, which :class:`FramePacker` packs into a small image
    for a second model. When that model runs on a DLA it runs concurrently on a
    worker thread; otherwise it runs after the first on the GPU.
    :class:`DriftScheduler` picks both models per frame and falls back to the
    baseline model on the full frame when splitting is not expected to pay off.

    Every model is built up front and stays resident. Detectors run on the thread
    that calls :meth:`end2end`, except DLA models, which always run on one worker
    thread. Call :meth:`close` when done.
    """

    def __init__(
        self: Self,
        models: Mapping[str, Callable[[], DetectorInterface]],
        traits: Mapping[str, ModelTraits],
        priority_classes: Collection[int],
        baseline: str | None = None,
        config: DriftConfig | None = None,
        *,
        seed: int | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create the DRIFT runtime.

        Parameters
        ----------
        models : Mapping[str, Callable[[], DetectorInterface]]
            A factory per model name that builds the detector, e.g.
            ``{"yolox_s_dla": lambda: Detector("s.engine", dla_core=0, cuda_graph=False)}``.
        traits : Mapping[str, ModelTraits]
            The characterization of the same models, from :func:`characterize`.
        priority_classes : Collection[int]
            The high-priority class ids.
        baseline : str, optional
            The GPU model that runs full frames. Default is None: the GPU model with
            the highest mean characterized F1.
        config : DriftConfig, optional
            The hyperparameters. Default is the paper's values.
        seed : int, optional
            Seed of the frame packer's sampler. Default is None (random).
        verbose : bool, optional
            Whether to log schedule changes. Default is None (False).

        Raises
        ------
        ValueError
            If ``models`` and ``traits`` name different models, there is no priority
            class, or a detector's input size or accelerator differs from its traits.

        """
        if set(models) != set(traits):
            err_msg = (
                f"models {sorted(models)} and traits {sorted(traits)} must name the same models"
            )
            raise ValueError(err_msg)
        if len(priority_classes) == 0:
            err_msg = "DRIFT needs at least one priority class"
            raise ValueError(err_msg)
        self._config = config if config is not None else DriftConfig()
        self._scheduler = DriftScheduler(traits, baseline, self._config)
        self._priority = frozenset(priority_classes)
        self._seed = seed
        self._verbose = verbose
        self._packer: FramePacker | None = None
        self._last: FrameResult | None = None

        self._detectors: dict[str, DetectorInterface] = {}
        self._dla = {name: trait.dla for name, trait in traits.items()}
        self._executor: ThreadPoolExecutor | None = None
        ready = False
        try:
            for name, factory in models.items():
                detector = factory()
                self._detectors[name] = detector
                dla = detector.engine.dla_core is not None
                if (
                    tuple(detector.input_shape) != tuple(traits[name].input_size)
                    or dla != traits[name].dla
                ):
                    err_msg = (
                        f"model {name!r} has input {tuple(detector.input_shape)} and dla={dla}, "
                        f"but its traits have input {tuple(traits[name].input_size)} and "
                        f"dla={traits[name].dla}"
                    )
                    raise ValueError(err_msg)
            if any(self._dla.values()):
                self._executor = ThreadPoolExecutor(max_workers=1)

            # run each model a few times so every CUDA graph is captured before any
            # concurrent execution
            for name, detector in self._detectors.items():
                width, height = detector.input_shape
                dummy = np.zeros((height, width, 3), dtype=np.uint8)
                for _ in range(_WARMUP_CALLS):
                    self._run(name, dummy)
            ready = True
        finally:
            # a failed construction releases the worker and every model built so far
            if not ready:
                self.close()

    @property
    def config(self: Self) -> DriftConfig:
        """The hyperparameters."""
        return self._config

    @property
    def scheduler(self: Self) -> DriftScheduler:
        """The scheduler."""
        return self._scheduler

    @property
    def baseline(self: Self) -> str:
        """The baseline model."""
        return self._scheduler.baseline.hp

    @property
    def last(self: Self) -> FrameResult | None:
        """What ran the last frame, None before the first frame or after a reset."""
        return self._last

    def end2end(self: Self, image: np.ndarray) -> list[Detection]:
        """
        Detect objects in one frame.

        The high-priority region, the schedule and the split all come from the
        previous frame's detections, so they always agree within a frame.

        Parameters
        ----------
        image : np.ndarray
            The frame, BGR.

        Returns
        -------
        list[Detection]
            The detections in frame coordinates.

        """
        if not self._detectors:
            err_msg = "DRIFT runtime is closed"
            raise RuntimeError(err_msg)
        height, width = image.shape[:2]
        size = (width, height)
        if self._packer is None or self._packer.image_size != size:
            cfg = self._config
            self._packer = FramePacker(
                size,
                grid_divisor=cfg.grid_divisor,
                annealing_rate=cfg.annealing_rate,
                annealing_floor=cfg.annealing_floor,
                fps=cfg.fps,
                counter_decrement=cfg.counter_decrement,
                dense_groups=cfg.dense_groups,
                anneal_counter=cfg.anneal_counter,
                seed=self._seed,
            )
            self._scheduler.reset()
            self._last = None
        previous = self._last
        region = None
        if previous is not None:
            region = hp_region(
                previous.detections,
                self._priority,
                size,
                self._config.hp_padding,
                min_area=self._config.hp_min_area,
            )
        schedule = self._scheduler(previous, region, size)
        if self._verbose and (previous is None or previous.schedule[:2] != schedule[:2]):
            LOG.info(f"DRIFT: schedule hp={schedule.hp} lp={schedule.lp}")
        lp = schedule.lp
        if lp is None or region is None:
            detections = list(self._detectors[schedule.hp].end2end(image))
            result = FrameResult(detections, detections, [], schedule)
        else:
            result = self._split(image, region, schedule, lp, self._packer)
        self._packer.update([d[0] for d in result.detections])
        self._last = result
        return result.detections

    def reset(self: Self) -> None:
        """Forget the previous frames (e.g. at a new video); models stay loaded."""
        self._scheduler.reset()
        self._packer = None
        self._last = None

    def close(self: Self) -> None:
        """Stop the worker thread and release every model."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._detectors.clear()

    def _split(
        self: Self,
        image: np.ndarray,
        region: Bbox,
        schedule: Schedule,
        lp_name: str,
        packer: FramePacker,
    ) -> FrameResult:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = _square(region, (width, height)) if self._config.hp_square else region
        packed = packer.pack(image, exclude=region)

        future: Future[list[Detection]] | None = None
        if packed is not None and schedule.parallel:
            future = self._submit(lp_name, packed[0])
        try:
            # crop after the submit so the copy does not delay the dla
            hp_raw = self._detectors[schedule.hp].end2end(np.ascontiguousarray(image[y1:y2, x1:x2]))
        finally:
            # never leave the worker running into the next frame, even if hp failed; an hp
            # error stays the one raised
            if future is not None:
                wait([future])
        hp = [
            ((b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1), float(conf), int(cls))
            for b, conf, cls in hp_raw
        ]
        if packed is None:
            # no low-priority cell was sampled: only the hp model ran, and the next
            # frame's prediction must come from it
            return FrameResult(hp, hp, [], schedule._replace(lp=None, parallel=False))

        if future is not None:
            lp_raw = future.result()
        else:
            lp_raw = self._detectors[lp_name].end2end(packed[0])
        lp: list[Detection] = []
        if len(lp_raw) > 0:
            boxes, valid = packer.unpack([d[0] for d in lp_raw], packed[1])
            lp = [
                ((int(box[0]), int(box[1]), int(box[2]), int(box[3])), float(conf), int(cls))
                for box, ok, (_, conf, cls) in zip(boxes, valid, lp_raw)
                if ok
            ]
        detections = nms(hp + lp, self._config.merge_iou) if lp else hp
        return FrameResult(detections, hp, lp, schedule)

    def _run(self: Self, name: str, image: np.ndarray) -> list[Detection]:
        if self._dla[name]:
            return self._submit(name, image).result()
        return self._detectors[name].end2end(image)

    def _submit(self: Self, name: str, image: np.ndarray) -> Future[list[Detection]]:
        if self._executor is None:
            err_msg = f"model {name!r} runs on a DLA but the worker is gone (was close() called?)"
            raise RuntimeError(err_msg)
        return self._executor.submit(_run_locked, self._detectors[name], image)
