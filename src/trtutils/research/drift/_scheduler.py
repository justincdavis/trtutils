# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""DRIFT region identification (Section 3.1) and runtime scheduler (Algorithm 2)."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from trtutils.research.shift import reduce_values

from ._estimator import LinearEstimator
from ._types import DriftConfig, Schedule

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from typing_extensions import Self

    from trtutils.image.types import Bbox, Detection

    from ._types import FrameResult, ModelTraits


def hp_region(
    detections: Sequence[Detection],
    priority_classes: Collection[int],
    image_size: tuple[int, int],
    padding: int = 64,
    *,
    min_area: float = 0.0,
) -> Bbox | None:
    """
    Find the high-priority region: the padded box around all priority detections.

    Parameters
    ----------
    detections : Sequence[Detection]
        The previous frame's detections.
    priority_classes : Collection[int]
        The high-priority class ids.
    image_size : tuple[int, int]
        The frame size as (width, height).
    padding : int, optional
        Pixels added on every side. Default is 64.
    min_area : float, optional
        Priority boxes with an area at or below this are ignored. Default is 0.0.

    Returns
    -------
    Bbox | None
        The region as (x1, y1, x2, y2), clipped to the frame, or None when there is
        no priority detection.

    """
    boxes = [
        box
        for box, _, cls in detections
        if cls in priority_classes and (box[2] - box[0]) * (box[3] - box[1]) > min_area
    ]
    if len(boxes) == 0:
        return None
    width, height = image_size
    x1 = max(0, min(b[0] for b in boxes) - padding)
    y1 = max(0, min(b[1] for b in boxes) - padding)
    x2 = min(width, max(b[2] for b in boxes) + padding)
    y2 = min(height, max(b[3] for b in boxes) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)


class DriftScheduler:
    """
    Chooses the high- and low-priority models for the next frame (Algorithm 2).

    Works from characterization data alone, no engines: usable on its own to
    replay or simulate schedules.
    """

    def __init__(
        self: Self,
        traits: Mapping[str, ModelTraits],
        baseline: str | None = None,
        config: DriftConfig | None = None,
    ) -> None:
        """
        Create the scheduler.

        Parameters
        ----------
        traits : Mapping[str, ModelTraits]
            The characterization of every model, from :func:`characterize`.
        baseline : str, optional
            The GPU model that runs full frames when splitting does not pay off.
            Default is None: the GPU model with the highest mean characterized F1.
        config : DriftConfig, optional
            The hyperparameters. Default is the paper's values.

        Raises
        ------
        ValueError
            If there is no GPU model, or ``baseline`` is unknown or on a DLA.

        """
        self._config = config if config is not None else DriftConfig()
        self._traits = dict(traits)
        gpu = [name for name, trait in self._traits.items() if not trait.dla]
        if len(gpu) == 0:
            err_msg = "DRIFT needs at least one GPU model"
            raise ValueError(err_msg)
        if baseline is None:
            f1 = {name: float(np.mean(self._traits[name].frame_f1())) for name in gpu}
            baseline = max(gpu, key=lambda name: f1[name])
        elif baseline not in gpu:
            err_msg = f"baseline {baseline!r} must be one of the GPU models {gpu}"
            raise ValueError(err_msg)
        self._baseline = baseline
        self._gpu = gpu
        self._latency = {
            name: trait.latency(self._config.latency_metric) for name, trait in self._traits.items()
        }
        self._estimator = LinearEstimator(
            {
                name: t.frame_confidences(self._config.confidence_metric)
                for name, t in traits.items()
            },
            {name: t.frame_f1() for name, t in traits.items()},
            self._config.accuracy_scale_metric,
        )
        window = self._config.prediction_window
        self._hp_predictions: deque[dict[str, float]] = deque(maxlen=window)
        self._lp_predictions: deque[dict[str, float]] = deque(maxlen=window)

    @property
    def config(self: Self) -> DriftConfig:
        """The hyperparameters."""
        return self._config

    @property
    def estimator(self: Self) -> LinearEstimator:
        """The accuracy estimator fitted on the characterization."""
        return self._estimator

    @property
    def baseline(self: Self) -> Schedule:
        """The baseline schedule: the baseline model on the full frame."""
        return Schedule(
            self._baseline, None, parallel=False, latency_ms=self._latency[self._baseline]
        )

    def __call__(
        self: Self,
        previous: FrameResult | None,
        region: Bbox | None,
        image_size: tuple[int, int],
    ) -> Schedule:
        """
        Schedule a frame.

        Parameters
        ----------
        previous : FrameResult | None
            What ran the previous frame, None for the first frame.
        region : Bbox | None
            The high-priority region of this frame, from :func:`hp_region`.
        image_size : tuple[int, int]
            The frame size as (width, height).

        Returns
        -------
        Schedule
            The models for this frame; the baseline schedule when splitting is not
            expected to be faster at the accuracy goal.

        """
        cfg = self._config
        baseline = self.baseline
        if previous is None or len(previous.detections) == 0 or region is None:
            return baseline
        base = self._baseline
        width, height = image_size

        # high-priority model: fastest GPU model that keeps the region's pixel scale and
        # is predicted at least as accurate as the baseline
        hp_conf = reduce_values([d[1] for d in previous.hp_detections], cfg.confidence_metric)
        hp_pred = self._smooth(
            self._hp_predictions, self._estimator.predict(previous.schedule.hp, hp_conf)
        )
        box_size = reduce_values(
            [((b[2] - b[0]) + (b[3] - b[1])) / 2.0 for b, _, _ in previous.detections],
            cfg.box_size_metric,
        )
        relax = 1.0 - box_size / max(width, height)
        threshold = self._scale(base, width, height) * relax
        region_w, region_h = region[2] - region[0], region[3] - region[1]
        valid = [
            m
            for m in self._gpu
            if m != base
            and self._scale(m, region_w, region_h) >= threshold
            and (hp_pred[m] >= hp_pred[base] or m in cfg.hp_trusted_models)
        ]
        if len(valid) == 0:
            return baseline
        hp = min(valid, key=lambda m: self._latency[m])

        # low-priority model: fastest schedule that meets the accuracy goal and beats
        # the baseline
        if previous.schedule.lp is None:
            source, lp_detections = previous.schedule.hp, previous.detections
        else:
            source, lp_detections = previous.schedule.lp, previous.lp_detections
        lp_conf = reduce_values([d[1] for d in lp_detections], cfg.confidence_metric)
        lp_pred = self._smooth(self._lp_predictions, self._estimator.predict(source, lp_conf))
        goal = cfg.lp_accuracy_goal / self._estimator.scale
        best: Schedule | None = None
        for lp, trait in self._traits.items():
            if lp == base or (cfg.lp_exclude_hp_model and lp == hp) or lp_pred[lp] < goal:
                continue
            serial = self._latency[hp] + self._latency[lp]
            total = serial
            if trait.dla:
                pair = self._traits[hp].pair_latency(lp, cfg.latency_metric)
                total = pair if pair is not None else max(self._latency[hp], self._latency[lp])
            tested = serial if cfg.lp_latency_test == "sum" else total
            if tested >= baseline.latency_ms:
                continue
            if best is None or total < best.latency_ms:
                best = Schedule(hp, lp, trait.dla, total)
        return best if best is not None else baseline

    def reset(self: Self) -> None:
        """Clear the averaged predictions."""
        self._hp_predictions.clear()
        self._lp_predictions.clear()

    def _scale(self: Self, model: str, width: float, height: float) -> float:
        # pixel scale at which the model sees a width x height region (imgsz / long side
        # for square inputs)
        in_w, in_h = self._traits[model].input_size
        return min(in_w / width, in_h / height)

    @staticmethod
    def _smooth(buffer: deque[dict[str, float]], prediction: dict[str, float]) -> dict[str, float]:
        buffer.append(prediction)
        return {m: float(np.mean([p[m] for p in buffer])) for m in prediction}
