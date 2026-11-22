# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""SHIFT per-frame model scheduler (Algorithm 1 of the paper)."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ._graph import ConfidenceGraph
from ._types import ShiftConfig, reduce_values

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

    from trtutils.image.types import Detection

    from ._types import ModelTraits

_FRAME_SIZE = (112, 112)
_BOX_SIZE = (24, 24)


def ncc(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute the zero-mean normalized cross-correlation of two images.

    Parameters
    ----------
    image1 : np.ndarray
        The first image.
    image2 : np.ndarray
        The second image, same shape as ``image1``.

    Returns
    -------
    float
        The NCC in [-1, 1], or 0.0 when either image has no variance.

    Raises
    ------
    ValueError
        If the images have different shapes.

    """
    if image1.shape != image2.shape:
        err_msg = f"images must have the same shape, got {image1.shape} and {image2.shape}"
        raise ValueError(err_msg)
    a = image1.astype(np.float64).ravel()
    b = image2.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _gray(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # resize first so the color conversion runs on the small image
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if image.ndim == 3:  # noqa: PLR2004
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _box_crop(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray | None:
    # the box enclosing every detection, clipped to the image
    if len(detections) == 0:
        return None
    height, width = image.shape[:2]
    x1 = max(min(int(d[0][0]) for d in detections), 0)
    y1 = max(min(int(d[0][1]) for d in detections), 0)
    x2 = min(max(int(d[0][2]) for d in detections), width)
    y2 = min(max(int(d[0][3]) for d in detections), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return _gray(image[y1:y2, x1:x2], _BOX_SIZE)


def _inverted(values: dict[str, float]) -> dict[str, float]:
    # normalized to [0, 1] and inverted, so bigger is better; all 0.0 when nothing was measured
    top = max(values.values())
    return {name: 1.0 - v / top if top > 0.0 else 0.0 for name, v in values.items()}


class ShiftScheduler:
    """
    Pick the (model, accelerator) pair to run on the next frame.

    Implements Algorithm 1 of the SHIFT paper. The current model is kept while
    the scene is stable and the model is confident, measured as
    ``similarity * confidence >= accuracy_threshold``, where similarity combines
    the NCC between consecutive frames and between consecutive detection boxes.
    Otherwise the confidence graph predicts every model's accuracy from the current
    model's confidence, predictions are averaged over the last ``momentum``
    reschedules, models below ``accuracy_threshold`` are dropped (unless none
    remain), and the model with the best weighted score of accuracy, inverted
    energy and inverted latency wins.
    """

    def __init__(
        self: Self,
        traits: Mapping[str, ModelTraits],
        config: ShiftConfig | None = None,
    ) -> None:
        """
        Create the scheduler.

        Parameters
        ----------
        traits : Mapping[str, ModelTraits]
            The characterized traits of each (model, accelerator) pair.
        config : ShiftConfig, optional
            The hyperparameters. Default is ``ShiftConfig()``, the paper's values.

        """
        self._config = config if config is not None else ShiftConfig()
        cfg = self._config
        self._graph = ConfidenceGraph(
            {name: t.frame_confidences(cfg.confidence_metric) for name, t in traits.items()},
            {name: t.frame_accuracies(cfg.accuracy_metric) for name, t in traits.items()},
            num_bins=cfg.num_bins,
            distance_threshold=cfg.distance_threshold,
            outlier_percentile=cfg.outlier_percentile,
            min_connects=cfg.min_connects,
        )
        self._energy = _inverted({name: t.energy_mj for name, t in traits.items()})
        self._latency = _inverted({name: t.latency_ms for name, t in traits.items()})
        self._buffers: dict[str, deque[float]] = {
            name: deque(maxlen=cfg.momentum) for name in traits
        }
        self._last_frame: np.ndarray | None = None
        self._last_box: np.ndarray | None = None

    @property
    def config(self: Self) -> ShiftConfig:
        """The scheduler's hyperparameters."""
        return self._config

    @property
    def graph(self: Self) -> ConfidenceGraph:
        """The confidence graph built from the characterization."""
        return self._graph

    def __call__(
        self: Self,
        model: str,
        image: np.ndarray,
        detections: Sequence[Detection],
    ) -> str:
        """
        Pick the model for the next frame.

        Parameters
        ----------
        model : str
            The model that produced ``detections``.
        image : np.ndarray
            The frame, BGR or grayscale.
        detections : Sequence[Detection]
            The model's detections on ``image``.

        Returns
        -------
        str
            The model to run on the next frame.

        Raises
        ------
        ValueError
            If ``model`` was not characterized.

        """
        if model not in self._buffers:
            err_msg = f"Unknown model {model!r}, expected one of {list(self._buffers)}"
            raise ValueError(err_msg)
        cfg = self._config
        confidence = reduce_values([d[1] for d in detections], cfg.confidence_metric)
        if self._similarity(image, detections) * confidence >= cfg.accuracy_threshold:
            return model

        predictions = self._graph.predict(model, confidence)
        if len(predictions) == 0:
            return model
        accuracies: dict[str, float] = {}
        for name, accuracy in predictions.items():
            self._buffers[name].append(accuracy)
            accuracies[name] = float(np.mean(self._buffers[name]))
        valid = {n: a for n, a in accuracies.items() if a >= cfg.accuracy_threshold}
        if len(valid) == 0:
            valid = accuracies
        scores = {
            name: accuracy * cfg.accuracy_weight
            + self._energy[name] * cfg.energy_weight
            + self._latency[name] * cfg.latency_weight
            for name, accuracy in valid.items()
        }
        return max(scores, key=lambda name: scores[name])

    def reset(self: Self) -> None:
        """Forget the previous frame and the averaged predictions."""
        for buffer in self._buffers.values():
            buffer.clear()
        self._last_frame = None
        self._last_box = None

    def _similarity(self: Self, image: np.ndarray, detections: Sequence[Detection]) -> float:
        frame = _gray(image, _FRAME_SIZE)
        box = _box_crop(image, detections)
        frame_ncc = ncc(self._last_frame, frame) if self._last_frame is not None else 0.0
        box_ncc = 0.0
        if self._last_box is not None and box is not None:
            box_ncc = ncc(self._last_box, box)
        self._last_frame = frame
        self._last_box = box
        frame_ncc = max(frame_ncc, 0.0)
        box_ncc = max(box_ncc, 0.0)
        if self._config.similarity == "min":
            return min(frame_ncc, box_ncc)
        return frame_ncc * box_ncc
