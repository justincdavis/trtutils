# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Characterization data and configuration for SHIFT."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

_REDUCERS: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": lambda values: float(np.mean(values)),
    "median": lambda values: float(np.median(values)),
    "min": min,
    "max": max,
}
_SIMILARITIES = ("min", "product")


def reduce_values(values: Sequence[float], metric: str) -> float:
    """
    Reduce per-object values (confidences or IoUs) to one per-frame value.

    Parameters
    ----------
    values : Sequence[float]
        The per-object values.
    metric : str
        One of "mean", "median", "min" or "max".

    Returns
    -------
    float
        The reduced value, or 0.0 when ``values`` is empty.

    Raises
    ------
    ValueError
        If ``metric`` is not a known reducer.

    """
    if metric not in _REDUCERS:
        err_msg = f"Unknown metric {metric!r}, expected one of {sorted(_REDUCERS)}"
        raise ValueError(err_msg)
    if len(values) == 0:
        return 0.0
    return float(_REDUCERS[metric](values))


@dataclass
class ModelTraits:
    """
    Characterized traits of one (model, accelerator) pair.

    The per-image lists are aligned by image index across all pairs of a
    characterization, which is what the confidence graph relies on.

    Attributes
    ----------
    latency_ms : float
        Mean engine latency in milliseconds.
    energy_mj : float
        Mean energy per inference in millijoules. 0.0 when not measured (non-Jetson).
    load_time_s : float
        Time to construct the model, in seconds.
    load_memory_mb : float
        Device memory used by the constructed model, in megabytes.
    confidences : list[list[float]]
        Per image, the confidence of every detection.
    ious : list[list[float]]
        Per image, the best IoU of any detection with each ground-truth box.

    """

    latency_ms: float
    energy_mj: float
    load_time_s: float
    load_memory_mb: float
    confidences: list[list[float]]
    ious: list[list[float]]

    def frame_confidences(self: Self, metric: str = "mean") -> list[float]:
        """
        Get one confidence per image.

        Parameters
        ----------
        metric : str, optional
            How to reduce the detections of an image: "mean", "median", "min"
            or "max". Images without detections get 0.0. Default is "mean".

        Returns
        -------
        list[float]
            The per-image confidences.

        """
        return [reduce_values(confs, metric) for confs in self.confidences]

    def frame_accuracies(self: Self, metric: str = "mean") -> list[float]:
        """
        Get one accuracy per image.

        An image without ground-truth boxes scores 1.0 when the model also found
        nothing, and 0.0 otherwise.

        Parameters
        ----------
        metric : str, optional
            How to reduce the per-ground-truth IoUs of an image: "mean",
            "median", "min" or "max". Default is "mean".

        Returns
        -------
        list[float]
            The per-image accuracies.

        """
        return [
            reduce_values(ious, metric) if len(ious) > 0 else float(len(confs) == 0)
            for ious, confs in zip(self.ious, self.confidences)
        ]


@dataclass
class ShiftConfig:
    """
    Hyperparameters of the SHIFT scheduler.

    The defaults are the values from the SHIFT paper (Table III). Use
    :meth:`ShiftConfig.shiftdetector` for the values of the reference
    implementation where it differs from the paper.

    Attributes
    ----------
    accuracy_threshold : float
        Goal accuracy. Keeps the current model while
        ``similarity * confidence`` is at least this value, and filters the
        candidates by predicted accuracy. Default is 0.25.
    momentum : int
        Number of predictions averaged per model. Default is 30.
    distance_threshold : float
        Maximum confidence-graph path cost to a neighbor. Default is 0.5.
    accuracy_weight : float
        Weight of the predicted accuracy in the score. Default is 1.0.
    energy_weight : float
        Weight of the inverted, normalized energy in the score. Default is 0.5.
    latency_weight : float
        Weight of the inverted, normalized latency in the score. Default is 0.5.
    similarity : str
        How the frame NCC and the bounding-box NCC combine: "min" (paper) or
        "product" (shiftdetector). Default is "min".
    confidence_metric : str
        Reduction of a frame's detection confidences: "mean", "median", "min"
        or "max" ("max" is the paper's top detection). Default is "mean".
    accuracy_metric : str
        Reduction of a frame's per-ground-truth IoUs, same choices. Default is "mean".
    num_bins : int
        Number of confidence bins per model in the confidence graph. Default is 10.
    outlier_percentile : float
        Percentile of a node's edge counts used to normalize its edges; counts
        above it are clamped. 100 normalizes by the maximum. Default is 100.0.
    min_connects : int
        Edges seen this many times or fewer are dropped. Default is 0.

    """

    accuracy_threshold: float = 0.25
    momentum: int = 30
    distance_threshold: float = 0.5
    accuracy_weight: float = 1.0
    energy_weight: float = 0.5
    latency_weight: float = 0.5
    similarity: str = "min"
    confidence_metric: str = "mean"
    accuracy_metric: str = "mean"
    num_bins: int = 10
    outlier_percentile: float = 100.0
    min_connects: int = 0

    def __post_init__(self: Self) -> None:
        if self.similarity not in _SIMILARITIES:
            err_msg = f"Unknown similarity {self.similarity!r}, expected one of {_SIMILARITIES}"
            raise ValueError(err_msg)
        for metric in (self.confidence_metric, self.accuracy_metric):
            if metric not in _REDUCERS:
                err_msg = f"Unknown metric {metric!r}, expected one of {sorted(_REDUCERS)}"
                raise ValueError(err_msg)
        if self.momentum < 1:
            err_msg = f"momentum must be at least 1, got {self.momentum}"
            raise ValueError(err_msg)

    @classmethod
    def shiftdetector(cls: type[Self]) -> Self:
        """
        Get the configuration used by the shiftdetector reference implementation.

        shiftdetector is single-object, so frames reduce to their top detection
        (``"max"`` metrics).

        Returns
        -------
        ShiftConfig
            The reference implementation's hyperparameters.

        """
        return cls(
            accuracy_threshold=0.5,
            momentum=10,
            distance_threshold=1.0,
            accuracy_weight=0.85,
            energy_weight=0.10,
            latency_weight=0.10,
            similarity="product",
            confidence_metric="max",
            accuracy_metric="max",
            outlier_percentile=80.0,
            min_connects=1,
        )


def save_characterization(traits: Mapping[str, ModelTraits], path: Path | str) -> None:
    """
    Save a characterization to a JSON file.

    Parameters
    ----------
    traits : Mapping[str, ModelTraits]
        The traits of each (model, accelerator) pair, keyed by pair name.
    path : Path | str
        The JSON file to write.

    """
    data = {name: asdict(trait) for name, trait in traits.items()}
    with Path(path).open("w") as f:
        json.dump(data, f, indent=4)


def load_characterization(path: Path | str) -> dict[str, ModelTraits]:
    """
    Load a characterization written by :func:`save_characterization`.

    Parameters
    ----------
    path : Path | str
        The JSON file to read.

    Returns
    -------
    dict[str, ModelTraits]
        The traits of each (model, accelerator) pair, keyed by pair name.

    """
    with Path(path).open() as f:
        data = json.load(f)
    return {name: ModelTraits(**entry) for name, entry in data.items()}
