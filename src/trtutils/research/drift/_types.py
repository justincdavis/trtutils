# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Characterization data, configuration and per-frame records for DRIFT."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from trtutils.research.shift import reduce_values

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

    from trtutils.image.types import Detection

_COUNTER_DECREMENTS = ("all", "empty")
_DENSE_GROUPS = ("rectangles", "blocks")
_ANNEAL_COUNTERS = ("frames", "packs")
_LATENCY_TESTS = ("sum", "schedule")


class Schedule(NamedTuple):
    """
    The models that run one frame.

    Attributes
    ----------
    hp : str
        The model on the high-priority crop, or on the full frame when ``lp`` is None.
    lp : str | None
        The model on the packed low-priority image. None means the baseline runs
        the full frame alone.
    parallel : bool
        Whether ``lp`` runs on a DLA concurrently with ``hp`` (else serially on the GPU).
    latency_ms : float
        The characterized latency of the schedule in milliseconds.

    """

    hp: str
    lp: str | None
    parallel: bool
    latency_ms: float


class FrameResult(NamedTuple):
    """
    What DRIFT produced for one frame, in frame coordinates.

    Attributes
    ----------
    detections : list[Detection]
        The merged detections of the frame.
    hp_detections : list[Detection]
        The detections of ``schedule.hp`` (all detections on a baseline frame).
    lp_detections : list[Detection]
        The detections of ``schedule.lp`` (empty on a baseline frame or when no
        low-priority cell was sampled).
    schedule : Schedule
        The schedule that ran the frame.

    """

    detections: list[Detection]
    hp_detections: list[Detection]
    lp_detections: list[Detection]
    schedule: Schedule


@dataclass
class ModelTraits:
    """
    Characterized traits of one detection model on one accelerator.

    The per-image lists are aligned by image index across all models of a
    characterization, which is what :class:`LinearEstimator` relies on. Raw
    samples are stored so every hyperparameter can change without re-profiling.

    Attributes
    ----------
    input_size : tuple[int, int]
        The model input as (width, height).
    dla : bool
        Whether the model runs on a DLA.
    latency_ms : list[float]
        Raw end-to-end latency samples (preprocess, inference, postprocess) in ms.
    confidences : list[list[float]]
        Per image, the confidence of every detection.
    true_positives : list[int]
        Per image, the number of detections matched to a ground-truth box.
    ground_truths : list[int]
        Per image, the number of ground-truth boxes.
    energy_mj : list[float]
        Raw energy samples per call in millijoules. Empty when not measured (non-Jetson).
    pair_latency_ms : dict[str, list[float]]
        On GPU models: raw latency samples of this model running concurrently with
        each DLA model, keyed by the DLA model's name.
    pair_energy_mj : dict[str, list[float]]
        The energy samples of the same pairs. Empty when not measured.

    """

    input_size: tuple[int, int]
    dla: bool
    latency_ms: list[float]
    confidences: list[list[float]]
    true_positives: list[int]
    ground_truths: list[int]
    energy_mj: list[float] = field(default_factory=list)
    pair_latency_ms: dict[str, list[float]] = field(default_factory=dict)
    pair_energy_mj: dict[str, list[float]] = field(default_factory=dict)

    def latency(self: Self, metric: str = "mean") -> float:
        """
        Get the model's latency.

        Parameters
        ----------
        metric : str, optional
            How to reduce the samples: "mean", "median", "min" or "max". Default is "mean".

        Returns
        -------
        float
            The latency in milliseconds, 0.0 without samples.

        """
        return reduce_values(self.latency_ms, metric)

    def pair_latency(self: Self, lp: str, metric: str = "mean") -> float | None:
        """
        Get the latency of this model running concurrently with a DLA model.

        Parameters
        ----------
        lp : str
            The DLA model.
        metric : str, optional
            How to reduce the samples: "mean", "median", "min" or "max". Default is "mean".

        Returns
        -------
        float | None
            The latency in milliseconds, or None when the pair was not measured.

        """
        samples = self.pair_latency_ms.get(lp)
        if not samples:
            return None
        return reduce_values(samples, metric)

    def frame_confidences(self: Self, metric: str = "mean") -> list[float]:
        """
        Get one confidence per image.

        Parameters
        ----------
        metric : str, optional
            How to reduce the detections of an image: "mean", "median", "min" or
            "max". Images without detections get 0.0. Default is "mean".

        Returns
        -------
        list[float]
            The per-image confidences.

        """
        return [reduce_values(confs, metric) for confs in self.confidences]

    def frame_f1(self: Self) -> list[float]:
        """
        Get the F1 score of every image.

        An image without detections and without ground truth scores 1.0.

        Returns
        -------
        list[float]
            The per-image F1 scores, ``2 * tp / (detections + ground_truths)``.

        """
        scores: list[float] = []
        for confs, tp, gt in zip(self.confidences, self.true_positives, self.ground_truths):
            total = len(confs) + gt
            scores.append(1.0 if total == 0 else 2.0 * tp / total)
        return scores


@dataclass
class DriftConfig:
    """
    Hyperparameters of DRIFT.

    The defaults are the DRIFT paper's values. Where the paper gives none, they are
    the values of the reference implementation's documented run. Use
    :meth:`DriftConfig.upstream` for the reference implementation's defaults.

    Attributes
    ----------
    hp_padding : int
        Pixels added around the high-priority box on every side. Default is 64.
    hp_square : bool
        Whether to grow the short side of the high-priority crop toward a square
        (reference implementation). The low-priority exclusion and the scheduler
        always use the unsquared box. Default is False.
    hp_min_area : float
        Priority-class boxes with an area at or below this are ignored. Default is 0.0.
    grid_divisor : int
        The packing grid's cell side is ``max(width, height) // grid_divisor``.
        Default is 10.
    annealing_rate : float
        The exponential decay rate (alpha) of the sampling probability. Default is 0.01.
    annealing_floor : float
        The minimum sampling probability of a cell (p_min). Default is 0.1.
    fps : float
        The frame-rate term that bounds the detection factor. Default is 30.0.
    counter_decrement : str
        How cell detection counters decay after each frame: "all" decrements every
        cell by one after adding its hits (the paper's description of the sampler's
        behaviour, and the reference implementation), "empty" decrements only cells
        without hits (the paper's formulation). Default is "all".
    dense_groups : str
        How sampled cells group before packing: "rectangles" keeps every fully filled
        connected component together (paper), "blocks" only forms 2x2, 1x2 and 2x1
        blocks (reference implementation). Default is "rectangles".
    anneal_counter : str
        What the annealing counter counts: "frames" (paper) or "packs", the frames
        that were split (reference implementation). Default is "frames".
    lp_accuracy_goal : float
        The low-priority accuracy goal, from 0 to 1. Default is 0.5.
    confidence_metric : str
        Reduction of a frame's detection confidences: "mean", "median", "min" or
        "max". Default is "mean".
    box_size_metric : str
        Reduction of the box sizes that relax the high-priority scale test, same
        choices. Default is "median".
    accuracy_scale_metric : str
        Reduction of all characterized F1 scores that scales the accuracy goal, same
        choices. Default is "median".
    latency_metric : str
        Reduction of raw latency samples, same choices. Default is "mean".
    prediction_window : int
        Number of frames of F1 predictions averaged (reference implementation: 10).
        Default is 1.
    hp_trusted_models : tuple[str, ...]
        High-priority models exempt from the accuracy check (the reference
        implementation exempts smaller-input variants of the baseline). Default is ().
    lp_exclude_hp_model : bool
        Whether the low-priority model must differ from the high-priority one
        (reference implementation). Default is False.
    lp_latency_test : str
        Which latency must beat the baseline: "sum" of the two models (paper) or the
        "schedule" latency, measured concurrently for DLA pairs (reference
        implementation). Default is "sum".
    merge_iou : float
        IoU threshold of the class-aware NMS that merges high- and low-priority
        detections. Default is 0.5.

    """

    hp_padding: int = 64
    hp_square: bool = False
    hp_min_area: float = 0.0
    grid_divisor: int = 10
    annealing_rate: float = 0.01
    annealing_floor: float = 0.1
    fps: float = 30.0
    counter_decrement: str = "all"
    dense_groups: str = "rectangles"
    anneal_counter: str = "frames"
    lp_accuracy_goal: float = 0.5
    confidence_metric: str = "mean"
    box_size_metric: str = "median"
    accuracy_scale_metric: str = "median"
    latency_metric: str = "mean"
    prediction_window: int = 1
    hp_trusted_models: tuple[str, ...] = ()
    lp_exclude_hp_model: bool = False
    lp_latency_test: str = "sum"
    merge_iou: float = 0.5

    def __post_init__(self: Self) -> None:
        self.hp_trusted_models = tuple(self.hp_trusted_models)
        choices = {
            "counter_decrement": (self.counter_decrement, _COUNTER_DECREMENTS),
            "dense_groups": (self.dense_groups, _DENSE_GROUPS),
            "anneal_counter": (self.anneal_counter, _ANNEAL_COUNTERS),
            "lp_latency_test": (self.lp_latency_test, _LATENCY_TESTS),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                err_msg = f"Unknown {name} {value!r}, expected one of {allowed}"
                raise ValueError(err_msg)
        for metric in (
            self.confidence_metric,
            self.box_size_metric,
            self.accuracy_scale_metric,
            self.latency_metric,
        ):
            reduce_values([0.0], metric)
        bounds = {
            "hp_padding": self.hp_padding >= 0,
            "hp_min_area": self.hp_min_area >= 0.0,
            "grid_divisor": self.grid_divisor >= 1,
            "annealing_rate": self.annealing_rate >= 0.0,
            "annealing_floor": 0.0 <= self.annealing_floor <= 1.0,
            "fps": self.fps > 0.0,
            "lp_accuracy_goal": 0.0 <= self.lp_accuracy_goal <= 1.0,
            "prediction_window": self.prediction_window >= 1,
            "merge_iou": 0.0 < self.merge_iou <= 1.0,
        }
        for name, ok in bounds.items():
            if not ok:
                err_msg = f"{name} is out of range, got {getattr(self, name)}"
                raise ValueError(err_msg)

    @classmethod
    def upstream(cls: type[Self], hp_trusted_models: Sequence[str] = ()) -> Self:
        """
        Get the configuration of the DRIFT reference implementation.

        Parameters
        ----------
        hp_trusted_models : Sequence[str], optional
            The high-priority models to exempt from the accuracy check. The
            reference implementation exempts every smaller-input variant of the
            baseline architecture, which depends on the model names. Default is ().

        Returns
        -------
        DriftConfig
            The reference implementation's hyperparameters.

        """
        return cls(
            hp_padding=128,
            hp_square=True,
            dense_groups="blocks",
            anneal_counter="packs",
            lp_accuracy_goal=0.4,
            prediction_window=10,
            hp_trusted_models=tuple(hp_trusted_models),
            lp_exclude_hp_model=True,
            lp_latency_test="schedule",
        )


def save_characterization(traits: Mapping[str, ModelTraits], path: Path | str) -> None:
    """
    Save a characterization to a JSON file.

    Parameters
    ----------
    traits : Mapping[str, ModelTraits]
        The traits of each model, keyed by model name.
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
        The traits of each model, keyed by model name.

    """
    with Path(path).open() as f:
        data = json.load(f)
    traits: dict[str, ModelTraits] = {}
    for name, entry in data.items():
        entry["input_size"] = tuple(entry["input_size"])
        traits[name] = ModelTraits(**entry)
    return traits
