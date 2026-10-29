# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Time the runtime options of an image model on your own images."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from trtutils._log import LOG

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ._image_model import ImageModel

_PREPROCESSORS = ("cpu", "cuda", "trt", "hybrid")


@dataclass(frozen=True)
class AutotuneTrial:
    """
    One timed runtime configuration.

    Attributes
    ----------
    preprocessor : str
        Preprocessor used.
    postprocessor : str
        Postprocessor used.
    cuda_graph : bool
        Whether CUDA graphs were enabled.
    median_ms : float
        Median end2end time per image in milliseconds (inf if the trial failed).
    p90_ms : float
        90th percentile end2end time per image in milliseconds (inf if the trial failed).
    same_as_default : bool | None
        Whether the results equal the default configuration's exactly, None if the trial failed.
    error : str | None
        The error message if the trial failed.

    """

    preprocessor: str
    postprocessor: str
    cuda_graph: bool
    median_ms: float
    p90_ms: float
    same_as_default: bool | None
    error: str | None

    @property
    def kwargs(self) -> dict[str, Any]:
        """Constructor keyword arguments for this configuration."""
        return {
            "preprocessor": self.preprocessor,
            "postprocessor": self.postprocessor,
            "cuda_graph": self.cuda_graph,
        }


def _copy(obj: object) -> object:
    # results can alias reused buffers, so detach them
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_copy(item) for item in obj)
    return obj


def _equal(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


def _run_trial(
    model_cls: type[ImageModel],
    engine_path: Path | str,
    images: Sequence[np.ndarray],
    kwargs: dict[str, Any],
    iterations: int,
    warmup: int,
    model_kwargs: dict[str, Any],
) -> tuple[float, float, list[Any]]:
    # end2end is defined per task class, not on ImageModel
    model: Any = model_cls(engine_path, warmup=False, **kwargs, **model_kwargs)
    try:
        for _ in range(warmup):
            for image in images:
                model.end2end([image])
        times: list[float] = []
        for _ in range(iterations):
            for image in images:
                t0 = time.perf_counter()
                model.end2end([image])
                times.append(time.perf_counter() - t0)
        results = [_copy(model.end2end([image])) for image in images]
    finally:
        del model
    median, p90 = np.percentile(times, [50, 90]) * 1000.0
    return float(median), float(p90), results


def autotune(
    model_cls: type[ImageModel],
    engine_path: Path | str,
    images: Sequence[np.ndarray],
    *,
    preprocessors: Sequence[str] | None = None,
    postprocessors: Sequence[str] | None = None,
    cuda_graphs: Sequence[bool] | None = None,
    iterations: int = 50,
    warmup: int = 5,
    model_kwargs: dict[str, Any] | None = None,
    verbose: bool | None = None,
) -> list[AutotuneTrial]:
    """
    Time every combination of runtime options and rank them.

    Each trial times ``end2end`` per image over all of ``images``. Configurations that fail
    to build or run are reported with ``error`` set instead of raising.

    Parameters
    ----------
    model_cls : type[ImageModel]
        The image model class to tune, e.g. ``Detector``.
    engine_path : Path | str
        Path to the TensorRT engine.
    images : Sequence[np.ndarray]
        HWC uint8 images to time on.
    preprocessors : Sequence[str], optional
        Preprocessors to try. By default all of cpu, cuda, trt and hybrid.
    postprocessors : Sequence[str], optional
        Postprocessors to try. By default cpu and cuda where the model has a CUDA
        postprocessor, otherwise cpu.
    cuda_graphs : Sequence[bool], optional
        CUDA graph settings to try. By default True and False.
    iterations : int, optional
        Timed passes over the images per trial, by default 50.
    warmup : int, optional
        Untimed passes over the images per trial, by default 5.
    model_kwargs : dict[str, Any], optional
        Extra constructor arguments for every model, e.g. thresholds.
    verbose : bool, optional
        Log each trial as it finishes.

    Returns
    -------
    list[AutotuneTrial]
        The trials, fastest median first and failed trials last.

    """
    preprocessors = preprocessors if preprocessors is not None else _PREPROCESSORS
    if postprocessors is None:
        postprocessors = ("cpu", "cuda") if model_cls._cuda_postprocess else ("cpu",)  # noqa: SLF001
    cuda_graphs = cuda_graphs if cuda_graphs is not None else (True, False)
    model_kwargs = model_kwargs if model_kwargs is not None else {}

    default_model: Any = model_cls(engine_path, warmup=False, **model_kwargs)
    default_results = [_copy(default_model.end2end([image])) for image in images]
    del default_model

    trials: list[AutotuneTrial] = []
    for pre, post, graph in itertools.product(preprocessors, postprocessors, cuda_graphs):
        kwargs = {"preprocessor": pre, "postprocessor": post, "cuda_graph": graph}
        try:
            median, p90, results = _run_trial(
                model_cls, engine_path, images, kwargs, iterations, warmup, model_kwargs
            )
            same = _equal(results, default_results)
            trial = AutotuneTrial(pre, post, graph, median, p90, same, None)
        except Exception as exc:  # noqa: BLE001
            trial = AutotuneTrial(pre, post, graph, float("inf"), float("inf"), None, str(exc))
        if verbose:
            LOG.info(f"autotune: {trial}")
        trials.append(trial)

    trials.sort(key=lambda t: (t.error is not None, t.median_ms))
    return trials
