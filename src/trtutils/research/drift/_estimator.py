# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""DRIFT accuracy estimation: confidence to F1 linear models (Section 3.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trtutils.research.shift import reduce_values

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self


class LinearEstimator:
    """
    Predicts every model's accuracy from one model's detection confidence.

    For every source model a least-squares line maps that model's per-sample
    confidence to the per-sample accuracy of every model (itself included), so the
    confidence of whichever model ran predicts all of them. Usable on its own, from
    any aligned per-sample confidences and accuracies.
    """

    def __init__(
        self: Self,
        confidences: Mapping[str, Sequence[float]],
        accuracies: Mapping[str, Sequence[float]],
        scale_metric: str = "median",
    ) -> None:
        """
        Fit the estimator.

        Parameters
        ----------
        confidences : Mapping[str, Sequence[float]]
            Per model, one confidence per sample.
        accuracies : Mapping[str, Sequence[float]]
            Per model, one accuracy (F1) per sample, aligned with ``confidences``.
        scale_metric : str, optional
            Reduction of all accuracies that sets :attr:`scale`: "mean", "median",
            "min" or "max". Default is "median".

        Raises
        ------
        ValueError
            If the models differ, there are none, or the sample counts differ or are zero.

        """
        if set(confidences) != set(accuracies) or len(accuracies) == 0:
            err_msg = (
                f"confidences {sorted(confidences)} and accuracies {sorted(accuracies)} "
                "must name the same, non-empty set of models"
            )
            raise ValueError(err_msg)
        self._models = list(accuracies)
        lengths = {len(confidences[m]) for m in self._models} | {
            len(accuracies[m]) for m in self._models
        }
        if len(lengths) != 1 or 0 in lengths:
            err_msg = f"every model needs the same, non-zero number of samples, got {lengths}"
            raise ValueError(err_msg)
        targets = np.array([accuracies[m] for m in self._models], dtype=np.float64).T
        target_mean = targets.mean(axis=0)
        self._slopes: dict[str, np.ndarray] = {}
        self._intercepts: dict[str, np.ndarray] = {}
        for model in self._models:
            x = np.asarray(confidences[model], dtype=np.float64)
            centred = x - x.mean()
            variance = float(centred @ centred)
            if variance > 0.0:
                slope = (centred @ (targets - target_mean)) / variance
            else:
                slope = np.zeros(len(self._models))
            self._slopes[model] = slope
            self._intercepts[model] = target_mean - slope * x.mean()
        pooled = reduce_values([float(a) for m in self._models for a in accuracies[m]], scale_metric)
        self._scale = 0.5 / pooled if pooled > 0.0 else 1.0

    @property
    def models(self: Self) -> list[str]:
        """The models, in fit order."""
        return list(self._models)

    @property
    def scale(self: Self) -> float:
        """
        Dataset-difficulty scale, ``0.5 / reduce(all accuracies)``.

        An accuracy goal ``g`` is compared against predictions as ``g / scale``, so a
        goal of 0.5 means the pooled (by default median) characterized accuracy.
        """
        return self._scale

    def predict(self: Self, model: str, confidence: float) -> dict[str, float]:
        """
        Predict the accuracy of every model.

        Parameters
        ----------
        model : str
            The model whose confidence is given.
        confidence : float
            That model's (reduced) detection confidence.

        Returns
        -------
        dict[str, float]
            The predicted accuracy of every model. Not clipped to [0, 1].

        Raises
        ------
        ValueError
            If ``model`` is unknown.

        """
        if model not in self._slopes:
            err_msg = f"Unknown model {model!r}, expected one of {self._models}"
            raise ValueError(err_msg)
        values = self._slopes[model] * confidence + self._intercepts[model]
        return {m: float(v) for m, v in zip(self._models, values)}
