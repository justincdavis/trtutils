# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Confidence graph: predict every model's accuracy from one model's confidence."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

#: a confidence graph node as (model, bin)
Node: TypeAlias = Tuple[str, int]


class ConfidenceGraph:
    """
    Predict how accurate each model would be on a sample from one model's confidence.

    Each node is a (model, confidence bin) pair, holding the mean accuracy of the
    samples where that model's confidence fell in that bin. Each directed edge
    ``u -> v`` counts the samples where both ``u`` and ``v`` occurred. Counts are
    normalized per source node and inverted into traversal costs in [0, 1], so
    nodes that often co-occur are close. A node's prediction for a model is the
    average accuracy of that model's nodes reachable within ``distance_threshold``,
    weighted by ``((threshold - distance) / threshold) ** 2``.

    The graph only needs per-sample confidences and accuracies, so it works for any
    set of models and any per-sample accuracy metric.

    Examples
    --------
    >>> graph = ConfidenceGraph(
    ...     confidences={"big": [0.9, 0.8, 0.2], "small": [0.7, 0.4, 0.0]},
    ...     accuracies={"big": [0.85, 0.8, 0.3], "small": [0.7, 0.35, 0.0]},
    ... )
    >>> graph.predict("small", 0.65)
    {'big': 0.85, 'small': 0.7}

    """

    def __init__(
        self: Self,
        confidences: Mapping[str, Sequence[float]],
        accuracies: Mapping[str, Sequence[float]],
        num_bins: int = 10,
        distance_threshold: float = 0.5,
        outlier_percentile: float = 100.0,
        min_connects: int = 0,
    ) -> None:
        """
        Build the confidence graph.

        Parameters
        ----------
        confidences : Mapping[str, Sequence[float]]
            Per model, one confidence in [0, 1] per sample. Samples are aligned by
            index across models.
        accuracies : Mapping[str, Sequence[float]]
            Per model, one accuracy per sample, aligned with ``confidences``.
        num_bins : int, optional
            Number of confidence bins per model. Bin 0 holds confidence 0 exactly
            (no detection); bin ``k`` holds ``((k - 1) / num_bins, k / num_bins]``.
            Default is 10.
        distance_threshold : float, optional
            Maximum path cost from a node to a neighbor. Default is 0.5.
        outlier_percentile : float, optional
            Percentile of a node's edge counts used to normalize its edges. Counts
            above it are clamped, so outlier edges do not squash the rest.
            100 normalizes by the maximum count. Default is 100.0.
        min_connects : int, optional
            Edges seen this many times or fewer are dropped. Default is 0.

        Raises
        ------
        ValueError
            If the models or sample counts of ``confidences`` and ``accuracies``
            differ, or a parameter is out of range.

        """
        if set(confidences) != set(accuracies) or len(confidences) == 0:
            err_msg = "confidences and accuracies must have the same, non-empty set of models"
            raise ValueError(err_msg)
        counts = {len(values) for values in (*confidences.values(), *accuracies.values())}
        if len(counts) != 1:
            err_msg = "every model needs the same number of confidences and accuracies"
            raise ValueError(err_msg)
        if num_bins < 1:
            err_msg = f"num_bins must be at least 1, got {num_bins}"
            raise ValueError(err_msg)
        if distance_threshold <= 0.0:
            err_msg = f"distance_threshold must be positive, got {distance_threshold}"
            raise ValueError(err_msg)
        if not 0.0 < outlier_percentile <= 100.0:  # noqa: PLR2004
            err_msg = f"outlier_percentile must be in (0, 100], got {outlier_percentile}"
            raise ValueError(err_msg)

        self._models = list(confidences)
        self._num_bins = num_bins
        self._distance_threshold = distance_threshold
        bins = {m: [self.bin(c) for c in confidences[m]] for m in self._models}

        # node accuracy is the mean accuracy of its samples
        samples: dict[Node, list[float]] = defaultdict(list)
        for model in self._models:
            for b, accuracy in zip(bins[model], accuracies[model]):
                samples[(model, b)].append(float(accuracy))
        self._nodes = {node: float(np.mean(accs)) for node, accs in samples.items()}

        # co-occurrence counts between nodes of different models
        edge_counts: dict[Node, Counter[Node]] = defaultdict(Counter)
        for i in range(counts.pop()):
            sample_nodes = [(m, bins[m][i]) for m in self._models]
            for u in sample_nodes:
                for v in sample_nodes:
                    if u[0] != v[0]:
                        edge_counts[u][v] += 1

        # normalize per source node and invert into costs
        self._edges: dict[Node, dict[Node, float]] = {}
        for u, targets in edge_counts.items():
            norm = float(np.percentile(list(targets.values()), outlier_percentile))
            self._edges[u] = {
                v: 1.0 - min(count, norm) / norm
                for v, count in targets.items()
                if count > min_connects
            }

        self._predictions = {node: self._predict_node(node) for node in self._nodes}

    @property
    def models(self: Self) -> list[str]:
        """The models in the graph."""
        return list(self._models)

    @property
    def nodes(self: Self) -> dict[tuple[str, int], float]:
        """The expected accuracy of each (model, bin) node."""
        return dict(self._nodes)

    @property
    def edges(self: Self) -> dict[tuple[str, int], dict[tuple[str, int], float]]:
        """The traversal cost of each directed edge, keyed by source node."""
        return {u: dict(targets) for u, targets in self._edges.items()}

    def bin(self: Self, confidence: float) -> int:
        """
        Get the bin index of a confidence.

        Parameters
        ----------
        confidence : float
            A confidence in [0, 1]; values outside are clamped.

        Returns
        -------
        int
            The bin index in [0, num_bins].

        """
        # round first so 0.3 * 10 lands in bin 3, not 4
        index = math.ceil(round(confidence * self._num_bins, 9))
        return min(max(index, 0), self._num_bins)

    def predict(self: Self, model: str, confidence: float) -> dict[str, float]:
        """
        Predict each model's accuracy given one model's confidence.

        Parameters
        ----------
        model : str
            The model that produced ``confidence``.
        confidence : float
            That model's confidence on the current sample.

        Returns
        -------
        dict[str, float]
            Predicted accuracy per reachable model, including ``model`` itself.
            Empty when the node was never seen while building the graph.

        Raises
        ------
        ValueError
            If ``model`` is not in the graph.

        """
        if model not in self._models:
            err_msg = f"Unknown model {model!r}, expected one of {self._models}"
            raise ValueError(err_msg)
        return dict(self._predictions.get((model, self.bin(confidence)), {}))

    def _predict_node(self: Self, source: Node) -> dict[str, float]:
        # dijkstra with a cutoff at the distance threshold
        threshold = self._distance_threshold
        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, cost in self._edges.get(u, {}).items():
                nd = d + cost
                if nd <= threshold and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        grouped: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
        for node, d in dist.items():
            accs, weights = grouped[node[0]]
            accs.append(self._nodes[node])
            weights.append(max(((threshold - d) / threshold) ** 2, 1e-8))
        return {
            model: float(np.average(grouped[model][0], weights=grouped[model][1]))
            for model in self._models
            if model in grouped
        }
