# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Performance estimation for partition plans.

Classes
-------
PerformanceEstimator
    Estimates accuracy and latency for candidate partitions.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._plan import PartitionPlan


class PerformanceEstimator:
    """Estimates accuracy and latency for partition plans without execution."""

    def __init__(self: Self, profiles: dict[str, dict]) -> None:
        """
        Initialize the estimator.

        Parameters
        ----------
        profiles : dict[str, dict]
            Network profiles from NNProfiler.

        """
        self.profiles = profiles

    def estimate_single(
        self: Self,
        detector_name: str,
        distribution: np.ndarray,
    ) -> tuple[float, float]:
        """
        Estimate performance for single detector on given distribution.

        Parameters
        ----------
        detector_name : str
            Name of detector to use.
        distribution : np.ndarray
            Object size distribution vector.

        Returns
        -------
        tuple[float, float]
            Estimated accuracy (eAP) and latency in seconds.

        """
        if detector_name not in self.profiles:
            err_msg = f"Detector {detector_name} not found in profiles"
            raise ValueError(err_msg)

        profile = self.profiles[detector_name]
        acc_vector = np.array(profile["acc_vector"])
        latency = profile["latency"]

        # Estimate accuracy as dot product
        eap = float(np.dot(acc_vector, distribution))

        return eap, latency

    def estimate_plan(
        self: Self,
        plan: PartitionPlan,
        block_distributions: list[np.ndarray] | None = None,
        global_distribution: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """
        Estimate performance for a partition plan.

        Parameters
        ----------
        plan : PartitionPlan
            Partition plan to estimate.
        block_distributions : list[np.ndarray], optional
            Distribution for each block, by default None.
        global_distribution : np.ndarray, optional
            Global distribution to use if block_distributions not provided.

        Returns
        -------
        tuple[float, float]
            Estimated accuracy and total latency.

        """
        if block_distributions is None:
            if global_distribution is None:
                # Use uniform distribution as fallback
                num_bins = 12
                global_distribution = np.ones(num_bins) / num_bins

            block_distributions = [global_distribution] * len(plan.blocks)

        if len(block_distributions) != len(plan.blocks):
            err_msg = (
                f"Number of distributions ({len(block_distributions)}) "
                f"must match number of blocks ({len(plan.blocks)})"
            )
            raise ValueError(err_msg)

        # Compute weighted accuracy across blocks
        total_eap = 0.0
        total_area = sum(block.area for block in plan.blocks)

        # Track unique detectors for latency
        detector_names = set()

        for block, dist in zip(plan.blocks, block_distributions):
            if block.detector_name is None:
                continue

            # Weight by block area (object density)
            weight = block.area / total_area if total_area > 0 else 0

            eap, _ = self.estimate_single(block.detector_name, dist)
            total_eap += eap * weight

            detector_names.add(block.detector_name)

        # Estimate total latency (sum of unique detectors)
        # Note: This assumes no batching or parallel execution
        total_lat = sum(
            self.profiles[name]["latency"] for name in detector_names
        )

        return float(total_eap), float(total_lat)

    def estimate_block_latency(
        self: Self,
        detector_name: str,
        num_blocks: int = 1,
    ) -> float:
        """
        Estimate latency for running detector on blocks.

        Parameters
        ----------
        detector_name : str
            Name of detector.
        num_blocks : int, optional
            Number of blocks using this detector, by default 1.

        Returns
        -------
        float
            Estimated latency in seconds.

        """
        if detector_name not in self.profiles:
            err_msg = f"Detector {detector_name} not found in profiles"
            raise ValueError(err_msg)

        # Without batching, latency scales linearly with number of blocks
        base_latency = self.profiles[detector_name]["latency"]
        return base_latency * num_blocks

    def get_detector_latency(self: Self, detector_name: str) -> float:
        """
        Get latency for a detector.

        Parameters
        ----------
        detector_name : str
            Name of detector.

        Returns
        -------
        float
            Latency in seconds.

        """
        if detector_name not in self.profiles:
            err_msg = f"Detector {detector_name} not found in profiles"
            raise ValueError(err_msg)

        return self.profiles[detector_name]["latency"]

    def get_detector_accuracy(
        self: Self,
        detector_name: str,
    ) -> np.ndarray:
        """
        Get accuracy vector for a detector.

        Parameters
        ----------
        detector_name : str
            Name of detector.

        Returns
        -------
        np.ndarray
            Accuracy vector.

        """
        if detector_name not in self.profiles:
            err_msg = f"Detector {detector_name} not found in profiles"
            raise ValueError(err_msg)

        return np.array(self.profiles[detector_name]["acc_vector"])

