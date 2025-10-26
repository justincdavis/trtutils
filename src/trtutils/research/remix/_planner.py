# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Adaptive partition planning with recursive subdivision.

Classes
-------
AdaptivePartitionPlanner
    Generates candidate partition plans using dynamic programming.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG

from ._estimator import PerformanceEstimator
from ._plan import PartitionBlock, PartitionPlan

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface


class AdaptivePartitionPlanner:
    """Generates partition plans using recursive subdivision and pruning."""

    def __init__(
        self: Self,
        detectors: list[DetectorInterface],
        profiles: dict[str, dict],
        estimator: PerformanceEstimator,
    ) -> None:
        """
        Initialize the partition planner.

        Parameters
        ----------
        detectors : list[DetectorInterface]
            Available detectors.
        profiles : dict[str, dict]
            Network profiles.
        estimator : PerformanceEstimator
            Performance estimator.

        """
        self.detectors = detectors
        self.profiles = profiles
        self.estimator = estimator
        self.detector_by_name = {d.name: d for d in detectors}

    def generate(
        self: Self,
        view_shape: tuple[int, int],
        distribution: np.ndarray,
        latency_budget: float,
        max_plans: int = 10,
        prune_threshold: float = 0.001,
        max_latency: float = 10.0,
        *,
        verbose: bool = False,
    ) -> list[PartitionPlan]:
        """
        Generate partition plans for given view and budget.

        Parameters
        ----------
        view_shape : tuple[int, int]
            Image dimensions as (width, height).
        distribution : np.ndarray
            Object size distribution.
        latency_budget : float
            Target latency budget in seconds.
        max_plans : int, optional
            Maximum number of plans to keep, by default 10.
        prune_threshold : float, optional
            Latency difference threshold for pruning, by default 0.001 (1ms).
        max_latency : float, optional
            Maximum latency cutoff in seconds, by default 10.0.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        list[PartitionPlan]
            List of candidate partition plans.

        """
        if verbose:
            LOG.info(
                f"Generating partition plans for {view_shape} "
                f"with budget {latency_budget*1000:.2f}ms",
            )

        plans = []

        # Generate single-detector full-frame plans
        full_plans = self._generate_full_frame_plans(
            view_shape,
            distribution,
            max_latency,
            verbose=verbose,
        )
        plans.extend(full_plans)

        # Generate subdivided plans
        subdivided = self._generate_subdivided_plans(
            view_shape,
            distribution,
            latency_budget,
            max_latency,
            verbose=verbose,
        )
        plans.extend(subdivided)

        # Prune similar plans
        plans = self._prune_plans(plans, prune_threshold)

        # Sort and filter by budget and accuracy
        plans = self._select_best_plans(
            plans,
            latency_budget,
            max_plans,
        )

        # Assign plan IDs
        for i, plan in enumerate(plans):
            plan.plan_id = i

        if verbose:
            LOG.info(f"Generated {len(plans)} plans")
            for plan in plans:
                LOG.debug(
                    f"Plan {plan.plan_id}: {plan.num_blocks} blocks, "
                    f"eAP={plan.est_ap:.3f}, lat={plan.est_lat*1000:.2f}ms",
                )

        return plans

    def _generate_full_frame_plans(
        self: Self,
        view_shape: tuple[int, int],
        distribution: np.ndarray,
        max_latency: float,
        *,
        verbose: bool = False,
    ) -> list[PartitionPlan]:
        """Generate plans using single detector on full frame."""
        plans = []
        w, h = view_shape

        for det in self.detectors:
            eap, lat = self.estimator.estimate_single(det.name, distribution)

            if lat > max_latency:
                continue

            block = PartitionBlock(0, 0, w, h, detector_name=det.name)
            plan = PartitionPlan(
                blocks=[block],
                est_ap=eap,
                est_lat=lat,
            )
            plans.append(plan)

            if verbose:
                LOG.debug(
                    f"Full-frame plan: {det.name}, "
                    f"eAP={eap:.3f}, lat={lat*1000:.2f}ms",
                )

        return plans

    def _generate_subdivided_plans(
        self: Self,
        view_shape: tuple[int, int],
        distribution: np.ndarray,
        latency_budget: float,
        max_latency: float,
        *,
        verbose: bool = False,
    ) -> list[PartitionPlan]:
        """Generate plans with subdivided blocks."""
        plans = []
        w, h = view_shape

        # Try different subdivision patterns
        subdivisions = [
            (2, 2),  # 2x2 grid
            (3, 3),  # 3x3 grid
            (2, 1),  # Horizontal split
            (1, 2),  # Vertical split
            (4, 4),  # 4x4 grid
        ]

        for rows, cols in subdivisions:
            # Generate grid
            blocks = self._create_grid(view_shape, rows, cols)

            # Try different detector combinations
            combos = self._generate_detector_combinations(
                len(blocks),
                latency_budget,
            )

            for combo in combos:
                if len(combo) != len(blocks):
                    continue

                # Create plan
                plan_blocks = []
                for block_coords, det_name in zip(blocks, combo):
                    x1, y1, x2, y2 = block_coords
                    plan_blocks.append(
                        PartitionBlock(x1, y1, x2, y2, detector_name=det_name),
                    )

                # Estimate performance
                eap, lat = self.estimator.estimate_plan(
                    PartitionPlan(
                        blocks=plan_blocks,
                        est_ap=0.0,
                        est_lat=0.0,
                    ),
                    global_distribution=distribution,
                )

                if lat <= max_latency:
                    plan = PartitionPlan(
                        blocks=plan_blocks,
                        est_ap=eap,
                        est_lat=lat,
                    )
                    plans.append(plan)

        return plans

    def _create_grid(
        self: Self,
        view_shape: tuple[int, int],
        rows: int,
        cols: int,
    ) -> list[tuple[int, int, int, int]]:
        """Create uniform grid of blocks."""
        w, h = view_shape
        blocks = []

        block_w = w // cols
        block_h = h // rows

        for r in range(rows):
            for c in range(cols):
                x1 = c * block_w
                y1 = r * block_h
                x2 = (c + 1) * block_w if c < cols - 1 else w
                y2 = (r + 1) * block_h if r < rows - 1 else h

                blocks.append((x1, y1, x2, y2))

        return blocks

    def _generate_detector_combinations(
        self: Self,
        num_blocks: int,
        latency_budget: float,
    ) -> list[list[str]]:
        """
        Generate viable detector combinations for blocks.

        For simplicity, we use homogeneous assignments (all blocks use same detector)
        and a few heterogeneous patterns.
        """
        combos = []

        # Homogeneous: all blocks use same detector
        for det in self.detectors:
            lat = self.estimator.get_detector_latency(det.name)
            if lat * num_blocks <= latency_budget * 1.5:
                combos.append([det.name] * num_blocks)

        # Heterogeneous: try mixing fast and slow detectors
        if len(self.detectors) >= 2:
            # Sort by latency
            sorted_dets = sorted(
                self.detectors,
                key=lambda d: self.estimator.get_detector_latency(d.name),
            )

            fast_det = sorted_dets[0].name
            slow_det = sorted_dets[-1].name

            # Mostly fast with one slow
            if num_blocks > 1:
                combo = [fast_det] * (num_blocks - 1) + [slow_det]
                combos.append(combo)

            # Half and half
            if num_blocks >= 4:
                half = num_blocks // 2
                combo = [slow_det] * half + [fast_det] * (num_blocks - half)
                combos.append(combo)

        return combos

    def _prune_plans(
        self: Self,
        plans: list[PartitionPlan],
        threshold: float,
    ) -> list[PartitionPlan]:
        """Prune plans with similar latency."""
        if len(plans) == 0:
            return []

        # Sort by latency
        sorted_plans = sorted(plans, key=lambda p: p.est_lat)

        pruned = [sorted_plans[0]]
        for plan in sorted_plans[1:]:
            # Keep if latency differs significantly from last kept plan
            if abs(plan.est_lat - pruned[-1].est_lat) >= threshold:
                pruned.append(plan)

        return pruned

    def _select_best_plans(
        self: Self,
        plans: list[PartitionPlan],
        latency_budget: float,
        max_plans: int,
    ) -> list[PartitionPlan]:
        """Select best plans based on budget and accuracy."""
        if len(plans) == 0:
            return []

        # Sort by: within budget first, then by accuracy (descending)
        sorted_plans = sorted(
            plans,
            key=lambda p: (
                p.est_lat > latency_budget,  # False (within budget) comes first
                -p.est_ap,  # Higher accuracy first
            ),
        )

        return sorted_plans[:max_plans]

    def save_plans(self: Self, plans: list[PartitionPlan], path: Path | str) -> None:
        """
        Save partition plans to JSON file.

        Parameters
        ----------
        plans : list[PartitionPlan]
            Plans to save.
        path : Path | str
            Output file path.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [plan.to_dict() for plan in plans]

        with path.open("w") as f:
            json.dump(data, f, indent=2)

        LOG.info(f"Saved {len(plans)} plans to {path}")

    @classmethod
    def load_plans(cls: type[Self], path: Path | str) -> list[PartitionPlan]:
        """
        Load partition plans from JSON file.

        Parameters
        ----------
        path : Path | str
            Input file path.

        Returns
        -------
        list[PartitionPlan]
            Loaded plans.

        """
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        plans = [PartitionPlan.from_dict(p) for p in data]

        LOG.info(f"Loaded {len(plans)} plans from {path}")
        return plans

