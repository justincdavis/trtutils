# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Remix: Flexible High-resolution Object Detection on Edge Devices.

This module implements the Remix system from the ACM MobiCom '21 paper,
providing adaptive non-uniform image partitioning and selective execution
for efficient object detection under latency constraints.

Classes
-------
RemixSystem
    Main Remix system integrating all components.
NNProfiler
    Profile detectors for latency and accuracy.
ObjectDistributionExtractor
    Extract object size distributions from frames.
PerformanceEstimator
    Estimate performance of partition plans.
AdaptivePartitionPlanner
    Generate partition plans with recursive subdivision.
PIDController
    Standard PID controller for feedback control.
PlanController
    Control plan selection using PID feedback.
SelectiveExecutor
    Execute partitions with AIMD-based block skipping.
PartitionBlock
    Represents an image region.
PartitionPlan
    Contains blocks and detector assignments.

"""

from __future__ import annotations

from ._controller import PIDController, PlanController
from ._distribution import ObjectDistributionExtractor
from ._estimator import PerformanceEstimator
from ._executor import SelectiveExecutor
from ._plan import PartitionBlock, PartitionPlan
from ._planner import AdaptivePartitionPlanner
from ._profiler import NNProfiler
from ._remix import RemixSystem

__all__ = [
    "AdaptivePartitionPlanner",
    "NNProfiler",
    "ObjectDistributionExtractor",
    "PartitionBlock",
    "PartitionPlan",
    "PerformanceEstimator",
    "PIDController",
    "PlanController",
    "RemixSystem",
    "SelectiveExecutor",
]

