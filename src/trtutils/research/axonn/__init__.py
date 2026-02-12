# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementation for the research paper AxoNN.

AxoNN finds optimal layer-to-accelerator mappings for neural network inference
on heterogeneous SoCs (e.g., NVIDIA Jetson with GPU + DLA) by solving a
constrained optimization problem.

**Goal**: Minimize execution time while staying under an Energy Consumption Target (ECT).

Reference
---------
AxoNN: Energy-Aware Execution of Neural Network Inference on Multi-Accelerator
Heterogeneous SoCs (DAC 2022)
https://doi.org/10.1145/3489517.3530572

Example:
-------
>>> from trtutils.builder import ImageBatcher
>>> from trtutils.research.axonn import build_axonn_engine, AxoNNConfig
>>>
>>> # Create calibration batcher
>>> batcher = ImageBatcher(
...     image_dir="calibration_images/",
...     shape=(640, 640, 3),
...     dtype=np.float32,
... )
>>>
>>> # Build optimized engine
>>> schedule = build_axonn_engine(
...     onnx="model.onnx",
...     output="axonn.engine",
...     calibration_batcher=batcher,
...     verbose=True,
... )
>>>
>>> # Or with custom config
>>> config = AxoNNConfig(
...     energy_target_ratio=0.7,  # Target 70% of GPU energy
...     max_transitions=2,
...     dla_core=0,
... )
>>> schedule = build_axonn_engine(
...     onnx="model.onnx",
...     output="axonn.engine",
...     calibration_batcher=batcher,
...     config=config,
... )

"""

from __future__ import annotations

from ._build import build_axonn_engine, optimize_and_build
from ._cost import (
    compute_dla_only_costs,
    compute_gpu_only_costs,
    compute_layer_energy,
    compute_layer_time,
    compute_total_energy,
    compute_total_time,
    create_dla_preferred_schedule,
    create_gpu_only_schedule,
    estimate_transition_cost,
)
from ._profile import extract_layer_info, profile_for_axonn
from ._solver import find_optimal_schedule, solve_schedule, solve_schedule_greedy
from ._types import AxoNNConfig, Layer, LayerCost, ProcessorType, Schedule, TransitionCost

__all__ = [
    "AxoNNConfig",
    "Layer",
    "LayerCost",
    "ProcessorType",
    "Schedule",
    "TransitionCost",
    "build_axonn_engine",
    "compute_dla_only_costs",
    "compute_gpu_only_costs",
    "compute_layer_energy",
    "compute_layer_time",
    "compute_total_energy",
    "compute_total_time",
    "create_dla_preferred_schedule",
    "create_gpu_only_schedule",
    "estimate_transition_cost",
    "extract_layer_info",
    "find_optimal_schedule",
    "optimize_and_build",
    "profile_for_axonn",
    "solve_schedule",
    "solve_schedule_greedy",
]
