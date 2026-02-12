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
>>> from trtutils.research.axonn import build_engine
>>>
>>> batcher = ImageBatcher(
...     image_dir="calibration_images/",
...     shape=(640, 640, 3),
...     dtype=np.float32,
... )
>>>
>>> time_ms, energy_mj, transitions, gpu_layers, dla_layers = build_engine(
...     onnx="model.onnx",
...     output="model.engine",
...     calibration_batcher=batcher,
...     energy_ratio=0.8,      # Target 80% of GPU energy (ECT)
...     max_transitions=3,     # Max GPU<->DLA transitions
...     verbose=True,
... )
>>>
>>> print(f"Time: {time_ms:.2f}ms, Energy: {energy_mj:.2f}mJ")

"""

from __future__ import annotations

from ._build import build_engine

__all__ = [
    "build_engine",
]
