# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementation for the research paper HaX-CoNN.

HaX-CoNN is a contention-aware multi-accelerator execution scheme for
concurrent DNNs on shared-memory heterogeneous SoCs (e.g., NVIDIA Jetson
with GPU + DLA). It models shared memory contention via the PCCS model
and optimizes scheduling across GPU and DLA for multiple concurrent DNNs.

**Goals**: Maximize throughput (minimize sum of DNN latencies) or minimize
the maximum DNN latency across all concurrent models.

Reference
---------
HaX-CoNN: Shared Memory Multi-Accelerator Execution of Concurrent DNN
Workloads (PPoPP 2024)

Example:
-------
>>> from trtutils.builder import ImageBatcher
>>> from trtutils.research.haxconn import build_engines, Objective
>>>
>>> batcher_a = ImageBatcher("calib_a/", (640, 640, 3), np.float32)
>>> batcher_b = ImageBatcher("calib_b/", (640, 640, 3), np.float32)
>>>
>>> schedule, executor = build_engines(
...     models=[("model_a.onnx", "model_a.engine"),
...             ("model_b.onnx", "model_b.engine")],
...     calibration_batchers=[batcher_a, batcher_b],
...     objective=Objective.MAX_THROUGHPUT,
...     verbose=True,
... )
>>>
>>> results = executor.execute([input_a, input_b])

"""

from __future__ import annotations

from ._build import build_engines
from ._executor import HaxconnExecutor
from ._types import Objective

__all__ = [
    "HaxconnExecutor",
    "Objective",
    "build_engines",
]
