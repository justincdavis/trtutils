# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementation of the research paper DRIFT.

DRIFT parallelizes object detection within a frame across the accelerators of an
SoC. The region holding the high-priority objects (by class) of the previous frame
is cropped and detected on the GPU; the rest of the frame is packed into a small
image and detected by a second, cheaper model, concurrently on a DLA or serially
on the GPU. A runtime scheduler picks both models per frame and falls back to a
baseline model on the full frame when splitting is not expected to pay off.

**Offline**: :func:`characterize` measures every model's end-to-end latency (alone
and concurrently with every DLA model), energy, and per-image confidences and
accuracy on labeled images.

**Online**: :class:`Drift` finds the high-priority region (:func:`hp_region`), packs
the low-priority remainder (:class:`FramePacker`), and runs the models chosen by
:class:`DriftScheduler`, whose :class:`LinearEstimator` predicts every model's
accuracy from the confidence of the model that ran.

:class:`FramePacker`, :class:`LinearEstimator` and :class:`DriftScheduler` work on
their own, from plain arrays and characterization data.

Reference
---------
DRIFT: Dynamic Intra-Frame Parallelization for Fast Continuous Object Detection on
Multi-Accelerator SoCs (ICCAD 2026), Justin Davis, Ismet Dagli and Mehmet E. Belviranli.
https://doi.org/10.1145/3831252.3834218

Examples
--------
>>> from trtutils.image import Detector
>>> from trtutils.research.drift import Drift, characterize
>>>
>>> models = {
...     "yolov10m_1280": lambda: Detector("yolov10m_1280.engine"),
...     "yolov10n_640": lambda: Detector("yolov10n_640.engine"),
...     "yoloxs_480_dla": lambda: Detector(
...         "yoloxs_480_dla.engine", dla_core=0, cuda_graph=False, input_range=(0, 255)
...     ),
... }
>>> traits = characterize(models, images, ground_truth)
>>> drift = Drift(models, traits, priority_classes={0})
>>> for frame in frames:
...     detections = drift.end2end(frame)
>>> drift.close()

"""

from __future__ import annotations

from ._characterize import characterize
from ._drift import Drift
from ._estimator import LinearEstimator
from ._packer import FramePacker, cell_probabilities, dense_groups, shelf_pack
from ._scheduler import DriftScheduler, hp_region
from ._types import (
    DriftConfig,
    FrameResult,
    ModelTraits,
    Schedule,
    load_characterization,
    save_characterization,
)

__all__ = [
    "Drift",
    "DriftConfig",
    "DriftScheduler",
    "FramePacker",
    "FrameResult",
    "LinearEstimator",
    "ModelTraits",
    "Schedule",
    "cell_probabilities",
    "characterize",
    "dense_groups",
    "hp_region",
    "load_characterization",
    "save_characterization",
    "shelf_pack",
]
