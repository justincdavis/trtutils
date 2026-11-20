# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementation of the research paper SHIFT.

SHIFT runs several object detectors, each compiled for one or more accelerators
(GPU, DLA), and picks the (model, accelerator) pair for every frame from the scene
context, trading accuracy against energy and latency.

**Offline**: :func:`characterize` measures each pair's latency, energy, load cost,
and per-image confidences and accuracy on labeled images.

**Online**: :class:`ShiftScheduler` keeps the current pair while the scene is
stable and the model is confident; otherwise a :class:`ConfidenceGraph` predicts
every pair's accuracy from the current confidence and the best weighted score wins.
:class:`Shift` runs the scheduled pair per frame and loads pairs on demand under a
memory budget.

:class:`ConfidenceGraph` works on its own, from any per-sample confidences and
accuracies.

Reference
---------
Context-aware Multi-Model Object Detection for Diversely Heterogeneous Compute
Systems (DATE 2024), Justin Davis and Mehmet E. Belviranli.
https://arxiv.org/abs/2402.07415

Examples
--------
>>> from trtutils.image import Detector
>>> from trtutils.research.shift import Shift, characterize
>>>
>>> models = {
...     "yolov10n_gpu": lambda: Detector("yolov10n.engine"),
...     "yolov10n_dla": lambda: Detector("yolov10n_dla.engine", dla_core=0, cuda_graph=False),
...     "yolov10m_gpu": lambda: Detector("yolov10m.engine"),
... }
>>> traits = characterize(models, images, ground_truth)
>>> shift = Shift(models, traits)
>>> for frame in frames:
...     detections = shift.end2end(frame)

"""

from __future__ import annotations

from ._characterize import characterize
from ._graph import ConfidenceGraph
from ._scheduler import ShiftScheduler, ncc
from ._shift import Shift
from ._types import (
    ModelTraits,
    ShiftConfig,
    load_characterization,
    reduce_values,
    save_characterization,
)

__all__ = [
    "ConfidenceGraph",
    "ModelTraits",
    "Shift",
    "ShiftConfig",
    "ShiftScheduler",
    "characterize",
    "load_characterization",
    "ncc",
    "reduce_values",
    "save_characterization",
]
