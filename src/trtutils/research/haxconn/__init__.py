# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementation of the research paper HaX-CoNN.

HaX-CoNN runs several DNNs concurrently on a GPU and a DLA. Each DNN is split into
layer groups at the points where it can move between accelerators, and every group is
mapped to one accelerator so that the slowest DNN finishes first (or the summed frame
rate is highest), accounting for the cost of moving between accelerators and the
slowdown that DNNs on the two accelerators cause each other through shared memory.

**Offline**: :func:`characterize` splits one DNN into layer groups
(:func:`group_layers`) and measures every group's time, transition costs and memory
throughput on the GPU and the DLA, on engines sliced out of the network
(:func:`slice_hook`). :func:`calibrate` measures how the GPU and the DLA slow each other
down, from which :class:`PCCSModel` predicts any slowdown.

**Online**: :class:`HaxConnScheduler` finds the schedule: Z3 proposes accelerator
mappings and :func:`simulate` scores them, until no better mapping exists.
:class:`HaxConn` builds one mixed GPU/DLA engine per DNN (:func:`build_engines`) and runs
the DNNs concurrently; with ``dynamic=True`` it starts on a naive schedule and switches
to better ones as the solver finds them (D-HaX-CoNN).

:func:`group_layers`, :class:`PCCSModel`, :func:`simulate` and
:class:`HaxConnScheduler` work on their own, from plain layer lists and traits.

Reference
---------
HaX-CoNN: Shared Memory-Contention-Aware Concurrent DNN Execution for Diversely
Heterogeneous System-on-Chips (PPoPP 2024), Ismet Dagli and Mehmet E. Belviranli.
https://arxiv.org/abs/2308.05869

Examples
--------
>>> from trtutils.research.haxconn import HaxConn, PCCSModel, calibrate, characterize
>>>
>>> models = {"resnet18": "resnet18.onnx", "googlenet": "googlenet.onnx"}
>>> traits = {name: characterize(onnx, memory=True) for name, onnx in models.items()}
>>> contention = {acc: PCCSModel.from_calibration(c) for acc, c in calibrate().items()}
>>> runtime = HaxConn(models, traits, contention)
>>> outputs = runtime.mock_execute()
>>> runtime.close()

"""

from __future__ import annotations

from ._build import build_engines, nominal_ranges_hook, slice_hook
from ._characterize import calibrate, characterize, measure_memory
from ._groups import LayerGroup, fused_spans, group_layers, transition_points
from ._haxconn import HaxConn
from ._pccs import PCCSModel, fit_pccs
from ._scheduler import HaxConnScheduler
from ._simulate import simulate
from ._types import (
    Calibration,
    GroupTraits,
    HaxConnConfig,
    ModelTraits,
    Schedule,
    load_calibration,
    load_schedule,
    load_traits,
    save_calibration,
    save_schedule,
    save_traits,
)

__all__ = [
    "Calibration",
    "GroupTraits",
    "HaxConn",
    "HaxConnConfig",
    "HaxConnScheduler",
    "LayerGroup",
    "ModelTraits",
    "PCCSModel",
    "Schedule",
    "build_engines",
    "calibrate",
    "characterize",
    "fit_pccs",
    "fused_spans",
    "group_layers",
    "load_calibration",
    "load_schedule",
    "load_traits",
    "measure_memory",
    "nominal_ranges_hook",
    "save_calibration",
    "save_schedule",
    "save_traits",
    "simulate",
    "slice_hook",
    "transition_points",
]
