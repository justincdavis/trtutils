# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Synthetic HaX-CoNN traits and contention models shared by the CPU tests."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from trtutils.research.haxconn import GroupTraits, ModelTraits, PCCSModel, _build, _haxconn
from trtutils.research.haxconn._build import engine_paths

# the PCCS paper's Xavier AGX fits (Table 7)
XAVIER = {
    "gpu": PCCSModel(38.1, 96.2, 4.9, 45.3, 87.2, 1.11),
    "dla": PCCSModel(0.0, 27.9, 0.0, 71.1, 22.1, 0.35),
}


def make_traits(
    gpu_ms,
    dla_ms,
    to_dla=None,
    to_gpu=None,
    gpu_memory=None,
    dla_memory=None,
) -> ModelTraits:
    """One-layer groups; a None DLA time makes a GPU-only group."""
    count = len(gpu_ms)
    to_dla = to_dla if to_dla is not None else [0.0] * count
    to_gpu = to_gpu if to_gpu is not None else [0.0] * count
    groups = []
    for i in range(count):
        time_ms = {"gpu": [gpu_ms[i]]}
        memory = {}
        if gpu_memory is not None:
            memory["gpu"] = [gpu_memory[i]]
        if dla_ms[i] is not None:
            time_ms["dla"] = [dla_ms[i]]
            if dla_memory is not None:
                memory["dla"] = [dla_memory[i]]
        groups.append(
            GroupTraits(
                first=i,
                last=i,
                time_ms=time_ms,
                transition_in_ms={"gpu": to_gpu[i], "dla": to_dla[i]},
                transition_out_ms={"gpu": 0.0, "dla": 0.0},
                memory=memory,
            )
        )
    return ModelTraits(groups)


@pytest.fixture
def paper_traits() -> dict[str, ModelTraits]:
    """GoogleNet from the HaX-CoNN paper's Table 2 and AlexNet from the reference solver."""
    googlenet = make_traits(
        [0.45, 0.19, 0.31, 0.18, 0.16, 0.17, 0.21, 0.25, 0.16, 0.24],
        [0.75, 0.34, 0.45, 0.37, 0.31, 0.33, 0.31, 0.35, 0.27, 0.36],
        to_dla=[0.0, 0.056, 0.075, 0.062, 0.011, 0.055, 0.024, 0.058, 0.030, 0.024],
        to_gpu=[0.0, 0.15, 0.13, 0.08, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07],
        gpu_memory=[41.97, 62.21, 78.49, 53.41, 55.70, 59.24, 62.60, 76.12, 66.95, 47.96],
        dla_memory=[30, 45, 55, 38, 40, 42, 45, 54, 48, 34],
    )
    alexnet = make_traits(
        [0.15, 0.06, 0.22, 0.08, 0.16, 0.70, 0.32, 0.10],
        [0.41, 0.20, 0.45, 0.22, 0.41, 1.91, 0.68, 0.13],
        to_dla=[0.0, 0.03, 0.11, 0.08, 0.03, 0.09, 0.08, 0.07],
        to_gpu=[0.0, 0.07, 0.14, 0.10, 0.05, 0.16, 0.12, 0.09],
        gpu_memory=[50, 68, 71, 49, 53, 62, 55, 48],
        dla_memory=[38, 49, 43, 37, 27, 39, 32, 30],
    )
    return {"googlenet": googlenet, "alexnet": alexnet}


class _FakeEngine:
    """Stands in for TRTEngine: sleeps briefly per run."""

    _capture_lock = threading.Lock()

    def __init__(self, path, **_kwargs: object) -> None:
        self.path = Path(path)

    def execute(self, _data) -> list[np.ndarray]:
        time.sleep(0.001)
        return [np.zeros(1, dtype=np.float32)]

    def get_random_input(self) -> list[np.ndarray]:
        return [np.zeros(1, dtype=np.float32)]


@pytest.fixture
def fakes(monkeypatch):
    """Fake engines and builds for the runtime; returns the loaded and built engine names."""
    loads: list[str] = []
    builds: list[str] = []

    class _Loaded(_FakeEngine):
        def __init__(self, path, **kwargs: object) -> None:
            super().__init__(path, **kwargs)
            loads.append(self.path.name)

    def build(models, traits, schedule, output_dir, **kwargs: object):
        paths = engine_paths(models, traits, schedule, output_dir, kwargs.get("data_batchers"))
        for path in paths.values():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                builds.append(path.name)
        return paths

    def build_one(_onnx, _traits, _accs, path, *_args: object, **_kwargs: object):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        builds.append(path.name)

    monkeypatch.setattr(_haxconn, "TRTEngine", _Loaded)
    monkeypatch.setattr(_haxconn, "build_engines", build)
    monkeypatch.setattr(_build, "_build_one", build_one)
    return loads, builds
