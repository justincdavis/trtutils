# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/shift/_shift.py -- the runtime and dynamic model loader."""

from __future__ import annotations

import gc
import threading
import time
import weakref

import numpy as np
import pytest

from trtutils.research.shift import ModelTraits, Shift, ShiftConfig

# energy-only scoring: every pair is predicted from every node, so the cheapest (small) wins
CONFIG = ShiftConfig(
    accuracy_threshold=0.01, accuracy_weight=0.0, energy_weight=1.0, latency_weight=0.0
)
BASE = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
SPECS = {
    # name: (confidence, iou, energy, memory)
    "big": (0.9, 0.9, 100.0, 100.0),
    "mid": (0.5, 0.6, 50.0, 50.0),
    "small": (0.2, 0.3, 10.0, 10.0),
}
TRAITS = {
    name: ModelTraits(
        latency_ms=energy / 10.0,
        energy_mj=energy,
        load_time_s=0.1,
        load_memory_mb=memory,
        confidences=[[conf]],
        ious=[[iou]],
    )
    for name, (conf, iou, energy, memory) in SPECS.items()
}


class FakeDetector:
    """Returns one full-frame detection at its pair's characterized confidence."""

    def __init__(self, name) -> None:
        """Remember which pair this detector stands in for."""
        self.name = name

    def end2end(self, image):
        return [((0, 0, image.shape[1], image.shape[0]), SPECS[self.name][0], 0)]


class Factories:
    """Counting detector factories; gated names block until released."""

    def __init__(self, gated=(), failing=()) -> None:
        """Set up call counters, gates and failures."""
        self.calls = dict.fromkeys(SPECS, 0)
        self.built = {}
        self.gates = {name: threading.Event() for name in gated}
        self.failing = set(failing)

    def make(self, name):
        def _factory():
            self.calls[name] += 1
            if name in self.gates:
                self.gates[name].wait(timeout=10)
            if name in self.failing:
                err_msg = f"cannot build {name}"
                raise RuntimeError(err_msg)
            detector = FakeDetector(name)
            self.built[name] = weakref.ref(detector)
            return detector

        return _factory

    def models(self):
        return {name: self.make(name) for name in SPECS}


def _frame(index):
    # consecutive frames are inverses (NCC -1), so the scene-stability gate never holds
    return BASE if index % 2 == 0 else 255 - BASE


def _run_until(shift, condition, start=0):
    # step frames until condition holds; background loads finish within the deadline
    deadline = time.monotonic() + 10.0
    seed = start
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        shift.end2end(_frame(seed))
        seed += 1
        time.sleep(0.001)
    return seed


@pytest.mark.cpu
def test_shift_without_budget() -> None:
    """Without a budget every pair is built once, the most accurate runs first, and switches apply on the next frame."""
    factories = Factories()
    shift = Shift(factories.models(), TRAITS, CONFIG)
    assert factories.calls == {"big": 1, "mid": 1, "small": 1}
    assert shift.current == "big"
    assert shift.last is None
    detections = shift.end2end(_frame(0))
    assert detections[0][1] == SPECS["big"][0]
    assert (shift.last, shift.current) == ("big", "small")
    assert shift.end2end(_frame(1))[0][1] == SPECS["small"][0]
    assert shift.last == "small"
    assert factories.calls == {"big": 1, "mid": 1, "small": 1}
    shift.close()
    assert shift.loaded == []


@pytest.mark.cpu
def test_shift_lazy_load_and_eviction() -> None:
    """With a budget, pairs load in the background while the current pair serves, and LRU pairs are evicted."""
    factories = Factories(gated=["small"])
    shift = Shift(factories.models(), TRAITS, CONFIG, memory_budget=150.0)
    assert factories.calls == {"big": 1, "mid": 0, "small": 0}
    assert shift.loaded == ["big"]

    # small is requested but blocked: big keeps serving
    for seed in range(3):
        assert shift.end2end(_frame(seed))[0][1] == SPECS["big"][0]
        assert shift.current == "big"
    assert factories.calls["small"] == 1

    factories.gates["small"].set()
    _run_until(shift, lambda: shift.current == "small", start=3)
    assert shift.loaded == ["big", "small"]

    # mid needs 50 MB more than the 150 MB budget allows: big (least recent) is evicted
    shift._load("mid")
    assert shift.loaded == ["small"]
    gc.collect()
    assert factories.built["big"]() is None
    _run_until(shift, lambda: "mid" in shift.loaded, start=10)
    assert shift.current == "small"
    assert shift.loaded == ["mid", "small"]

    # bringing big back evicts mid and rebuilds big
    shift._load("big")
    _run_until(shift, lambda: "big" in shift.loaded, start=20)
    assert shift.loaded == ["big", "small"]
    assert factories.calls == {"big": 2, "mid": 1, "small": 1}


@pytest.mark.cpu
def test_shift_stale_load_gives_way_to_target() -> None:
    """A finished load the scheduler no longer wants is admitted, then evicted for the current target."""
    factories = Factories(gated=["mid"])
    shift = Shift(factories.models(), TRAITS, CONFIG, memory_budget=150.0)
    shift._load("mid")
    # the target is small, but only one load runs at a time
    for index in range(3):
        shift.end2end(_frame(index))
    assert factories.calls["small"] == 0
    factories.gates["mid"].set()
    _run_until(shift, lambda: shift.current == "small", start=3)
    # big (current while mid loaded) and small fill the budget; mid made room
    assert shift.loaded == ["big", "small"]
    assert factories.calls == {"big": 1, "mid": 1, "small": 1}
    shift.close()


@pytest.mark.cpu
def test_shift_failed_load_raises_from_end2end() -> None:
    """A background factory error is raised by the next end2end call."""
    factories = Factories(failing=["small"])
    shift = Shift(factories.models(), TRAITS, CONFIG, memory_budget=150.0)
    with pytest.raises(RuntimeError, match="cannot build small"):
        _run_until(shift, lambda: False)
    assert shift.current == "big"
    assert shift.end2end(_frame(99))[0][1] == SPECS["big"][0]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("models", "budget", "match"),
    [
        pytest.param(["big", "small"], None, "must name the same pairs", id="name-mismatch"),
        pytest.param(list(SPECS), 149.0, "cannot hold the two largest", id="budget-too-small"),
    ],
)
def test_shift_invalid(models, budget, match) -> None:
    """Mismatched pair names and a budget below the two largest pairs raise ValueError."""
    factories = Factories()
    with pytest.raises(ValueError, match=match):
        Shift({name: factories.make(name) for name in models}, TRAITS, CONFIG, memory_budget=budget)
