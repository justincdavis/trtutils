# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/drift/_drift.py -- the per-frame split and concurrent runtime."""

from __future__ import annotations

import dataclasses
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from trtutils.research.drift import Drift, DriftConfig, ModelTraits

pytestmark = pytest.mark.cpu

WIDTH, HEIGHT = 1920, 1080
PERSON = (500, 400, 600, 600)  # class 0, high priority
CAR = (1550, 200, 1700, 260)  # class 1, inside one packing cell
CONFIDENCE = {"base": 0.9, "m640": 0.85, "dla": 0.6, "n320": 0.7}


def _trait(size, latency, f1, *, dla=False, pairs=None):
    # same construction as test_scheduler: constant F1 = tp / 4
    return ModelTraits(
        input_size=(size, size),
        dla=dla,
        latency_ms=[latency],
        confidences=[[0.2] * 4, [0.8] * 4],
        true_positives=[round(f1 * 4)] * 2,
        ground_truths=[4, 4],
        pair_latency_ms={
            name: [value] for name, value in (pairs if pairs is not None else {}).items()
        },
    )


TRAITS = {
    "base": _trait(1280, 30.0, 1.0),
    "m640": _trait(640, 10.0, 1.0, pairs={"dla": 12.5}),
    "dla": _trait(480, 12.0, 0.5, dla=True),
}
GPU_ONLY = {
    "base": TRAITS["base"],
    "m640": TRAITS["m640"],
    "n320": _trait(320, 4.0, 0.75),
}
CONFIG = DriftConfig(lp_accuracy_goal=0.25)


def _frame(*objects: tuple) -> np.ndarray:
    # objects are filled rectangles: red (BGR channel 2) is class 0, green is class 1
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for (x1, y1, x2, y2), cls in objects:
        frame[y1:y2, x1:x2, 2 - cls] = 255
    return frame


class FakeDetector:
    """Finds the coloured rectangles in whatever image it gets, and logs every call."""

    def __init__(self, name, trait, log, hooks) -> None:
        """Stand in for one characterized model."""
        self.name = name
        self.input_shape = trait.input_size
        self.engine = SimpleNamespace(dla_core=0 if trait.dla else None)
        self.log = log
        self.hooks = hooks

    def end2end(self, image):
        self.log.append((self.name, image.shape[:2], threading.get_ident()))
        hook = self.hooks.get(self.name)
        if hook is not None:
            hook()
        detections = []
        for cls in (0, 1):
            mask = (image[..., 2 - cls] == 255).astype(np.uint8)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            for x, y, w, h, _ in stats[1:count]:
                detections.append(
                    ((int(x), int(y), int(x + w), int(y + h)), CONFIDENCE[self.name], cls)
                )
        return detections


class Harness:
    """Builds a Drift over fake detectors and records their calls."""

    def __init__(self, traits=TRAITS, config=CONFIG, **kwargs: object) -> None:
        """Construct the runtime, then clear the warmup calls from the log."""
        self.log = []
        self.hooks = {}
        self.built = {}

        def _factory(name):
            def _build():
                self.built[name] = FakeDetector(name, traits[name], self.log, self.hooks)
                return self.built[name]

            return _build

        self.drift = Drift(
            {name: _factory(name) for name in traits}, traits, {0}, config=config, seed=0, **kwargs
        )
        self.warmup = list(self.log)
        self.log.clear()

    def run(self, frame):
        self.log.clear()
        detections = self.drift.end2end(frame)
        return detections, list(self.log), self.drift.last


def test_warmup_runs_dla_models_on_the_worker() -> None:
    """Construction calls every model three times at its input size, DLA models on the worker."""
    harness = Harness()
    main = threading.get_ident()
    assert sorted(name for name, _, _ in harness.warmup) == ["base"] * 3 + ["dla"] * 3 + ["m640"] * 3
    for name, shape, thread in harness.warmup:
        size = TRAITS[name].input_size[0]
        assert shape == (size, size)
        assert (thread != main) is TRAITS[name].dla
    harness.drift.close()


def test_baseline_then_split() -> None:
    """Frame 0 runs the baseline on the full frame; frame 1 splits, and detections map back exactly."""
    harness = Harness()
    frame = _frame((PERSON, 0), (CAR, 1))
    detections, calls, last = harness.run(frame)
    assert [(name, shape) for name, shape, _ in calls] == [("base", (HEIGHT, WIDTH))]
    assert last.schedule.lp is None
    assert sorted(d[0] for d in detections) == sorted([PERSON, CAR])

    detections, calls, last = harness.run(frame)
    assert (last.schedule.hp, last.schedule.lp, last.schedule.parallel) == ("m640", "dla", True)
    shapes = {name: shape for name, shape, _ in calls}
    # HP crop = person box padded by 64 px
    assert shapes["m640"] == (PERSON[3] - PERSON[1] + 128, PERSON[2] - PERSON[0] + 128)
    # frame 1 still explores every cell: the 52 cells outside the HP cells are single
    # groups (the frame minus a hole is no rectangle), shelf-packed 8 high and 7 wide
    assert shapes["dla"] == (8 * 192, 7 * 192)
    assert [d[0] for d in last.hp_detections] == [PERSON]
    assert [d[0] for d in last.lp_detections] == [CAR]
    assert sorted(d[0] for d in detections) == sorted([PERSON, CAR])
    harness.drift.close()


def test_no_split_lag() -> None:
    """Each frame's calls match that frame's schedule: the baseline always gets the full frame."""
    harness = Harness()
    busy, empty = _frame((PERSON, 0), (CAR, 1)), _frame()
    for frame in (busy, busy, empty, empty, busy, busy):
        _, calls, last = harness.run(frame)
        names = [name for name, _, _ in calls]
        if last.schedule.lp is None:
            assert [(n, s) for n, s, _ in calls] == [(last.schedule.hp, (HEIGHT, WIDTH))]
        else:
            assert names[0] in (last.schedule.hp, last.schedule.lp)
            assert all(shape != (HEIGHT, WIDTH) for _, shape, _ in calls)
    harness.drift.close()


def test_parallel_lp_overlaps_hp() -> None:
    """The DLA model runs on the worker thread while the HP model runs on the caller."""
    harness = Harness()
    frame = _frame((PERSON, 0), (CAR, 1))
    harness.run(frame)
    hp_running, lp_running = threading.Event(), threading.Event()

    def _hp():
        hp_running.set()
        assert lp_running.wait(timeout=5), "LP did not run while HP was running"

    def _lp():
        lp_running.set()
        assert hp_running.wait(timeout=5), "HP did not run while LP was running"

    harness.hooks.update(m640=_hp, dla=_lp)
    _, calls, last = harness.run(frame)
    assert last.schedule.parallel
    threads = {name: thread for name, _, thread in calls}
    assert threads["m640"] == threading.get_ident()
    assert threads["dla"] != threading.get_ident()
    harness.drift.close()


def test_serial_lp_runs_after_hp_on_caller() -> None:
    """A GPU LP model runs after the HP model on the calling thread."""
    harness = Harness(traits=GPU_ONLY)
    frame = _frame((PERSON, 0), (CAR, 1))
    harness.run(frame)
    _, calls, last = harness.run(frame)
    assert (last.schedule.hp, last.schedule.lp, last.schedule.parallel) == ("m640", "n320", False)
    assert [name for name, _, _ in calls] == ["m640", "n320"]
    assert {thread for _, _, thread in calls} == {threading.get_ident()}
    assert sorted(d[0] for d in last.detections) == sorted([PERSON, CAR])
    harness.drift.close()


def test_lp_skipped_when_no_cell_sampled() -> None:
    """When the sampler picks no cell, only the HP model runs, and the frame says so."""
    config = dataclasses.replace(CONFIG, annealing_rate=1000.0, annealing_floor=0.0)
    harness = Harness(config=config)
    frame = _frame((PERSON, 0))
    harness.run(frame)
    _, calls, last = harness.run(frame)
    # the next prediction must come from the model that ran, not from a skipped one
    assert (last.schedule.hp, last.schedule.lp) == ("m640", None)
    assert [name for name, _, _ in calls] == ["m640"]
    assert last.lp_detections == []
    assert [d[0] for d in last.detections] == [PERSON]
    harness.drift.close()


def test_merge_removes_duplicates() -> None:
    """An object in both the squared HP crop and the packed image is reported once."""
    tall_person = (500, 100, 540, 900)
    near_car = (100, 400, 180, 440)  # inside the squared crop, outside the excluded cells
    harness = Harness(config=dataclasses.replace(CONFIG, hp_square=True))
    frame = _frame((tall_person, 0), (near_car, 1))
    harness.run(frame)
    detections, _, last = harness.run(frame)
    assert last.schedule.lp == "dla"
    assert near_car in [d[0] for d in last.hp_detections]
    assert near_car in [d[0] for d in last.lp_detections]
    assert sorted(d[0] for d in detections) == sorted([tall_person, near_car])
    harness.drift.close()


def test_hp_error_drains_lp() -> None:
    """If the HP model fails, the LP call finishes first and the HP error is the one raised."""
    harness = Harness()
    frame = _frame((PERSON, 0), (CAR, 1))
    harness.run(frame)
    lp_done = threading.Event()

    def _lp():
        time.sleep(0.2)
        lp_done.set()
        err_msg = "lp failed"
        raise RuntimeError(err_msg)

    def _hp():
        err_msg = "hp failed"
        raise RuntimeError(err_msg)

    harness.hooks.update(m640=_hp, dla=_lp)
    with pytest.raises(RuntimeError, match="hp failed"):
        harness.drift.end2end(frame)
    assert lp_done.is_set()
    harness.drift.close()


def test_reset_and_frame_size_change() -> None:
    """Reset or a new frame size forgets the previous frame, so the baseline runs next."""
    harness = Harness()
    frame = _frame((PERSON, 0), (CAR, 1))
    harness.run(frame)
    harness.drift.reset()
    assert harness.drift.last is None
    assert harness.run(frame)[2].schedule.lp is None
    assert harness.run(frame)[2].schedule.lp == "dla"
    small = np.ascontiguousarray(frame[:540, :960])
    _, calls, last = harness.run(small)
    assert last.schedule.lp is None
    assert calls[0][1] == (540, 960)


def test_close_releases_models() -> None:
    """Close stops the worker and drops every detector; a closed runtime refuses frames."""
    harness = Harness()
    harness.drift.close()
    assert harness.drift._detectors == {}
    assert harness.drift._executor is None
    with pytest.raises(RuntimeError, match="closed"):
        harness.drift.end2end(_frame())


def test_failed_construction_releases_the_worker() -> None:
    """A model failing its warmup stops the DLA worker before the error propagates."""
    workers = []

    def _dla():
        workers.append(threading.current_thread())
        err_msg = "bad dla"
        raise RuntimeError(err_msg)

    log, hooks = [], {"dla": _dla}
    models = {
        name: (lambda name=name: FakeDetector(name, TRAITS[name], log, hooks)) for name in TRAITS
    }
    with pytest.raises(RuntimeError, match="bad dla"):
        Drift(models, TRAITS, {0}, config=CONFIG)
    assert len(workers) == 1
    assert not workers[0].is_alive()


@pytest.mark.parametrize(
    ("traits", "priority", "match"),
    [
        pytest.param(
            {**TRAITS, "dla": dataclasses.replace(TRAITS["dla"], input_size=(320, 320))},
            {0},
            "has input",
            id="input-size",
        ),
        pytest.param(
            {**TRAITS, "dla": dataclasses.replace(TRAITS["dla"], dla=False)},
            {0},
            "dla=True",
            id="accelerator",
        ),
        pytest.param({"base": TRAITS["base"]}, {0}, "must name the same models", id="names"),
        pytest.param(TRAITS, set(), "priority class", id="no-priority"),
    ],
)
def test_invalid(traits, priority, match) -> None:
    """Detectors that disagree with their traits, mismatched names or no priority class raise."""
    log, hooks = [], {}
    models = {
        name: (lambda name=name: FakeDetector(name, TRAITS[name], log, hooks)) for name in TRAITS
    }
    with pytest.raises(ValueError, match=match):
        Drift(models, traits, priority, config=CONFIG)
