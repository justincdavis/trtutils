# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_types.py -- config, traits and JSON round trips."""

from __future__ import annotations

import pytest

from trtutils.research.haxconn import (
    Calibration,
    HaxConnConfig,
    Schedule,
    load_calibration,
    load_schedule,
    load_traits,
    save_calibration,
    save_schedule,
    save_traits,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"objective": "energy"}, "Unknown objective", id="objective"),
        pytest.param({"contention": "pccs"}, "Unknown contention", id="contention"),
        pytest.param({"solver": "greedy"}, "Unknown solver", id="solver"),
        pytest.param({"latency_metric": "p99"}, "Unknown metric", id="latency-metric"),
        pytest.param({"memory_metric": "p99"}, "Unknown metric", id="memory-metric"),
        pytest.param({"max_transitions": -1}, "max_transitions is out of range", id="transitions"),
        pytest.param({"time_budget": 0.0}, "time_budget is out of range", id="budget"),
        pytest.param({"update_ms": (100.0, 50.0)}, "update_ms is out of range", id="updates"),
        pytest.param({"update_ms": (-1.0,)}, "update_ms is out of range", id="negative-update"),
        pytest.param(
            {"solver": "exhaustive", "max_transitions": None},
            "needs max_transitions",
            id="unbounded",
        ),
    ],
)
def test_config_rejects_invalid_values(kwargs, match) -> None:
    """Every invalid hyperparameter raises a ValueError naming it."""
    with pytest.raises(ValueError, match=match):
        HaxConnConfig(**kwargs)


def test_config_presets() -> None:
    """The paper preset is overlap contention; the upstream preset is a static slowdown."""
    paper, upstream = HaxConnConfig(), HaxConnConfig.upstream()
    assert (paper.contention, paper.memory_metric, paper.max_transitions) == ("overlap", "median", 1)
    assert (upstream.contention, upstream.memory_metric) == ("static", "max")
    assert HaxConnConfig(max_transitions=None, time_budget=None).max_transitions is None
    assert paper.update_ms == (25.0, 100.0, 250.0, 500.0, 1500.0)
    assert HaxConnConfig(update_ms=[0, 10]).update_ms == (0.0, 10.0)


def test_traits_round_trip(tmp_path, paper_traits) -> None:
    """Traits survive a JSON round trip and reduce their raw samples."""
    traits = paper_traits["googlenet"]
    traits.dla_launch_ms = 0.1
    traits.groups[0].time_ms["gpu"] = [0.4, 0.5, 0.9]
    save_traits(traits, tmp_path / "traits.json")
    loaded = load_traits(tmp_path / "traits.json")
    assert loaded == traits
    assert loaded.num_layers == 10
    group = loaded.groups[0]
    assert group.dla is True
    assert group.time("gpu", "median") == pytest.approx(0.5)
    assert group.throughput("gpu") == pytest.approx(41.97)


def test_calibration_and_schedule_round_trip(tmp_path) -> None:
    """Calibrations and schedules survive a JSON round trip."""
    calibration = {
        "gpu": Calibration("gpu", [10.0, 20.0], [5.0, 15.0], [[100.0, 98.0], [97.0, 90.0]]),
        "dla": Calibration("dla", [5.0, 15.0], [10.0, 20.0], [[99.0, 95.0], [96.0, 88.0]]),
    }
    save_calibration(calibration, tmp_path / "calibration.json")
    assert load_calibration(tmp_path / "calibration.json") == calibration
    schedule = Schedule(
        {"a": ("gpu", "dla"), "b": ("dla",)}, {"a": 1.5, "b": 2.0}, 2.0, optimal=True
    )
    save_schedule(schedule, tmp_path / "schedule.json")
    assert load_schedule(tmp_path / "schedule.json") == schedule
