# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_haxconn.py -- the runtime, on fake engines."""

from __future__ import annotations

import time

import pytest

from tests.research.haxconn.conftest import XAVIER, make_traits
from trtutils.research.haxconn import HaxConn, HaxConnConfig, HaxConnScheduler, prebuild_engines

pytestmark = pytest.mark.cpu

_FAST = HaxConnConfig(solver="exhaustive", update_ms=None)


@pytest.fixture
def pool(tmp_path, paper_traits):
    """Three DNNs: the paper pair and a small third one, with empty ONNX files."""
    traits = dict(paper_traits)
    traits["small"] = make_traits(
        [0.3, 0.2], [0.4, 0.3], gpu_memory=[40.0, 50.0], dla_memory=[30.0, 35.0]
    )
    models = {}
    for name in traits:
        models[name] = tmp_path / f"{name}.onnx"
        models[name].write_bytes(b"")
    return models, traits


def _runtime(pool, tmp_path, **kwargs: object) -> HaxConn:
    models, traits = pool
    kwargs.setdefault("config", _FAST)
    return HaxConn(models, traits, XAVIER, engine_dir=tmp_path / "engines", **kwargs)


def _wait(condition, runtime: HaxConn, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        runtime.mock_execute()
        time.sleep(0.005)


def test_set_active_runs_only_the_chosen_dnns(fakes, pool, tmp_path) -> None:
    """Each call runs the active DNNs; set_active changes them between calls."""
    runtime = _runtime(pool, tmp_path, active=["googlenet", "alexnet"])
    assert set(runtime.mock_execute()) == {"googlenet", "alexnet"}
    schedule = runtime.set_active(["small"])
    assert runtime.active == ["small"]
    assert set(schedule.assignments) == {"small"}
    assert set(runtime.mock_execute()) == {"small"}
    assert runtime.last is not None
    assert set(runtime.last) == {"small"}
    with pytest.raises(ValueError, match="non-empty subset"):
        runtime.set_active(["missing"])
    with pytest.raises(ValueError, match="non-empty subset"):
        runtime.set_active([])
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.set_active(["small"])


def test_a_set_that_comes_back_reuses_its_schedule_and_engines(fakes, pool, tmp_path) -> None:
    """A set seen before starts on its schedule, with its engines still loaded."""
    loads, builds = fakes
    runtime = _runtime(pool, tmp_path, active=["googlenet", "alexnet"])
    first = runtime.schedule
    runtime.set_active(["small"])
    loaded, built = len(loads), len(builds)
    assert runtime.set_active(["alexnet", "googlenet"]) == first
    assert (len(loads), len(builds)) == (loaded, built)
    assert set(runtime.schedules) == {frozenset({"googlenet", "alexnet"}), frozenset({"small"})}
    runtime.close()


def test_dynamic_starts_naive_and_reaches_the_optimum(fakes, pool, tmp_path) -> None:
    """D-HaX-CoNN starts on the best naive schedule and switches to the proven optimum."""
    _, traits = pool
    pair = {name: traits[name] for name in ("googlenet", "alexnet")}
    scheduler = HaxConnScheduler(pair, XAVIER, _FAST)
    naive = next(scheduler.solve_iter())
    best = scheduler.solve()
    assert best.assignments != naive.assignments
    runtime = _runtime(pool, tmp_path, active=list(pair), dynamic=True)
    assert runtime.schedule.assignments == naive.assignments
    _wait(lambda: runtime.schedule.optimal, runtime)
    assert runtime.schedule == best
    assert runtime.searching is False
    assert runtime.schedules[frozenset(pair)] == best
    runtime.close()


def test_update_times_hold_back_switches(fakes, pool, tmp_path) -> None:
    """With update_ms, a better schedule found at once is only switched in at the first time."""
    config = HaxConnConfig(solver="exhaustive", update_ms=(400.0,))
    runtime = _runtime(pool, tmp_path, config=config, active=["googlenet", "alexnet"], dynamic=True)
    first = runtime.schedule
    start = time.monotonic()
    while time.monotonic() - start < 0.2:
        runtime.mock_execute()
    assert runtime.schedule == first
    _wait(lambda: runtime.schedule.optimal, runtime)
    assert runtime.schedule.assignments != first.assignments
    runtime.close()


def test_prebuilt_mode_switches_only_to_built_engines(fakes, pool, tmp_path) -> None:
    """build="prebuilt" never builds in the background; with every engine prebuilt it switches."""
    _, builds = fakes
    models, traits = pool
    active = ["googlenet", "alexnet"]
    runtime = _runtime(pool, tmp_path, active=active, dynamic=True, build="prebuilt")
    first, built = runtime.schedule, len(builds)
    _wait(lambda: not runtime.searching, runtime)
    time.sleep(0.1)
    runtime.mock_execute()
    assert runtime.schedule.assignments == first.assignments
    assert len(builds) == built
    runtime.close()

    pair = {name: models[name] for name in active}
    prebuild_engines(pair, {name: traits[name] for name in active}, tmp_path / "engines")
    built = len(builds)
    runtime = _runtime(pool, tmp_path, active=active, dynamic=True, build="prebuilt")
    _wait(lambda: runtime.schedule.optimal, runtime)
    assert runtime.schedule.assignments != first.assignments
    assert len(builds) == built
    runtime.close()


def test_switching_sets_stops_the_running_search(fakes, pool, tmp_path) -> None:
    """set_active ends an unbounded search for the old set; close() returns promptly."""
    config = HaxConnConfig(max_transitions=None, time_budget=None, update_ms=None)
    runtime = _runtime(pool, tmp_path, config=config, active=["googlenet", "alexnet"], dynamic=True)
    time.sleep(0.3)
    assert runtime.searching is True
    runtime.set_active(["small"])
    _wait(lambda: not runtime.searching, runtime)
    assert runtime.searching is False
    start = time.monotonic()
    runtime.close()
    assert time.monotonic() - start < 5.0


def test_a_proven_schedule_skips_the_search(fakes, pool, tmp_path) -> None:
    """A proven-optimal schedule passed in starts its set without a background search."""
    _, traits = pool
    pair = {name: traits[name] for name in ("googlenet", "alexnet")}
    best = HaxConnScheduler(pair, XAVIER, _FAST).solve()
    runtime = _runtime(pool, tmp_path, active=list(pair), schedules=[best], dynamic=True)
    assert runtime.schedule == best
    assert runtime.searching is False
    runtime.close()


@pytest.mark.parametrize(("max_engines", "reloads"), [(1, 1), (2, 0)])
def test_engine_cache_keeps_the_most_recent(fakes, pool, tmp_path, max_engines, reloads) -> None:
    """Engines beyond max_engines are dropped, least recently used first, and reloaded."""
    loads, _ = fakes
    runtime = _runtime(pool, tmp_path, active=["small"], max_engines=max_engines)
    runtime.set_active(["alexnet"])
    loaded = len(loads)
    runtime.set_active(["small"])
    assert len(loads) - loaded == reloads
    runtime.close()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"build": "lazy"}, "Unknown build", id="build"),
        pytest.param({"max_engines": 0}, "max_engines", id="max-engines"),
        pytest.param({"active": ["missing"]}, "non-empty subset", id="active"),
    ],
)
def test_runtime_rejects_bad_arguments(fakes, pool, tmp_path, kwargs, match) -> None:
    """Unknown build modes, an empty engine cache and unknown DNNs raise ValueError."""
    with pytest.raises(ValueError, match=match):
        _runtime(pool, tmp_path, **kwargs)


def test_schedule_must_cover_the_active_dnns(fakes, pool, tmp_path) -> None:
    """A schedule for other DNNs than the active ones raises ValueError."""
    _, traits = pool
    best = HaxConnScheduler({"small": traits["small"]}, XAVIER, _FAST).solve()
    with pytest.raises(ValueError, match="active DNNs"):
        _runtime(pool, tmp_path, active=["alexnet"], schedule=best)
