# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/haxconn/_scheduler.py -- the Z3 and exhaustive solvers."""

from __future__ import annotations

import threading
import time

import pytest
from z3 import Z3Exception

from tests.research.haxconn.conftest import XAVIER, make_traits
from trtutils.research.haxconn import HaxConnConfig, HaxConnScheduler, _scheduler, simulate

pytestmark = pytest.mark.cpu


def _small() -> dict:
    # a GPU-heavy DNN with a GPU-only head and a short one that is slow on the DLA
    first = make_traits(
        [0.4, 0.3, 0.3, 0.2, 0.2],
        [0.6, 0.5, 0.4, 0.3, None],
        to_dla=[0.0, 0.05, 0.04, 0.03, 0.0],
        to_gpu=[0.0, 0.08, 0.06, 0.05, 0.04],
        gpu_memory=[60, 70, 55, 50, 40],
        dla_memory=[40, 45, 35, 30, 0],
    )
    second = make_traits(
        [0.3, 0.2, 0.5, 0.2],
        [0.9, 0.4, 1.2, 0.3],
        to_dla=[0.0, 0.05, 0.05, 0.05],
        to_gpu=[0.0, 0.05, 0.05, 0.05],
        gpu_memory=[65, 50, 75, 45],
        dla_memory=[45, 35, 50, 30],
    )
    return {"first": first, "second": second}


@pytest.mark.parametrize(
    ("objective", "contention", "max_transitions"),
    [
        pytest.param("latency", "overlap", 1, id="latency-overlap"),
        pytest.param("latency", "static", 2, id="latency-static-two"),
        pytest.param("latency", "none", 1, id="latency-none"),
        pytest.param("throughput", "overlap", 1, id="throughput-overlap"),
    ],
)
def test_z3_matches_exhaustive(objective, contention, max_transitions) -> None:
    """Z3 proves the same optimum the exhaustive search finds, and both beat the baselines."""
    traits = _small()
    results = {}
    for solver in ("exhaustive", "z3"):
        config = HaxConnConfig(
            objective=objective,
            contention=contention,
            max_transitions=max_transitions,
            solver=solver,
            time_budget=None,
        )
        scheduler = HaxConnScheduler(traits, XAVIER, config)
        results[solver] = scheduler.solve()
    assert results["z3"].optimal is True
    assert results["exhaustive"].optimal is True
    assert results["z3"].objective == pytest.approx(results["exhaustive"].objective)
    sign = 1.0 if objective == "latency" else -1.0
    for baseline in scheduler.baselines().values():
        assert sign * results["z3"].objective <= sign * baseline.objective + 1e-9
    for name, accs in results["z3"].assignments.items():
        assert sum(a != b for a, b in zip(accs, accs[1:])) <= max_transitions
        assert all(g.dla for g, acc in zip(traits[name].groups, accs) if acc == "dla")


def test_paper_pair_beats_the_baselines(paper_traits) -> None:
    """On the paper's GoogleNet and AlexNet numbers the schedule splits both DNNs across accelerators."""
    scheduler = HaxConnScheduler(paper_traits, XAVIER, HaxConnConfig(time_budget=None))
    schedules = list(scheduler.solve_iter())
    best = schedules[-1]
    assert best.optimal is True
    assert schedules[0] == min(scheduler.baselines().values(), key=lambda s: s.objective)
    objectives = [s.objective for s in schedules]
    assert objectives == sorted(objectives, reverse=True)
    assert best.objective < 0.9 * scheduler.baselines()["gpu_only"].objective
    assert all({"gpu", "dla"} <= set(accs) for accs in best.assignments.values())
    # evaluate and simulate agree with the solver's own scoring
    assert scheduler.evaluate(best.assignments) == best._replace(optimal=False)
    predicted = simulate(list(paper_traits.values()), list(best.assignments.values()), XAVIER)
    assert predicted == pytest.approx(list(best.latency_ms.values()))


@pytest.mark.parametrize("solver", ["z3", "exhaustive"])
def test_budget_starts_with_the_search(solver) -> None:
    """Time spent using the first schedule (e.g. building its engines) does not eat the budget."""
    config = HaxConnConfig(solver=solver, time_budget=0.5)
    iterator = HaxConnScheduler(_small(), XAVIER, config).solve_iter()
    next(iterator)
    time.sleep(0.6)
    assert list(iterator)[-1].optimal is True


def test_interrupt_stops_a_running_search(paper_traits) -> None:
    """interrupt() ends an unbounded Z3 search from another thread with the best so far."""
    config = HaxConnConfig(max_transitions=None, time_budget=None)
    scheduler = HaxConnScheduler(paper_traits, XAVIER, config)
    found = []
    errors = []

    def search() -> None:
        try:
            found.extend(scheduler.solve_iter())
        except Z3Exception as err:
            errors.append(err)

    thread = threading.Thread(target=search)
    thread.start()
    time.sleep(1.0)
    scheduler.interrupt()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert errors == []
    assert found[-1].optimal is False


@pytest.mark.parametrize("solver", ["z3", "exhaustive"])
def test_interrupt_before_the_search_stops_it(paper_traits, solver) -> None:
    """interrupt() while the caller holds the first schedule stops the search before it runs."""
    config = HaxConnConfig(max_transitions=1, time_budget=None, solver=solver)
    scheduler = HaxConnScheduler(paper_traits, XAVIER, config)
    iterator = scheduler.solve_iter()
    next(iterator)
    scheduler.interrupt()
    assert all(schedule.optimal is False for schedule in iterator)


def test_interrupt_between_z3_calls_ends_cleanly(paper_traits, monkeypatch) -> None:
    """An interrupt that cancels a Z3 call outside check() ends the search, not with an error."""
    config = HaxConnConfig(max_transitions=1, time_budget=None)
    scheduler = HaxConnScheduler(paper_traits, XAVIER, config)

    def cancelled_push(self) -> None:
        # what Z3 does when interrupt() lands between two checks
        scheduler.interrupt()
        err_msg = b"push canceled"
        raise Z3Exception(err_msg)

    monkeypatch.setattr(_scheduler.Solver, "push", cancelled_push)
    schedules = list(scheduler.solve_iter())
    assert len(schedules) == 1
    assert schedules[0].optimal is False


def test_pause_sleeps_after_each_scored_schedule(paper_traits, monkeypatch) -> None:
    """With a pause, scoring sleeps after every schedule; without one it never sleeps."""
    slept = []
    monkeypatch.setattr(_scheduler.time, "sleep", slept.append)
    config = HaxConnConfig(solver="exhaustive")
    HaxConnScheduler(paper_traits, XAVIER, config, pause=1e-4).solve()
    assert len(slept) > 1
    assert set(slept) == {1e-4}
    slept.clear()
    HaxConnScheduler(paper_traits, XAVIER, config).solve()
    assert slept == []
    with pytest.raises(ValueError, match="pause"):
        HaxConnScheduler(paper_traits, XAVIER, config, pause=-1.0)


def test_solve_iter_starts_from_a_better_schedule(paper_traits) -> None:
    """A schedule found before replaces the naive start only when it is better."""
    config = HaxConnConfig(solver="exhaustive")
    scheduler = HaxConnScheduler(paper_traits, XAVIER, config)
    naive = next(scheduler.solve_iter())
    best = scheduler.solve()
    assert next(scheduler.solve_iter(start=best)).assignments == best.assignments
    gpu_only = scheduler.baselines()["gpu_only"]
    assert next(scheduler.solve_iter(start=gpu_only)).assignments == naive.assignments
    assert scheduler.solve().assignments == best.assignments


def test_first_schedule_is_the_best_naive_within_the_limit() -> None:
    """A DNN whose DLA option breaks the transition limit stays on the GPU; the others may not."""
    alternating = make_traits([2.0, 2.0, 2.0, 2.0], [None, 0.1, None, 0.1])
    whole = make_traits([2.0, 2.0], [1.0, 1.0])
    scheduler = HaxConnScheduler(
        {"a": alternating, "b": whole}, config=HaxConnConfig(contention="none")
    )
    first = next(scheduler.solve_iter())
    assert first.assignments == {"a": ("gpu",) * 4, "b": ("dla", "dla")}


def test_time_budget_returns_the_best_so_far(paper_traits) -> None:
    """A tiny budget still returns a valid schedule, not proven optimal."""
    config = HaxConnConfig(max_transitions=None, time_budget=0.01)
    best = HaxConnScheduler(paper_traits, XAVIER, config).solve()
    assert best.optimal is False
    assert set(best.assignments) == {"googlenet", "alexnet"}


@pytest.mark.parametrize(
    ("traits", "contention", "config", "match"),
    [
        pytest.param({}, None, None, "at least one DNN", id="no-dnn"),
        pytest.param("small", None, None, "needs a contention model", id="no-contention"),
        pytest.param("no-memory", XAVIER, None, "characterize it with memory=True", id="no-memory"),
    ],
)
def test_scheduler_rejects_missing_inputs(traits, contention, config, match) -> None:
    """No DNN, no contention model, or no memory throughput raises."""
    if traits == "small":
        traits = _small()
    elif traits == "no-memory":
        traits = {"a": make_traits([1.0], [1.0])}
    with pytest.raises(ValueError, match=match):
        HaxConnScheduler(traits, contention, config)


def test_evaluate_rejects_missing_dnns() -> None:
    """Evaluating a schedule without every DNN raises."""
    scheduler = HaxConnScheduler(_small(), XAVIER)
    with pytest.raises(ValueError, match="missing"):
        scheduler.evaluate({"first": ("gpu",) * 5})
