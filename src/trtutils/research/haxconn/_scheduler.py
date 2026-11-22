# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""HaX-CoNN scheduler: find the accelerator mapping of every layer group of every DNN."""

from __future__ import annotations

import itertools
import threading
import time
from typing import TYPE_CHECKING

from z3 import (
    And,
    ArithRef,
    Bool,
    BoolRef,
    Context,
    If,
    Not,
    Or,
    Real,
    RealVal,
    Solver,
    Sum,
    Z3Exception,
    is_true,
)
from z3 import sat as z3_sat
from z3 import unsat as z3_unsat

from ._simulate import encode_assignment, make_timing, option_indices, run_timeline
from ._types import ACCELERATORS, HaxConnConfig, Schedule

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from typing_extensions import Self

    from ._pccs import PCCSModel
    from ._types import ModelTraits

_THROUGHPUT_SEGMENTS = 16


def _real(value: float, ctx: Context) -> ArithRef:
    return RealVal(f"{value:.9f}", ctx)


class HaxConnScheduler:
    """
    Schedule concurrent DNNs across a GPU and a DLA (HaX-CoNN).

    Every schedule is scored by :func:`simulate`, which models transitions, waiting for
    a busy accelerator and shared-memory contention. The "z3" solver keeps the paper's
    formulation in Z3 (a device variable per layer group, start and end times,
    precedence with transition costs, mutual exclusion per accelerator, the transition
    limit) without contention, which bounds every schedule from below. It asks Z3 for an
    assignment whose bound beats the best schedule so far, scores it, excludes it, and
    stops when Z3 proves none is left or the time budget runs out. The "exhaustive"
    solver scores every assignment within the transition limit.

    The scheduler works from traits alone, without engines.
    """

    def __init__(
        self: Self,
        traits: Mapping[str, ModelTraits],
        contention: Mapping[str, PCCSModel] | None = None,
        config: HaxConnConfig | None = None,
        *,
        pause: float = 0.0,
    ) -> None:
        """
        Create the scheduler.

        Parameters
        ----------
        traits : Mapping[str, ModelTraits]
            The traits of each DNN, keyed by name, from :func:`characterize`.
        contention : Mapping[str, PCCSModel], optional
            The contention model of each accelerator, keyed "gpu" and "dla", e.g.
            ``{acc: PCCSModel.from_calibration(c) for acc, c in calibrate().items()}``.
            Default is None, only valid with contention "none".
        config : HaxConnConfig, optional
            The hyperparameters. Default is the paper's values.
        pause : float, optional
            Seconds to sleep after scoring each schedule. Scoring is pure Python, so a
            search in a thread beside running DNNs holds the interpreter lock and stalls
            their worker threads (rounds 20x slower on Orin) unless it pauses; 1e-4 is
            enough. Default is 0.0.

        Raises
        ------
        ValueError
            If no DNN is given, contention is required but the models or the memory
            throughput of a group are missing, or ``pause`` is negative.

        """
        if pause < 0.0:
            err_msg = f"pause must be at least 0, got {pause}"
            raise ValueError(err_msg)
        self._pause = pause
        if len(traits) == 0:
            err_msg = "HaX-CoNN needs at least one DNN"
            raise ValueError(err_msg)
        self._config = config if config is not None else HaxConnConfig()
        if self._config.contention != "none":
            if contention is None or any(acc not in contention for acc in ACCELERATORS):
                err_msg = (
                    f"contention={self._config.contention!r} needs a contention model for "
                    f"{ACCELERATORS}, see calibrate()"
                )
                raise ValueError(err_msg)
            for name, trait in traits.items():
                for group in trait.groups:
                    if any(acc not in group.memory for acc in group.time_ms):
                        err_msg = (
                            f"group {group.first}-{group.last} of {name!r} has no memory "
                            "throughput, characterize it with memory=True"
                        )
                        raise ValueError(err_msg)
        self._names = list(traits)
        self._traits = [traits[name] for name in self._names]
        self._timing = make_timing(self._traits, contention, self._config)
        self._overlap = contention if self._config.contention == "overlap" else None
        self._context: Context | None = None
        self._stop = threading.Event()

    @property
    def config(self: Self) -> HaxConnConfig:
        """The hyperparameters."""
        return self._config

    @property
    def names(self: Self) -> list[str]:
        """The DNNs, in the order the solver indexes them."""
        return list(self._names)

    def evaluate(self: Self, assignments: Mapping[str, Sequence[str]]) -> Schedule:
        """
        Score a schedule.

        Parameters
        ----------
        assignments : Mapping[str, Sequence[str]]
            Per DNN, the accelerator ("gpu" or "dla") of each layer group.

        Returns
        -------
        Schedule
            The schedule with its predicted latencies; ``optimal`` is False.

        Raises
        ------
        ValueError
            If a DNN is missing or an assignment does not fit its groups.

        """
        missing = sorted(set(self._names) - set(assignments))
        if missing:
            err_msg = f"assignments are missing {missing}"
            raise ValueError(err_msg)
        indices = [
            encode_assignment(trait, assignments[name])
            for name, trait in zip(self._names, self._traits)
        ]
        return self._schedule(indices)

    def baselines(self: Self) -> dict[str, Schedule]:
        """
        Score the naive schedules.

        Returns
        -------
        dict[str, Schedule]
            "gpu_only": every DNN on the GPU. "naive": the best schedule that runs each
            DNN either on the GPU or on the DLA wherever its groups can (no layer-level
            collaboration, the paper's GPU & DLA baseline). Neither is held to the
            transition limit.

        """
        gpu_only = self._schedule([[0] * len(t.groups) for t in self._traits])
        return {"gpu_only": gpu_only, "naive": min(self._naive(None), key=self._key)}

    def interrupt(self: Self) -> None:
        """Stop a running search from another thread; it then ends as out of time."""
        self._stop.set()
        if self._context is not None:
            self._context.interrupt()

    def _naive(self: Self, limit: int | None) -> list[Schedule]:
        # each DNN wholly on the GPU, or on the DLA wherever its groups can
        whole: list[list[list[int]]] = []
        for trait in self._traits:
            on_dla = [1 if group.dla else 0 for group in trait.groups]
            options = [[0] * len(on_dla)]
            if any(on_dla) and (limit is None or self._transitions(on_dla) <= limit):
                options.append(on_dla)
            whole.append(options)
        return [self._schedule(list(combo)) for combo in itertools.product(*whole)]

    def solve(self: Self, time_budget: float | None = None) -> Schedule:
        """
        Find the best schedule.

        Parameters
        ----------
        time_budget : float, optional
            Seconds to search, overriding the config. Default is None (the config's).

        Returns
        -------
        Schedule
            The best schedule found; ``optimal`` tells whether it is proven best.

        """
        schedules = list(self.solve_iter(time_budget))
        return schedules[-1]

    def solve_iter(
        self: Self,
        time_budget: float | None = None,
        start: Schedule | None = None,
    ) -> Iterator[Schedule]:
        """
        Search for schedules, yielding each improvement (D-HaX-CoNN).

        The first schedule is the best naive one within the transition limit, so a
        runtime can start right away and switch whenever a better schedule arrives.

        Parameters
        ----------
        time_budget : float, optional
            Seconds to search, overriding the config. Default is None (the config's).
        start : Schedule, optional
            A schedule found before, e.g. by a search that was stopped. The search
            starts from it instead of the best naive schedule when it is better.
            Default is None.

        Yields
        ------
        Schedule
            Ever better schedules. The last one has ``optimal`` True when the search
            finished within the budget.

        """
        budget = time_budget if time_budget is not None else self._config.time_budget
        self._stop.clear()
        best = min(self._naive(self._config.max_transitions), key=self._key)
        if start is not None:
            resumed = self.evaluate(start.assignments)
            if self._key(resumed) < self._key(best):
                best = resumed
        yield best
        # the budget starts when the search does, not while the caller uses the first one
        deadline = None if budget is None else time.monotonic() + budget
        search = self._exhaustive if self._config.solver == "exhaustive" else self._z3
        yield from search(best, deadline)

    def _exhaustive(self: Self, best: Schedule, deadline: float | None) -> Iterator[Schedule]:
        limit = self._config.max_transitions if self._config.max_transitions is not None else 0
        options = [option_indices(trait, limit) for trait in self._traits]
        for combo in itertools.product(*options):
            if self._stop.is_set() or (deadline is not None and time.monotonic() > deadline):
                return
            schedule = self._schedule([list(a) for a in combo])
            if self._key(schedule) < self._key(best):
                best = schedule
                yield schedule
        yield best._replace(optimal=True)

    def _z3(self: Self, best: Schedule, deadline: float | None) -> Iterator[Schedule]:
        try:
            yield from self._z3_search(best, deadline)
        except Z3Exception:
            # interrupt() between two checks cancels the next Z3 call instead
            if not self._stop.is_set():
                raise

    def _z3_search(self: Self, best: Schedule, deadline: float | None) -> Iterator[Schedule]:
        # an own context, so interrupt() stops only this search
        ctx = Context()
        self._context = ctx
        solver = Solver(ctx=ctx)
        on_gpu, ends = self._formulate(solver, ctx)
        rates = (
            self._rate_bounds(solver, ends, ctx) if self._config.objective == "throughput" else []
        )
        self._exclude(solver, on_gpu, best)
        while not self._stop.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0.0:
                return
            if remaining is not None:
                solver.set("timeout", max(int(remaining * 1000), 1))
            solver.push()
            if self._config.objective == "latency":
                solver.add(And([end < _real(best.objective, ctx) for end in ends]))
            else:
                solver.add(Sum(rates) > _real(best.objective, ctx))
            result = solver.check()
            if result != z3_sat:
                solver.pop()
                # unsat: nothing can beat the best; unknown: out of time
                if result == z3_unsat:
                    yield best._replace(optimal=True)
                return
            model = solver.model()
            indices = [
                [0 if is_true(model.eval(g, model_completion=True)) else 1 for g in groups]
                for groups in on_gpu
            ]
            solver.pop()
            schedule = self._schedule(indices)
            self._exclude(solver, on_gpu, schedule)
            if self._key(schedule) < self._key(best):
                best = schedule
                yield schedule

    def _formulate(
        self: Self, solver: Solver, ctx: Context
    ) -> tuple[list[list[BoolRef]], list[ArithRef]]:
        # the paper's constraints without contention: a lower bound on every schedule
        timing = self._timing
        on_gpu: list[list[BoolRef]] = []
        starts: list[list[ArithRef]] = []
        group_ends: list[list[ArithRef]] = []
        ends: list[ArithRef] = []
        for n, trait in enumerate(self._traits):
            count = len(trait.groups)
            gpu = [Bool(f"gpu_{n}_{i}", ctx) for i in range(count)]
            start = [Real(f"start_{n}_{i}", ctx) for i in range(count)]
            end = [Real(f"end_{n}_{i}", ctx) for i in range(count)]
            t_in, t_out = timing.transition_in[n], timing.transition_out[n]
            for i in range(count):
                dur_g, dur_d = timing.duration[n][i]
                if dur_d is None:
                    solver.add(gpu[i])
                    solver.add(end[i] == start[i] + _real(dur_g or 0.0, ctx))
                else:
                    solver.add(
                        end[i] == start[i] + If(gpu[i], _real(dur_g or 0.0, ctx), _real(dur_d, ctx))
                    )
            solver.add(start[0] >= If(gpu[0], _real(t_in[0][0], ctx), _real(t_in[0][1], ctx)))
            for i in range(count - 1):
                hop = If(
                    gpu[i] == gpu[i + 1],
                    _real(0.0, ctx),
                    If(
                        gpu[i],
                        _real(t_out[i][0] + t_in[i + 1][1], ctx),
                        _real(t_out[i][1] + t_in[i + 1][0], ctx),
                    ),
                )
                solver.add(start[i + 1] >= end[i] + hop)
            limit = self._config.max_transitions
            if limit is not None and count > 1:
                switches = [If(gpu[i] != gpu[i + 1], 1, 0) for i in range(count - 1)]
                solver.add(Sum(switches) <= limit)
            # mutual exclusion per accelerator (eq. 9, without the epsilon)
            for m in range(n):
                for i in range(count):
                    for j in range(len(on_gpu[m])):
                        solver.add(
                            Or(
                                gpu[i] != on_gpu[m][j],
                                end[i] <= starts[m][j],
                                group_ends[m][j] <= start[i],
                            )
                        )
            last = count - 1
            ends.append(
                end[last] + If(gpu[last], _real(t_out[last][0], ctx), _real(t_out[last][1], ctx))
            )
            on_gpu.append(gpu)
            starts.append(start)
            group_ends.append(end)
        return on_gpu, ends

    def _rate_bounds(
        self: Self, solver: Solver, ends: list[ArithRef], ctx: Context
    ) -> list[ArithRef]:
        # 1000 / T is convex, so the max of its chords bounds it from above on [lo, hi];
        # hi is every group of every DNN in series with every transition
        timing = self._timing
        hi = sum(
            max(t for t in times if t is not None) + max(t_in) + max(t_out)
            for n in range(len(ends))
            for times, t_in, t_out in zip(
                timing.duration[n], timing.transition_in[n], timing.transition_out[n]
            )
        )
        rates: list[ArithRef] = []
        for n, end in enumerate(ends):
            lo = max(sum(min(t for t in g if t is not None) for g in timing.duration[n]), 1e-6)
            top = max(hi, lo * 2.0)
            points = [
                lo * (top / lo) ** (k / _THROUGHPUT_SEGMENTS)
                for k in range(_THROUGHPUT_SEGMENTS + 1)
            ]
            rate = Real(f"rate_{n}", ctx)
            chords = []
            for a, b in zip(points, points[1:]):
                slope = (1000.0 / b - 1000.0 / a) / (b - a)
                chords.append(
                    rate <= _real(1000.0 / a, ctx) + _real(slope, ctx) * (end - _real(a, ctx))
                )
            solver.add(Or(chords))
            solver.add(rate <= _real(1000.0 / lo, ctx))
            rates.append(rate)
        return rates

    @staticmethod
    def _exclude(solver: Solver, on_gpu: list[list[BoolRef]], schedule: Schedule) -> None:
        literals = []
        for gpu, accs in zip(on_gpu, schedule.assignments.values()):
            literals.extend(g if acc == "dla" else Not(g) for g, acc in zip(gpu, accs))
        solver.add(Or(literals))

    @staticmethod
    def _transitions(accelerators: Sequence[int]) -> int:
        return sum(a != b for a, b in zip(accelerators, accelerators[1:]))

    def _key(self: Self, schedule: Schedule) -> float:
        # smaller is better for both objectives
        return schedule.objective if self._config.objective == "latency" else -schedule.objective

    def _schedule(self: Self, indices: Sequence[Sequence[int]]) -> Schedule:
        latency = run_timeline(self._timing, indices, self._overlap)
        if self._pause > 0.0:
            # a real sleep hands the interpreter lock over; sleep(0) lets this thread take
            # it straight back and starves the others
            time.sleep(self._pause)
        if self._config.objective == "latency":
            objective = max(latency)
        else:
            objective = sum(1000.0 / t for t in latency if t > 0.0)
        return Schedule(
            assignments={
                name: tuple(ACCELERATORS[a] for a in accs)
                for name, accs in zip(self._names, indices)
            },
            latency_ms=dict(zip(self._names, latency)),
            objective=objective,
            optimal=False,
        )
