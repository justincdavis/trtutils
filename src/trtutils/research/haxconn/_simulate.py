# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Event-driven timeline of concurrent DNNs mapped onto a GPU and a DLA."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, NamedTuple

from ._types import ACCELERATORS, HaxConnConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ._pccs import PCCSModel
    from ._types import ModelTraits

_EPS = 1e-12


class Timing(NamedTuple):
    """
    The reduced numbers :func:`simulate` and the solvers work from.

    Every list is indexed by DNN, then layer group, then accelerator (0 = GPU,
    1 = DLA). Group durations are None on accelerators the group cannot run on.

    Attributes
    ----------
    duration : list[list[list[float | None]]]
        Standalone compute time, in ms; with "static" contention already slowed down.
    transition_in : list[list[list[float]]]
        Cost of starting the group on an accelerator after a transition, in ms.
    transition_out : list[list[list[float]]]
        Cost of handing the group's output to the other accelerator, in ms.
    memory : list[list[list[float]]]
        Standalone memory throughput, EMC percent (0.0 when not measured).

    """

    duration: list[list[list[float | None]]]
    transition_in: list[list[list[float]]]
    transition_out: list[list[list[float]]]
    memory: list[list[list[float]]]


def make_timing(
    traits: Sequence[ModelTraits],
    contention: Mapping[str, PCCSModel] | None,
    config: HaxConnConfig,
) -> Timing:
    """
    Reduce traits to the numbers of :class:`Timing`.

    Parameters
    ----------
    traits : Sequence[ModelTraits]
        The traits of each DNN.
    contention : Mapping[str, PCCSModel] | None
        The contention model of each accelerator, needed unless contention is "none".
    config : HaxConnConfig
        The hyperparameters (contention mode and sample reductions).

    Returns
    -------
    Timing
        The reduced numbers.

    """
    duration: list[list[list[float | None]]] = []
    t_in: list[list[list[float]]] = []
    t_out: list[list[list[float]]] = []
    memory: list[list[list[float]]] = []
    for trait in traits:
        duration.append([])
        t_in.append([])
        t_out.append([])
        memory.append([])
        for group in trait.groups:
            mems = [group.throughput(a, config.memory_metric) for a in ACCELERATORS]
            times: list[float | None] = []
            for a, acc in enumerate(ACCELERATORS):
                if acc not in group.time_ms:
                    times.append(None)
                    continue
                t = group.time(acc, config.latency_metric)
                if config.contention == "static" and contention is not None:
                    # reference implementation: saturated external demand, always
                    t *= contention[acc].slowdown(mems[a], contention[acc].peak)
                times.append(t)
            duration[-1].append(times)
            t_in[-1].append([group.transition_in_ms.get(a, 0.0) for a in ACCELERATORS])
            t_out[-1].append([group.transition_out_ms.get(a, 0.0) for a in ACCELERATORS])
            memory[-1].append(mems)
    return Timing(duration, t_in, t_out, memory)


def run_timeline(
    timing: Timing,
    assignments: Sequence[Sequence[int]],
    contention: Mapping[str, PCCSModel] | None = None,
) -> list[float]:
    """
    Simulate one round of every DNN, all released at time zero.

    Parameters
    ----------
    timing : Timing
        The reduced numbers, from :func:`make_timing`.
    assignments : Sequence[Sequence[int]]
        Per DNN and group, the accelerator index (0 = GPU, 1 = DLA).
    contention : Mapping[str, PCCSModel], optional
        When given, groups running at the same time on the two accelerators slow each
        other down by the PCCS prediction (overlap contention). Default is None.

    Returns
    -------
    list[float]
        The latency of each DNN in milliseconds.

    """
    count = len(assignments)
    group = [0] * count
    remaining = [0.0] * count
    delay = [timing.transition_in[n][0][assignments[n][0]] for n in range(count)]
    # "delay" counts down a transition, "wait" queues for an accelerator, "run" computes
    phase = ["delay"] * count
    finishing = [False] * count
    owner: list[int | None] = [None, None]
    queue: list[deque[int]] = [deque(), deque()]
    done = [0.0] * count
    now = 0.0
    cache: dict[tuple[int, float, float], float] = {}

    def _rate(n: int) -> float:
        acc = assignments[n][group[n]]
        other = owner[1 - acc]
        if contention is None or other is None:
            return 1.0
        own = timing.memory[n][group[n]][acc]
        ext = timing.memory[other][group[other]][1 - acc]
        key = (acc, own, ext)
        if key not in cache:
            cache[key] = 1.0 / contention[ACCELERATORS[acc]].slowdown(own, ext)
        return cache[key]

    def _start(n: int, acc: int) -> None:
        owner[acc] = n
        phase[n] = "run"
        remaining[n] = timing.duration[n][group[n]][acc] or 0.0

    while any(p != "done" for p in phase):
        for n in range(count):
            if phase[n] == "delay" and delay[n] <= _EPS:
                if finishing[n]:
                    phase[n] = "done"
                    done[n] = now
                    continue
                acc = assignments[n][group[n]]
                if owner[acc] is None and not queue[acc]:
                    _start(n, acc)
                else:
                    phase[n] = "wait"
                    queue[acc].append(n)
        for acc in (0, 1):
            if owner[acc] is None and queue[acc]:
                _start(queue[acc].popleft(), acc)
        rates = {n: _rate(n) for n in range(count) if phase[n] == "run"}
        steps = [remaining[n] / rate for n, rate in rates.items()]
        steps += [delay[n] for n in range(count) if phase[n] == "delay"]
        if not steps:
            break
        step = max(min(steps), 0.0)
        now += step
        for n in range(count):
            if phase[n] == "delay":
                delay[n] -= step
        for n, rate in rates.items():
            remaining[n] -= step * rate
            if remaining[n] > _EPS:
                continue
            acc = assignments[n][group[n]]
            last = group[n] + 1 == len(assignments[n])
            if not last and assignments[n][group[n] + 1] == acc:
                # same accelerator: the segment keeps it
                group[n] += 1
                remaining[n] = timing.duration[n][group[n]][acc] or 0.0
                continue
            owner[acc] = None
            phase[n] = "delay"
            delay[n] = timing.transition_out[n][group[n]][acc]
            if last:
                finishing[n] = True
            else:
                group[n] += 1
                delay[n] += timing.transition_in[n][group[n]][1 - acc]
    return done


def simulate(
    traits: Sequence[ModelTraits],
    assignments: Sequence[Sequence[str]],
    contention: Mapping[str, PCCSModel] | None = None,
    config: HaxConnConfig | None = None,
) -> list[float]:
    """
    Predict the latency of every DNN for a schedule.

    All DNNs are released together. Each layer group runs on its accelerator; a run of
    consecutive groups of one DNN on the same accelerator holds it until the run ends,
    and DNNs waiting for a busy accelerator queue in arrival order. Moving between
    accelerators costs the transition-out of the group before and the transition-in of
    the group after, and every DNN pays the transition-in of its first group and the
    transition-out of its last. With "overlap" contention, while both accelerators are
    busy each group progresses at ``1 / slowdown(own, other)`` of its standalone speed.

    Parameters
    ----------
    traits : Sequence[ModelTraits]
        The traits of each DNN.
    assignments : Sequence[Sequence[str]]
        Per DNN, the accelerator ("gpu" or "dla") of each layer group.
    contention : Mapping[str, PCCSModel], optional
        The contention model of each accelerator, keyed "gpu" and "dla". Needed unless
        the config's contention is "none". Default is None.
    config : HaxConnConfig, optional
        The hyperparameters. Default is the paper's values.

    Returns
    -------
    list[float]
        The latency of each DNN in milliseconds.

    Raises
    ------
    ValueError
        If an assignment has the wrong length or names an accelerator a group cannot
        run on, or contention is required but missing.

    """
    config = config if config is not None else HaxConnConfig()
    if config.contention != "none" and contention is None:
        err_msg = f"contention={config.contention!r} needs a contention model per accelerator"
        raise ValueError(err_msg)
    indices = [encode_assignment(trait, accs) for trait, accs in zip(traits, assignments)]
    timing = make_timing(traits, contention, config)
    return run_timeline(timing, indices, contention if config.contention == "overlap" else None)


def encode_assignment(traits: ModelTraits, accelerators: Sequence[str]) -> list[int]:
    """
    Turn accelerator names into indices, checking them against the groups.

    Parameters
    ----------
    traits : ModelTraits
        The traits of the DNN.
    accelerators : Sequence[str]
        The accelerator ("gpu" or "dla") of each group.

    Returns
    -------
    list[int]
        The accelerator index (0 = GPU, 1 = DLA) of each group.

    Raises
    ------
    ValueError
        If the length differs from the number of groups, or an accelerator is unknown
        or not available to its group.

    """
    if len(accelerators) != len(traits.groups):
        err_msg = f"expected {len(traits.groups)} accelerators, got {len(accelerators)}"
        raise ValueError(err_msg)
    indices: list[int] = []
    for group, acc in zip(traits.groups, accelerators):
        if acc not in group.time_ms:
            err_msg = f"group {group.first}-{group.last} cannot run on {acc!r}"
            raise ValueError(err_msg)
        indices.append(ACCELERATORS.index(acc))
    return indices
