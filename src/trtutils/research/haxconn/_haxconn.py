# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""HaX-CoNN runtime: run DNNs concurrently on the schedule's mixed GPU/DLA engines."""

from __future__ import annotations

import gc
import os
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING

from trtutils._engine import TRTEngine
from trtutils._log import LOG

from ._build import build_engines, engine_paths
from ._scheduler import HaxConnScheduler
from ._types import HaxConnConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from concurrent.futures import Future

    import numpy as np
    from typing_extensions import Self

    from trtutils.builder._batcher import AbstractBatcher
    from trtutils.core._interface import EngineInputs

    from ._pccs import PCCSModel
    from ._types import ModelTraits, Schedule

_BARRIER_TIMEOUT_S = 60.0
_WARMUP_ITERATIONS = 5
_BUILDS = ("background", "prebuilt")
# the background search's pause per scored schedule, so the DNN threads keep the interpreter
_SOLVER_PAUSE_S = 1e-4


def _run(
    engine: TRTEngine,
    data: EngineInputs,
    barrier: threading.Barrier,
) -> tuple[list[np.ndarray], float]:
    # every worker starts its DNN at the same moment, as the schedule assumes
    barrier.wait(_BARRIER_TIMEOUT_S)
    t0 = time.perf_counter()
    outputs = engine.execute(data)
    return outputs, (time.perf_counter() - t0) * 1000.0


class _Search:
    """The background search of one set of DNNs: a solver thread and a switching thread."""

    def __init__(
        self: Self,
        names: frozenset[str],
        scheduler: HaxConnScheduler,
        first: Schedule,
    ) -> None:
        self.names = names
        self.scheduler = scheduler
        self.history = [first]
        self.finished = False
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.improved = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures: list[Future[None]] = []


class HaxConn:
    """
    Concurrent DNN execution across a GPU and a DLA on HaX-CoNN schedules.

    Each DNN runs as one engine whose layer groups sit on the accelerators the schedule
    picked. :meth:`execute` starts every active DNN at the same moment, each on its own
    worker thread, and returns when all are done. Engines are built with
    :func:`build_engines` and kept in ``engine_dir``, so schedules seen before load
    without a build; loaded engines stay in memory up to ``max_engines``.

    ``models`` is the pool of DNNs that may run; :meth:`set_active` picks the ones that
    run together, and may be called between two calls to change them. The best schedule
    found for each set is kept, so a set that comes back starts on it.

    With ``dynamic`` (D-HaX-CoNN), a new set starts right away on the best naive
    schedule (or the best schedule found for it before) while a background thread keeps
    solving; better schedules are switched in between two calls, at the config's
    ``update_ms`` times. With ``build="background"`` a better schedule's engines are
    built while the DNNs run, which competes with them for the GPU; with
    ``build="prebuilt"`` only schedules whose engines are already in ``engine_dir`` are
    switched to (see :func:`prebuild_engines`), so switching costs a load.

    No engine captures CUDA graphs. Call :meth:`close` when done.
    """

    def __init__(
        self: Self,
        models: Mapping[str, Path | str],
        traits: Mapping[str, ModelTraits],
        contention: Mapping[str, PCCSModel] | None = None,
        config: HaxConnConfig | None = None,
        schedule: Schedule | None = None,
        engine_dir: Path | str | None = None,
        dla_core: int = 0,
        workspace: float = 4.0,
        timing_cache: Path | str | None = None,
        data_batchers: Mapping[str, AbstractBatcher] | None = None,
        *,
        active: Sequence[str] | None = None,
        schedules: Iterable[Schedule] | None = None,
        dynamic: bool = False,
        build: str = "background",
        solver_core: int | None = None,
        max_engines: int = 16,
        verbose: bool | None = None,
    ) -> None:
        """
        Create the HaX-CoNN runtime.

        Parameters
        ----------
        models : Mapping[str, Path | str]
            The ONNX file of each DNN that may run, keyed by name.
        traits : Mapping[str, ModelTraits]
            The traits of the same DNNs, from :func:`characterize`.
        contention : Mapping[str, PCCSModel], optional
            The contention model of each accelerator. Default is None, only valid with
            contention "none".
        config : HaxConnConfig, optional
            The hyperparameters. Default is the paper's values.
        schedule : Schedule, optional
            The schedule of the first active set. Default is None: solve for it (or,
            with ``dynamic``, start on the best naive schedule). With ``dynamic`` the
            search starts from it unless it is proven optimal.
        engine_dir : Path | str, optional
            The directory for the engines. Default is None: a temporary directory
            removed by :meth:`close`.
        dla_core : int, optional
            The DLA core the DLA layers run on. Default is 0.
        workspace : float, optional
            The TensorRT workspace in GiB. Default is 4.0.
        timing_cache : Path | str, optional
            A timing cache file shared by the builds. Default is None.
        data_batchers : Mapping[str, AbstractBatcher], optional
            The INT8 calibration data of each DNN, for traits measured in INT8. A DNN
            without one gets engines that time correctly but compute nothing useful.
            Default is None.
        active : Sequence[str], optional
            The DNNs that run first. Default is None: all of them.
        schedules : Iterable[Schedule], optional
            Schedules found before (e.g. ``runtime.schedules.values()`` of an earlier
            run), each for the set of DNNs it names. A set with a schedule here starts on
            it, and with ``dynamic`` skips the search if it is proven optimal.
            Default is None.
        dynamic : bool, optional
            Whether to keep solving in the background and switch to better schedules.
            Default is False.
        build : str, optional
            Dynamic mode only: "background" builds a better schedule's engines while the
            DNNs run; "prebuilt" switches only to schedules whose engines exist. The
            first schedule of a set is always built if needed. Default is "background".
        solver_core : int, optional
            Dynamic mode only: the CPU core the solver thread is pinned to, as in the
            paper (Linux only). Default is None (not pinned).
        max_engines : int, optional
            The most engines kept loaded, the active ones included. Default is 16.
        verbose : bool, optional
            Whether to log builds and switches. Default is None (False).

        Raises
        ------
        ValueError
            If ``models`` and ``traits`` name different DNNs, the traits differ in
            precision, ``active`` names an unknown DNN, ``schedule`` does not cover the
            active DNNs, a schedule does not fit the traits' layer groups, ``build`` is
            unknown, or ``max_engines`` is below 1.

        """
        if set(models) != set(traits):
            err_msg = f"models {sorted(models)} and traits {sorted(traits)} must name the same DNNs"
            raise ValueError(err_msg)
        if len({t.precision for t in traits.values()}) > 1:
            err_msg = "every DNN's traits must be measured in the same precision"
            raise ValueError(err_msg)
        if build not in _BUILDS:
            err_msg = f"Unknown build {build!r}, expected one of {_BUILDS}"
            raise ValueError(err_msg)
        if max_engines < 1:
            err_msg = f"max_engines must be at least 1, got {max_engines}"
            raise ValueError(err_msg)
        first = list(active) if active is not None else list(models)
        if schedule is not None and set(schedule.assignments) != set(first):
            err_msg = (
                f"schedule covers {sorted(schedule.assignments)}, the active DNNs are "
                f"{sorted(first)}"
            )
            raise ValueError(err_msg)
        self._models = dict(models)
        self._traits = dict(traits)
        self._contention = contention
        self._config = config if config is not None else HaxConnConfig()
        self._dynamic = dynamic
        self._build = build
        self._solver_core = solver_core
        self._max_engines = max_engines
        self._tmp = tempfile.TemporaryDirectory() if engine_dir is None else None
        self._engine_dir = Path(self._tmp.name if self._tmp is not None else str(engine_dir))
        self._dla_core = dla_core
        self._workspace = workspace
        self._timing_cache = timing_cache
        self._data_batchers = dict(data_batchers) if data_batchers is not None else {}
        self._verbose = verbose

        self._schedules: dict[frozenset[str], Schedule] = {}
        seeds = [*(schedules if schedules is not None else []), schedule]
        for seed in seeds:
            if seed is None:
                continue
            names = frozenset(seed.assignments)
            # a schedule from another characterization must not build silently
            self._new_scheduler(names).evaluate(seed.assignments)
            self._schedules[names] = seed

        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict[Path, TRTEngine] = OrderedDict()
        self._closed = threading.Event()
        self._executors = {name: ThreadPoolExecutor(max_workers=1) for name in self._models}
        self._last: dict[str, float] | None = None
        self._pending: tuple[Schedule, dict[str, TRTEngine]] | None = None
        self._search: _Search | None = None
        self._retired: list[_Search] = []
        self._active: list[str] = []
        self._scheduler: HaxConnScheduler | None = None
        self._schedule: Schedule | None = None
        self._engines: dict[str, TRTEngine] = {}
        self.set_active(first)

    @property
    def active(self: Self) -> list[str]:
        """The DNNs the next call runs."""
        return list(self._active)

    @property
    def schedule(self: Self) -> Schedule:
        """The schedule the next call runs."""
        if self._schedule is None:
            err_msg = "HaX-CoNN runtime has no schedule"
            raise RuntimeError(err_msg)
        return self._schedule

    @property
    def scheduler(self: Self) -> HaxConnScheduler:
        """The scheduler of the active DNNs."""
        if self._scheduler is None:
            err_msg = "HaX-CoNN runtime has no scheduler"
            raise RuntimeError(err_msg)
        return self._scheduler

    @property
    def schedules(self: Self) -> dict[frozenset[str], Schedule]:
        """The best schedule found for each set of DNNs that ran."""
        with self._lock:
            return dict(self._schedules)

    @property
    def searching(self: Self) -> bool:
        """Whether a background search for the active DNNs is still running."""
        with self._lock:
            return self._search is not None and not self._search.finished

    @property
    def last(self: Self) -> dict[str, float] | None:
        """The latency of each DNN in the last call, in milliseconds."""
        return self._last

    def set_active(self: Self, names: Iterable[str]) -> Schedule:
        """
        Change the DNNs that run together.

        Any background search for the previous DNNs stops. The new set starts on the best
        schedule found for it before; otherwise it is solved (static mode) or starts on
        the best naive schedule while a new background search runs (dynamic mode). The
        first schedule's engines are built and loaded before this returns.

        Parameters
        ----------
        names : Iterable[str]
            The DNNs to run, from the runtime's models.

        Returns
        -------
        Schedule
            The schedule the next call runs.

        Raises
        ------
        ValueError
            If ``names`` is empty or names an unknown DNN.
        RuntimeError
            If the runtime was closed.

        """
        if self._closed.is_set():
            err_msg = "HaX-CoNN runtime is closed"
            raise RuntimeError(err_msg)
        chosen = list(dict.fromkeys(names))
        unknown = sorted(set(chosen) - set(self._models))
        if len(chosen) == 0 or unknown:
            err_msg = f"active DNNs must be a non-empty subset of {sorted(self._models)}"
            raise ValueError(err_msg)
        self._check_background()
        self._retire()
        key = frozenset(chosen)
        scheduler = self._new_scheduler(key)
        known = self._schedules.get(key)
        iterator: Iterator[Schedule] | None = None
        if self._dynamic and (known is None or not known.optimal):
            iterator = scheduler.solve_iter(start=known)
            schedule = next(iterator)
        elif known is not None:
            schedule = known
        else:
            schedule = scheduler.solve()
        engines = self._engines_for(schedule)
        search = _Search(key, scheduler, schedule) if iterator is not None else None
        with self._lock:
            self._schedules[key] = schedule
            self._active = chosen
            self._scheduler = scheduler
            self._schedule = schedule
            self._engines = engines
            self._pending = None
            self._search = search
        if search is not None and iterator is not None:
            search.futures = [
                search.executor.submit(self._solve, search, iterator),
                search.executor.submit(self._follow, search),
            ]
        if self._verbose:
            LOG.info(f"HaX-CoNN: running {chosen}, predicted {schedule.objective:.3f}")
        return schedule

    def execute(self: Self, inputs: Mapping[str, EngineInputs]) -> dict[str, list[np.ndarray]]:
        """
        Run every active DNN once, all started together.

        Parameters
        ----------
        inputs : Mapping[str, EngineInputs]
            The inputs of each active DNN, as for :meth:`TRTEngine.execute`.

        Returns
        -------
        dict[str, list[np.ndarray]]
            The outputs of each active DNN.

        Raises
        ------
        ValueError
            If an active DNN has no inputs.
        RuntimeError
            If the runtime was closed.

        """
        if self._closed.is_set():
            err_msg = "HaX-CoNN runtime is closed"
            raise RuntimeError(err_msg)
        self._admit()
        missing = sorted(set(self._engines) - set(inputs))
        if missing:
            err_msg = f"inputs are missing {missing}"
            raise ValueError(err_msg)
        barrier = threading.Barrier(len(self._engines))
        # nothing captures a CUDA graph while the DNNs run
        with TRTEngine._capture_lock:  # noqa: SLF001
            futures = {
                name: self._executors[name].submit(_run, engine, inputs[name], barrier)
                for name, engine in self._engines.items()
            }
            wait(futures.values())
        results = {name: future.result() for name, future in futures.items()}
        self._last = {name: ms for name, (_, ms) in results.items()}
        return {name: outputs for name, (outputs, _) in results.items()}

    def mock_execute(self: Self) -> dict[str, list[np.ndarray]]:
        """
        Run every active DNN once on random inputs, all started together.

        Returns
        -------
        dict[str, list[np.ndarray]]
            The outputs of each active DNN.

        """
        self._admit()
        return self.execute({name: e.get_random_input() for name, e in self._engines.items()})

    def close(self: Self) -> None:
        """
        Stop background solving and release every engine and worker.

        A running Z3 check is interrupted; an engine build in progress is waited for,
        since TensorRT cannot stop one.
        """
        self._closed.set()
        self._retire()
        for search in self._retired:
            search.executor.shutdown(wait=True)
        self._retired.clear()
        for executor in self._executors.values():
            executor.shutdown(wait=True)
        with self._lock:
            self._pending = None
            self._engines = {}
        with self._cache_lock:
            self._cache.clear()
        gc.collect()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _new_scheduler(self: Self, names: frozenset[str]) -> HaxConnScheduler:
        # a scheduler per search, so stopping one never touches another
        traits = {name: self._traits[name] for name in self._models if name in names}
        pause = _SOLVER_PAUSE_S if self._dynamic else 0.0
        return HaxConnScheduler(traits, self._contention, self._config, pause=pause)

    def _engines_for(self: Self, schedule: Schedule) -> dict[str, TRTEngine]:
        names = list(schedule.assignments)
        paths = build_engines(
            {name: self._models[name] for name in names},
            {name: self._traits[name] for name in names},
            schedule,
            self._engine_dir,
            dla_core=self._dla_core,
            workspace=self._workspace,
            timing_cache=self._timing_cache,
            data_batchers={n: b for n, b in self._data_batchers.items() if n in names},
            verbose=self._verbose,
        )
        engines: dict[str, TRTEngine] = {}
        for name, path in paths.items():
            with self._cache_lock:
                engine = self._cache.get(path)
                if engine is not None:
                    self._cache.move_to_end(path)
            if engine is None:
                engine = TRTEngine(
                    path,
                    warmup_iterations=_WARMUP_ITERATIONS,
                    dla_core=self._dla_core if "dla" in schedule.assignments[name] else None,
                    cuda_graph=False,
                    warmup=True,
                )
                with self._cache_lock:
                    self._cache[path] = engine
                    # an evicted engine still in use lives on until it is switched out
                    while len(self._cache) > self._max_engines:
                        self._cache.popitem(last=False)
            engines[name] = engine
        return engines

    def _ready(self: Self, schedule: Schedule) -> bool:
        names = list(schedule.assignments)
        paths = engine_paths(
            {name: self._models[name] for name in names},
            {name: self._traits[name] for name in names},
            schedule,
            self._engine_dir,
            {n: b for n, b in self._data_batchers.items() if n in names},
        )
        return all(path.exists() for path in paths.values())

    def _solve(self: Self, search: _Search, iterator: Iterator[Schedule]) -> None:
        if self._solver_core is not None:
            # pid 0 is the calling thread on Linux
            os.sched_setaffinity(0, {self._solver_core})
        for schedule in iterator:
            if search.stop.is_set():
                return
            with self._lock:
                search.history.append(schedule)
                self._schedules[search.names] = schedule
            search.improved.set()
        with self._lock:
            search.finished = True
        search.improved.set()

    def _follow(self: Self, search: _Search) -> None:
        # switch to the best schedule so far at each update time, then on every improvement
        update_ms = self._config.update_ms
        updates = list(update_ms) if update_ms is not None else []
        loaded = search.history[0].assignments
        while True:
            if updates:
                delay = search.started + updates.pop(0) / 1000.0 - time.monotonic()
                if search.stop.wait(max(delay, 0.0)):
                    return
            else:
                search.improved.wait()
                search.improved.clear()
                if search.stop.is_set():
                    return
            with self._lock:
                finished = search.finished
                history = list(search.history)
            target = history[-1]
            if self._build == "prebuilt":
                target = next((s for s in reversed(history) if self._ready(s)), history[0])
            if target.assignments == loaded:
                self._renew(search, target)
            else:
                engines = self._engines_for(target)
                if search.stop.is_set():
                    return
                with self._lock:
                    if self._search is search:
                        self._pending = (target, engines)
                loaded = target.assignments
                if self._verbose:
                    LOG.info(f"HaX-CoNN: schedule ready, predicted {target.objective:.3f}")
            if finished:
                return

    def _renew(self: Self, search: _Search, schedule: Schedule) -> None:
        # same engines (e.g. the last schedule, now proven optimal): new metadata
        with self._lock:
            if self._search is not search:
                return
            if self._pending is not None:
                self._pending = (schedule, self._pending[1])
            else:
                self._schedule = schedule

    def _retire(self: Self) -> None:
        with self._lock:
            search, self._search = self._search, None
            self._pending = None
        if search is None:
            return
        search.stop.set()
        search.improved.set()
        search.scheduler.interrupt()
        search.executor.shutdown(wait=False)
        self._retired.append(search)

    def _check_background(self: Self) -> None:
        # a failed background search raises here, once
        searches = [s for s in (self._search, *self._retired) if s is not None]
        for search in searches:
            failed = [f for f in search.futures if f.done() and f.exception() is not None]
            if failed:
                search.futures = [f for f in search.futures if f not in failed]
                failed[0].result()
        self._retired = [s for s in self._retired if not all(f.done() for f in s.futures)]

    def _admit(self: Self) -> None:
        self._check_background()
        with self._lock:
            pending, self._pending = self._pending, None
            if pending is not None:
                self._schedule, self._engines = pending
        if pending is not None and self._verbose:
            LOG.info(f"HaX-CoNN: switched to {self.schedule.assignments}")
