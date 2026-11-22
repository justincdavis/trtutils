# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""HaX-CoNN runtime: run DNNs concurrently on the schedule's mixed GPU/DLA engines."""

from __future__ import annotations

import gc
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING

from trtutils._engine import TRTEngine
from trtutils._log import LOG

from ._build import build_engines
from ._scheduler import HaxConnScheduler

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from concurrent.futures import Future

    import numpy as np
    from typing_extensions import Self

    from trtutils.builder._batcher import AbstractBatcher
    from trtutils.core._interface import EngineInputs

    from ._pccs import PCCSModel
    from ._types import HaxConnConfig, ModelTraits, Schedule

_BARRIER_TIMEOUT_S = 60.0
_WARMUP_ITERATIONS = 5


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


class HaxConn:
    """
    Concurrent DNN execution across a GPU and a DLA on a HaX-CoNN schedule.

    Each DNN runs as one engine whose layer groups sit on the accelerators the schedule
    picked. :meth:`execute` starts every DNN at the same moment, each on its own worker
    thread, and returns when all are done. Engines are built with :func:`build_engines`
    and kept in ``engine_dir``, so schedules seen before load without a build.

    With ``dynamic`` (D-HaX-CoNN), execution starts right away on the best naive
    schedule while a background thread keeps solving; every better schedule is built
    and loaded in the background and swapped in between two calls. Builds compete
    with the running DNNs for the GPU, so a warm ``engine_dir`` makes switches cheap.

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
        dynamic: bool = False,
        verbose: bool | None = None,
    ) -> None:
        """
        Create the HaX-CoNN runtime.

        Parameters
        ----------
        models : Mapping[str, Path | str]
            The ONNX file of each DNN, keyed by name.
        traits : Mapping[str, ModelTraits]
            The traits of the same DNNs, from :func:`characterize`.
        contention : Mapping[str, PCCSModel], optional
            The contention model of each accelerator. Default is None, only valid with
            contention "none".
        config : HaxConnConfig, optional
            The hyperparameters. Default is the paper's values.
        schedule : Schedule, optional
            The schedule to run. Default is None: solve for it (or, with ``dynamic``,
            start on the best naive schedule).
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
        dynamic : bool, optional
            Whether to keep solving in the background and switch to better schedules.
            Default is False.
        verbose : bool, optional
            Whether to log builds and switches. Default is None (False).

        Raises
        ------
        ValueError
            If ``models`` and ``traits`` name different DNNs, the traits differ in
            precision, a schedule does not fit the traits' layer groups, or a schedule
            is given together with ``dynamic``.

        """
        if set(models) != set(traits):
            err_msg = f"models {sorted(models)} and traits {sorted(traits)} must name the same DNNs"
            raise ValueError(err_msg)
        if len({t.precision for t in traits.values()}) > 1:
            err_msg = "every DNN's traits must be measured in the same precision"
            raise ValueError(err_msg)
        if dynamic and schedule is not None:
            err_msg = "dynamic mode finds its own schedules; do not pass one"
            raise ValueError(err_msg)
        self._models = dict(models)
        self._traits = dict(traits)
        self._scheduler = HaxConnScheduler(traits, contention, config)
        self._tmp = tempfile.TemporaryDirectory() if engine_dir is None else None
        self._engine_dir = Path(self._tmp.name if self._tmp is not None else str(engine_dir))
        self._dla_core = dla_core
        self._workspace = workspace
        self._timing_cache = timing_cache
        self._data_batchers = data_batchers
        self._verbose = verbose
        self._last: dict[str, float] | None = None
        self._executors = {name: ThreadPoolExecutor(max_workers=1) for name in self._models}

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: tuple[Schedule, dict[str, TRTEngine]] | None = None
        self._search: Future[None] | None = None
        self._background: ThreadPoolExecutor | None = None
        iterator = self._scheduler.solve_iter() if dynamic else None
        if iterator is not None:
            schedule = next(iterator)
        elif schedule is None:
            schedule = self._scheduler.solve()
        else:
            # a schedule from another characterization must not build silently
            self._scheduler.evaluate(schedule.assignments)
        self._schedule = schedule
        self._engines = self._load(schedule)
        if iterator is not None:
            self._background = ThreadPoolExecutor(max_workers=1)
            self._search = self._background.submit(self._keep_solving, iterator)

    @property
    def schedule(self: Self) -> Schedule:
        """The schedule the next call runs."""
        return self._schedule

    @property
    def scheduler(self: Self) -> HaxConnScheduler:
        """The scheduler."""
        return self._scheduler

    @property
    def last(self: Self) -> dict[str, float] | None:
        """The latency of each DNN in the last call, in milliseconds."""
        return self._last

    def execute(self: Self, inputs: Mapping[str, EngineInputs]) -> dict[str, list[np.ndarray]]:
        """
        Run every DNN once, all started together.

        Parameters
        ----------
        inputs : Mapping[str, EngineInputs]
            The inputs of each DNN, as for :meth:`TRTEngine.execute`.

        Returns
        -------
        dict[str, list[np.ndarray]]
            The outputs of each DNN.

        Raises
        ------
        ValueError
            If a DNN has no inputs.
        RuntimeError
            If the runtime was closed.

        """
        if self._stop.is_set():
            err_msg = "HaX-CoNN runtime is closed"
            raise RuntimeError(err_msg)
        missing = sorted(set(self._engines) - set(inputs))
        if missing:
            err_msg = f"inputs are missing {missing}"
            raise ValueError(err_msg)
        self._admit()
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
        Run every DNN once on random inputs, all started together.

        Returns
        -------
        dict[str, list[np.ndarray]]
            The outputs of each DNN.

        """
        return self.execute({name: e.get_random_input() for name, e in self._engines.items()})

    def close(self: Self) -> None:
        """
        Stop background solving and release every engine and worker.

        A running Z3 check is interrupted; an engine build in progress is waited for,
        since TensorRT cannot stop one.
        """
        self._stop.set()
        if self._background is not None:
            self._scheduler.interrupt()
            self._background.shutdown(wait=True)
            self._background = None
        for executor in self._executors.values():
            executor.shutdown(wait=True)
        with self._lock:
            self._pending = None
        self._engines.clear()
        gc.collect()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _load(self: Self, schedule: Schedule) -> dict[str, TRTEngine]:
        paths = build_engines(
            self._models,
            self._traits,
            schedule,
            self._engine_dir,
            dla_core=self._dla_core,
            workspace=self._workspace,
            timing_cache=self._timing_cache,
            data_batchers=self._data_batchers,
            verbose=self._verbose,
        )
        return {
            name: TRTEngine(
                path,
                warmup_iterations=_WARMUP_ITERATIONS,
                dla_core=self._dla_core if "dla" in schedule.assignments[name] else None,
                cuda_graph=False,
                warmup=True,
            )
            for name, path in paths.items()
        }

    def _keep_solving(self: Self, iterator: Iterator[Schedule]) -> None:
        loaded = self._schedule.assignments
        for schedule in iterator:
            if self._stop.is_set():
                return
            if schedule.assignments == loaded:
                # same engines (e.g. the last schedule, now proven optimal): new metadata
                with self._lock:
                    if self._pending is not None:
                        self._pending = (schedule, self._pending[1])
                    else:
                        self._schedule = schedule
                continue
            engines = self._load(schedule)
            if self._stop.is_set():
                return
            loaded = schedule.assignments
            with self._lock:
                self._pending = (schedule, engines)
            if self._verbose:
                LOG.info(f"HaX-CoNN: schedule ready, predicted {schedule.objective:.3f}")

    def _admit(self: Self) -> None:
        # a failed background search raises here, once
        if self._search is not None and self._search.done():
            search, self._search = self._search, None
            search.result()
        with self._lock:
            pending, self._pending = self._pending, None
            if pending is not None:
                self._schedule, self._engines = pending
        if pending is None:
            return
        gc.collect()
        if self._verbose:
            LOG.info(f"HaX-CoNN: switched to {self._schedule.assignments}")
