# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""SHIFT runtime: run the scheduled model per frame and load models on demand."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable

import numpy as np

from trtutils._engine import TRTEngine
from trtutils._log import LOG

from ._scheduler import ShiftScheduler

if TYPE_CHECKING:
    from collections.abc import Mapping
    from concurrent.futures import Future

    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface
    from trtutils.image.types import Detection

    from ._types import ModelTraits, ShiftConfig


class Shift:
    """
    Context-aware multi-model object detection.

    Runs one (model, accelerator) pair per frame and lets :class:`ShiftScheduler`
    pick the pair for the next frame. Without a memory budget every pair is built
    up front. With one, pairs are built on demand in a background thread while the
    current pair keeps serving frames (the paper's Dynamic Model Loader); the
    least recently requested pairs are evicted to stay within the budget, and the
    switch happens on the first frame after the requested pair finishes loading.

    Detectors are built and used from the thread that calls :meth:`end2end`,
    except for background loads, which are handed over only once built. Call
    :meth:`close` when done to stop background loading and release the pairs.
    """

    def __init__(
        self: Self,
        models: Mapping[str, Callable[[], DetectorInterface]],
        traits: Mapping[str, ModelTraits],
        config: ShiftConfig | None = None,
        memory_budget: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Create the SHIFT runtime.

        Parameters
        ----------
        models : Mapping[str, Callable[[], DetectorInterface]]
            A factory per pair name that builds the detector, e.g.
            ``{"yolov10n_dla": lambda: Detector("b.engine", dla_core=0, cuda_graph=False)}``.
        traits : Mapping[str, ModelTraits]
            The characterization of the same pairs, from :func:`characterize`.
        config : ShiftConfig, optional
            The scheduler hyperparameters. Default is the paper's values.
        memory_budget : float, optional
            Device memory in MB the loaded pairs may use, based on their
            characterized ``load_memory_mb``. Default is None: build every pair up front.
        verbose : bool, optional
            Whether to log loads, evictions and switches. Default is None (False).

        Raises
        ------
        ValueError
            If ``models`` and ``traits`` name different pairs, or the budget cannot
            hold the two largest pairs at once.

        """
        if set(models) != set(traits):
            err_msg = f"models {sorted(models)} and traits {sorted(traits)} must name the same pairs"
            raise ValueError(err_msg)
        self._scheduler = ShiftScheduler(traits, config)
        self._factories = dict(models)
        self._memory = {name: t.load_memory_mb for name, t in traits.items()}
        self._budget = memory_budget
        self._verbose = verbose
        if memory_budget is not None:
            largest = sorted(self._memory.values(), reverse=True)[:2]
            if memory_budget < sum(largest):
                err_msg = (
                    f"memory_budget of {memory_budget} MB cannot hold the two largest pairs "
                    f"({sum(largest):.1f} MB)"
                )
                raise ValueError(err_msg)

        # start on the most accurate pair
        metric = self._scheduler.config.accuracy_metric
        accuracy = {name: float(np.mean(t.frame_accuracies(metric))) for name, t in traits.items()}
        self._current = max(accuracy, key=lambda name: accuracy[name])
        self._target = self._current
        self._last: str | None = None

        # loaded detectors, least recently requested first
        self._loaded: OrderedDict[str, DetectorInterface] = OrderedDict()
        self._executor: ThreadPoolExecutor | None = None
        self._pending: tuple[str, Future[DetectorInterface]] | None = None
        if memory_budget is None:
            for name, factory in self._factories.items():
                self._loaded[name] = factory()
        else:
            self._loaded[self._current] = self._factories[self._current]()
            self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def current(self: Self) -> str:
        """The pair that runs the next frame."""
        return self._current

    @property
    def last(self: Self) -> str | None:
        """The pair that ran the last frame, None before the first frame."""
        return self._last

    @property
    def loaded(self: Self) -> list[str]:
        """The loaded pairs, least recently requested first."""
        return list(self._loaded)

    @property
    def scheduler(self: Self) -> ShiftScheduler:
        """The scheduler."""
        return self._scheduler

    def end2end(self: Self, image: np.ndarray) -> list[Detection]:
        """
        Detect objects in one frame and schedule the pair for the next frame.

        Parameters
        ----------
        image : np.ndarray
            The frame, BGR.

        Returns
        -------
        list[Detection]
            The detections of the pair that ran this frame.

        """
        self._admit()
        self._switch()
        self._last = self._current
        detections = self._loaded[self._current].end2end(image)
        self._target = self._scheduler(self._current, image, detections)
        self._switch()
        return detections

    def reset(self: Self) -> None:
        """Reset the scheduler state; loaded pairs stay loaded."""
        self._scheduler.reset()
        self._target = self._current

    def close(self: Self) -> None:
        """Stop background loading and release every loaded pair."""
        if self._pending is not None:
            self._pending[1].cancel()
            self._pending = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._loaded.clear()

    def _switch(self: Self) -> None:
        # move to the target once it is loaded, else (re)request it; touching it keeps it
        # off the eviction list
        if self._target not in self._loaded:
            self._load(self._target)
            return
        if self._verbose and self._target != self._current:
            LOG.info(f"SHIFT: switch {self._current} -> {self._target}")
        self._current = self._target
        self._loaded.move_to_end(self._current)

    def _admit(self: Self) -> None:
        # take a finished background load; a failed load raises here, once
        if self._pending is None or not self._pending[1].done():
            return
        name, future = self._pending
        self._pending = None
        self._loaded[name] = future.result()
        if self._verbose:
            LOG.info(f"SHIFT: loaded {name}")

    def _build(self: Self, name: str) -> DetectorInterface:
        # building an engine uses the legacy default stream, which breaks any CUDA graph
        # capture running in the serving thread, so hold the capture lock for the build
        with TRTEngine._capture_lock:  # noqa: SLF001
            return self._factories[name]()

    def _load(self: Self, name: str) -> None:
        if self._executor is None or self._budget is None or self._pending is not None:
            return
        # evict least recently requested pairs, never the current one
        used = sum(self._memory[n] for n in self._loaded)
        for victim in list(self._loaded):
            if used + self._memory[name] <= self._budget:
                break
            if victim == self._current:
                continue
            del self._loaded[victim]
            used -= self._memory[victim]
            if self._verbose:
                LOG.info(f"SHIFT: evicted {victim}")
        self._pending = (name, self._executor.submit(self._build, name))
