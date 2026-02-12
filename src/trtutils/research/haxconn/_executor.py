# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Concurrent multi-engine executor for HaX-CoNN."""

from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from trtutils._engine import TRTEngine
from trtutils._log import LOG

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


class HaxconnExecutor:
    """
    Concurrent multi-engine execution wrapper for HaX-CoNN.

    Loads multiple TensorRT engines and executes them concurrently using
    a thread pool. Each engine has its own CUDA stream, enabling true
    GPU+DLA parallelism on Jetson platforms.

    Parameters
    ----------
    engine_paths : list[Path | str]
        Paths to the serialized TensorRT engine files.
    dla_core : int, optional
        DLA core to use for DLA-enabled engines. Default 0.
    warmup : bool | None, optional
        Whether to do warmup iterations. Default None (no warmup).
    warmup_iterations : int, optional
        Number of warmup iterations. Default 5.
    verbose : bool | None, optional
        Whether to print verbose output.

    """

    def __init__(
        self: Self,
        engine_paths: list[Path | str],
        dla_core: int = 0,
        *,
        warmup: bool | None = None,
        warmup_iterations: int = 5,
        verbose: bool | None = None,
    ) -> None:
        self._engines: list[TRTEngine] = []
        self._verbose = verbose if verbose is not None else False

        for i, path in enumerate(engine_paths):
            if self._verbose:
                LOG.info(f"Loading engine {i}: {path}")

            engine = TRTEngine(
                engine_path=path,
                warmup_iterations=warmup_iterations,
                dla_core=dla_core,
                warmup=warmup,
                cuda_graph=False,
                verbose=verbose,
            )
            self._engines.append(engine)

        self._pool = ThreadPoolExecutor(max_workers=len(self._engines))

        if self._verbose:
            LOG.info(f"HaxconnExecutor ready with {len(self._engines)} engines")

    @property
    def num_models(self: Self) -> int:
        """Number of loaded models."""
        return len(self._engines)

    @property
    def input_specs(self: Self) -> list[list[tuple[list[int], np.dtype]]]:
        """
        Input specs for each model.

        Returns
        -------
        list[list[tuple[list[int], np.dtype]]]
            One list of (shape, dtype) tuples per model.

        """
        return [engine.input_spec for engine in self._engines]

    @property
    def output_specs(self: Self) -> list[list[tuple[list[int], np.dtype]]]:
        """
        Output specs for each model.

        Returns
        -------
        list[list[tuple[list[int], np.dtype]]]
            One list of (shape, dtype) tuples per model.

        """
        return [engine.output_spec for engine in self._engines]

    def execute(
        self: Self,
        data: list[list[np.ndarray]],
    ) -> tuple[list[np.ndarray], ...]:
        """
        Execute all models concurrently with the given inputs.

        Parameters
        ----------
        data : list[list[np.ndarray]]
            One input list per model. ``data[i]`` is the input list for engine ``i``.

        Returns
        -------
        tuple[list[np.ndarray], ...]
            One output list per model. ``result[i]`` is the output list from engine ``i``.

        Raises
        ------
        ValueError
            If the number of input lists doesn't match the number of engines.

        """
        if len(data) != len(self._engines):
            err_msg = f"Expected {len(self._engines)} input lists, got {len(data)}"
            raise ValueError(err_msg)

        # Submit all executions concurrently
        futures = []
        for engine, engine_data in zip(self._engines, data):
            future = self._pool.submit(engine.execute, engine_data)
            futures.append(future)

        # Collect results in order
        return tuple(future.result() for future in futures)

    def __call__(
        self: Self,
        data: list[list[np.ndarray]],
    ) -> tuple[list[np.ndarray], ...]:
        """Execute all models concurrently. Alias for ``execute``."""
        return self.execute(data)

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError):
            self._pool.shutdown(wait=False)
        with contextlib.suppress(AttributeError):
            for engine in self._engines:
                del engine
