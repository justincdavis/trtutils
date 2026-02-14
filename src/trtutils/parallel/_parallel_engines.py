# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import nvtx

from trtutils._flags import FLAGS

from ._queued_engine import QueuedTRTEngine

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self

    from trtutils._engine import TRTEngine


class ParallelTRTEngines:
    """Handle many TRTEngines in parallel."""

    def __init__(
        self: Self,
        engines: Sequence[TRTEngine | Path | str | tuple[TRTEngine | Path | str, int]],
        warmup_iterations: int = 5,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Create a ParallelTRTEngines instance.

        Parameters
        ----------
        engines : Sequence[TRTEngine | Path | str | tuple[TRTEngine | Path | str, int]]
            The Paths to the compiled engines to use.
        warmup_iterations : int
            The number of iterations to perform warmup for.
            By default 5
        warmup : bool, optional
            Whether or not to run warmup iterations on the engines.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("parallel_engines::init")
        self._engines: list[QueuedTRTEngine] = []
        for engine_info in engines:
            engine: TRTEngine | Path | str
            dla_core: int | None = None
            if isinstance(engine_info, tuple):
                engine, dla_core = engine_info  # type: ignore[assignment]
            else:
                engine = engine_info
            q_engine = QueuedTRTEngine(
                engine=engine,
                warmup_iterations=warmup_iterations,
                warmup=warmup,
                dla_core=dla_core,
            )
            self._engines.append(q_engine)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    def get_random_input(
        self: Self,
        *,
        new: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Get a random input to the underlying TRTEngines.

        Parameters
        ----------
        new : bool, optional
            Whether or not to get a new input or the cached already generated one.
            By default, None/False

        Returns
        -------
        list[list[np.ndarray]]
            The random inputs.

        """
        return [e.get_random_input(new=new) for e in self._engines]

    def stop(self: Self) -> None:
        """Stop the underlying engine threads."""
        for engine in self._engines:
            engine.stop()

    def submit(
        self: Self,
        inputs: list[list[np.ndarray]],
    ) -> None:
        """
        Submit data to be processed by the engines.

        Parameters
        ----------
        inputs : list[list[np.ndarray]]
            The inputs to pass to the engines.
            Should be a list of the same lenght of engines created.

        Raises
        ------
        ValueError
            If the inputs are not the same size as the engines.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("parallel_engines::submit")
        if len(inputs) != len(self._engines):
            err_msg = f"Cannot match {len(inputs)} inputs to {len(self._engines)} engines."
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            raise ValueError(err_msg)
        for data, engine in zip(inputs, self._engines):
            engine.submit(data)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    def mock_submit(
        self: Self,
    ) -> None:
        """Send random data to the engines."""
        for engine in self._engines:
            engine.mock_submit()

    def retrieve(
        self: Self,
        timeout: float | None = None,
    ) -> list[list[np.ndarray] | None]:
        """
        Get the outputs from the engines.

        Parameters
        ----------
        timeout : float, optional
            Timeout for waiting for data.

        Returns
        -------
        list[np.ndarray]
            The output from the engines.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range("parallel_engines::retrieve")
        results = [engine.retrieve(timeout=timeout) for engine in self._engines]
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return results
