# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING

import nvtx

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


class QueuedTRTEngine:
    """Interact with TRTEngine over Thread and Queue."""

    def __init__(
        self: Self,
        engine: TRTEngine | Path | str,
        warmup_iterations: int = 5,
        dla_core: int | None = None,
        device: int | None = None,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Create a QueuedTRTEngine.

        Parameters
        ----------
        engine : Path, str
            The Path to the compiled TensorRT engine.
        warmup_iterations : int
            The number of iterations to warmup the engine.
            By default 5
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this engine. Default is None,
            which uses the current device.
        warmup : bool, optional
            Whether or not to perform warmup iterations.

        """
        self._stopped = False  # flag for if user stopped thread
        self._engine: TRTEngine
        if isinstance(engine, TRTEngine):
            self._engine = engine
        else:
            self._engine = TRTEngine(
                engine_path=engine,
                warmup_iterations=warmup_iterations,
                warmup=warmup,
                dla_core=dla_core,
                device=device,
            )
        self._nvtx_tags = {
            "init": f"queued_engine::init [{self._engine.name}]",
            "_run": f"queued_engine::_run [{self._engine.name}]",
            "submit": f"queued_engine::submit [{self._engine.name}]",
            "retrieve": f"queued_engine::retrieve [{self._engine.name}]",
        }
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["init"])
        self._input_queue: Queue[list[np.ndarray]] = Queue()
        self._output_queue: Queue[list[np.ndarray]] = Queue()
        self._thread = Thread(
            target=self._run,
            args=(),
            daemon=True,
        )
        self._thread.start()
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    def __del__(self: Self) -> None:
        self.stop()

    @property
    def input_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each input tensor.

        """
        return self._engine.input_spec

    @property
    def input_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the input tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each input tensor.

        """
        return self._engine.input_shapes

    @property
    def input_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the input tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each input tensor.

        """
        return self._engine.input_dtypes

    @property
    def output_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each output tensor.

        """
        return self._engine.output_spec

    @property
    def output_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the output tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each output tensor.

        """
        return self._engine.output_shapes

    @property
    def output_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the output tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each output tensor.

        """
        return self._engine.output_dtypes

    def get_random_input(self: Self, *, new: bool | None = None) -> list[np.ndarray]:
        """
        Get a random input to the underlying TRTEngine.

        Parameters
        ----------
        new : bool, optional
            Whether or not to get a new input or the cached already generated one.
            By default, None/False

        Returns
        -------
        list[np.ndarray]
            The random input.

        """
        return self._engine.get_random_input(new=new)

    def stop(
        self: Self,
    ) -> None:
        """Stop the thread containing the TRTEngine."""
        self._stopped = True
        self._thread.join()

    def submit(
        self: Self,
        data: list[np.ndarray],
    ) -> None:
        """
        Put data in the input queue.

        Parameters
        ----------
        data : list[np.ndarray]
            The data to have the engine run.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["submit"])
        self._input_queue.put(data)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()

    def mock_submit(
        self: Self,
    ) -> None:
        """Send a random input to the engine."""
        data = self._engine.get_random_input()
        self._input_queue.put(data)

    def retrieve(
        self: Self,
        timeout: float | None = None,
    ) -> list[np.ndarray] | None:
        """
        Get an output from the engine thread.

        Parameters
        ----------
        timeout : float, optional
            Timeout for waiting for data.

        Returns
        -------
        list[np.ndarray]
            The output from the engine.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["retrieve"])
        with contextlib.suppress(Empty):
            result = self._output_queue.get(timeout=timeout)
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
            return result
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()
        return None

    def _run(
        self: Self,
    ) -> None:
        while not self._stopped:
            try:
                inputs = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            if FLAGS.NVTX_ENABLED:
                nvtx.push_range(self._nvtx_tags["_run"])
            result = self._engine(inputs)

            self._output_queue.put(result)
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()
